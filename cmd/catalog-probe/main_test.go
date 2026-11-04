// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"encoding/json"
	"maps"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// TestRunTriesOnceMore: a call that didn't work is sent once more and the
// second answer stands, so one flaky answer doesn't reach the PR; a call
// that worked is sent once. The results file holds every probe, skips too.
func TestRunTriesOnceMore(t *testing.T) {
	t.Parallel()

	calls := map[string]int{}
	h, _ := fakeHosts(t, func(_ *http.Request, body map[string]any) (int, string) {
		model, _ := body["model"].(string)
		calls[model]++

		switch {
		case model == "claude-gone":
			return 404, `{"type": "error", "error": {"type": "not_found_error", "message": "model: claude-gone"}}`
		case model == "claude-flaky" && calls[model] == 1:
			return 529, `{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}`
		}

		return 200, `{}`
	})

	const messages = "anthropic_messages/direct"

	dir := vendorDir(t, offerings{}, offerings{
		"anthropic":  {offering("claude-flaky", messages), offering("claude-gone", messages), offering("claude-ok", messages)},
		"gcp.vertex": {offering("claude-ok@20261101", "anthropic_messages/vertex")},
	})
	out := t.TempDir()

	require.NoError(t, run(t.Context(), h, []string{dir}, out, 0))

	assert.Equal(t, map[string]int{"claude-flaky": 2, "claude-gone": 2, "claude-ok": 1}, calls)

	raw, err := os.ReadFile(filepath.Join(out, "probe-results.json"))
	require.NoError(t, err)

	var results struct {
		Probes []Probe `json:"probes"`
	}
	require.NoError(t, json.Unmarshal(raw, &results))

	got := make([]string, 0, len(results.Probes))
	for _, p := range results.Probes {
		got = append(got, p.Provider+" "+p.ID+" "+p.Outcome+" "+p.Detail+" "+p.RequestID)
	}

	assert.Equal(t, []string{
		"anthropic claude-flaky callable 200 req_anthropic",
		"anthropic claude-gone not_found 404 not_found_error req_anthropic",
		"anthropic claude-ok callable 200 req_anthropic",
		"vertex claude-ok@20261101 skipped no CI credentials for this host yet ",
	}, got)

	_, err = os.Stat(filepath.Join(out, "probes.md"))
	assert.NoError(t, err)
}

// TestMarkdownShowsEachProbe: the PR body's block puts what didn't work
// first and counts it, shows each host's answer in a few checked words, and
// names the region a Bedrock copy was called from.
func TestMarkdownShowsEachProbe(t *testing.T) {
	t.Parallel()

	block := markdown([]Probe{
		{Provider: "anthropic", ID: "claude-ok", Outcome: outcomeCallable, Detail: "200", RequestID: "req_1"},
		{Provider: "anthropic", ID: "claude-gone", Outcome: outcomeNotFound, Detail: "404 not_found_error", RequestID: "req_2"},
		{Provider: "bedrock", ID: "us.anthropic.claude-z", Region: "us-east-1", Outcome: outcomeNotSubscribed, Detail: "entitlement NOT_AVAILABLE"},
		{Provider: "vertex", ID: "claude-z@20261101", Outcome: outcomeSkipped, Detail: "no Vertex credentials in CI yet"},
		{Provider: "anthropic", Outcome: outcomeSkipped, Detail: "its ID doesn't fit the offering pattern, so it isn't shown or called"},
	})

	assert.Equal(t, `## Live checks

Before this PR opened, the run called each offering the PR adds or changes once on its host, with a tiny request. The calls ran the workflow's own code, not this PR's. A call that didn't work was tried a second time. Read any row that didn't work before approving, because the PR opens either way.

**2 of 3 calls didn't work.**

| Offering | Host | Result | Detail | Request ID |
|---|---|---|---|---|
| `+"`claude-gone`"+` | anthropic | not found | 404 not_found_error | `+"`req_2`"+` |
| `+"`us.anthropic.claude-z`"+` | bedrock (us-east-1) | not subscribed | entitlement NOT_AVAILABLE |  |
| `+"`claude-ok`"+` | anthropic | callable | 200 | `+"`req_1`"+` |
| `+"`claude-z@20261101`"+` | vertex | skipped | no Vertex credentials in CI yet |  |
| (not shown) | anthropic | skipped | its ID doesn't fit the offering pattern, so it isn't shown or called |  |
`, block)

	assert.Contains(t, markdown([]Probe{{Provider: "anthropic", ID: "claude-ok", Outcome: outcomeCallable, Detail: "200"}}), "**The 1 call worked.**")
	assert.Contains(t, markdown([]Probe{{Provider: "vertex", ID: "claude-z", Outcome: outcomeSkipped, Detail: "no Vertex credentials in CI yet"}}), "**Nothing was called.**")
}

// TestWorkflowAllowsEveryHost: the probe job blocks every host off its
// allowlist, so each host this command calls, in each region it calls
// Bedrock from, must be on it; a missing one would only show as "no answer".
func TestWorkflowAllowsEveryHost(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "reconcile-models.yml"))
	require.NoError(t, err)

	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				Uses string            `yaml:"uses"`
				With map[string]string `yaml:"with"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	require.NoError(t, yaml.Unmarshal(raw, &wf))

	steps := wf.Jobs["probe"].Steps
	require.NotEmpty(t, steps)
	allowed := strings.Fields(steps[0].With["allowed-endpoints"])

	h := realHosts()
	targets := make([]string, 0, 3+2*(len(geoRegions)+1))
	targets = append(targets, h.anthropicURL, h.openaiURL, h.geminiURL)

	for _, region := range append(slices.Collect(maps.Values(geoRegions)), bareRegion) {
		targets = append(targets, h.awsURL("bedrock", region), h.awsURL("bedrock-runtime", region))
	}

	for _, target := range targets {
		u, err := url.Parse(target)
		require.NoError(t, err)
		assert.Contains(t, allowed, u.Host+":443", "the probe job can't reach %s", target)
	}
}
