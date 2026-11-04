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
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// offerings is a snapshot's providers, by snapshot provider name.
type offerings map[string][]map[string]any

// vendorDir writes a vendor leg's folder as pr-job.sh leaves it after a
// change: the catalogue before and after the vendor's patch.
func vendorDir(t *testing.T, base, patched offerings) string {
	t.Helper()

	dir := t.TempDir()
	write := func(name string, providers offerings) {
		list := make([]any, 0, len(providers))
		for provider, offered := range providers {
			list = append(list, map[string]any{"provider": provider, "offerings": offered})
		}

		raw, err := json.Marshal(map[string]any{"facts": map[string]any{}, "providers": list})
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(dir, name), raw, 0o600))
	}

	write("base-snapshot.json", base)
	write("snapshot.json", patched)

	return dir
}

// offering is a snapshot offering with one wire API, protocol/transport.
func offering(id, wire string, extra ...any) map[string]any {
	o := map[string]any{"id": id}

	if wire != "" {
		protocol, transport, _ := strings.Cut(wire, "/")
		o["wire_apis"] = []any{map[string]any{"protocol": protocol, "transport": transport}}
	}

	for i := 0; i+1 < len(extra); i += 2 {
		key, _ := extra[i].(string)
		o[key] = extra[i+1]
	}

	return o
}

// planned is what a test compares: each probe's host, ID, region and call,
// or its declared skip.
func planned(probes []Probe) []string {
	out := make([]string, 0, len(probes))
	for _, p := range probes {
		line := p.Provider + " " + p.ID
		if p.Region != "" {
			line += " @" + p.Region
		}

		if p.Outcome == outcomeSkipped {
			line += " skipped: " + p.Detail
		} else {
			line += " via " + string(p.wire.Protocol)
		}

		out = append(out, line)
	}

	return out
}

// TestPlanListsNewAndChangedOfferings: the probes are the offerings a
// vendor's patch adds or changes, in every folder, once each; an offering
// whose only change is a derived value (it follows its neighbours) and one
// the patch leaves alone aren't called.
func TestPlanListsNewAndChangedOfferings(t *testing.T) {
	t.Parallel()

	const messages = "anthropic_messages/direct"

	main := offerings{
		"anthropic": {offering("claude-a", messages, "input", 5), offering("claude-b", messages, "input", 3)},
		"openai":    {offering("gpt-x", "openai_responses/direct", "derived", map[string]any{"price_tier": "low"})},
	}
	afterAnthropic := offerings{
		"anthropic": {offering("claude-a", messages, "input", 6), offering("claude-b", messages, "input", 3), offering("claude-c", messages)},
		"openai":    {offering("gpt-x", "openai_responses/direct", "derived", map[string]any{"price_tier": "high"})},
	}
	afterOpenAI := offerings{
		"anthropic": afterAnthropic["anthropic"],
		"openai":    {afterAnthropic["openai"][0], offering("gpt-y", "openai_chat_completions/direct")},
	}

	probes, err := plan([]string{vendorDir(t, main, afterAnthropic), vendorDir(t, afterAnthropic, afterOpenAI)})
	require.NoError(t, err)

	assert.Equal(t, []string{
		"anthropic claude-a via anthropic_messages",
		"anthropic claude-c via anthropic_messages",
		"openai gpt-y via openai_chat_completions",
	}, planned(probes))
}

// TestPlanRoutesEachHost: each new offering gets the call its host and wire
// API need. A Bedrock copy is called from a region in its geography, a bare
// ID from us-east-1. Offerings no call can reach are listed as skipped with
// the reason: hosts with no CI credentials, global. profiles (CI's account
// denies them), and routes this command has no request for.
func TestPlanRoutesEachHost(t *testing.T) {
	t.Parallel()

	const converse = "bedrock_converse/bedrock_runtime"

	patched := offerings{
		"gcp.gemini": {offering("gemini-9-flash", "gemini_generate_content/direct")},
		"aws.bedrock": {
			offering("us.anthropic.claude-z-v1:0", converse),
			offering("eu.anthropic.claude-z-v1:0", converse),
			offering("au.anthropic.claude-z-v1:0", converse),
			offering("jp.anthropic.claude-z-v1:0", converse),
			offering("in.anthropic.claude-z-v1:0", converse),
			offering("global.anthropic.claude-z-v1:0", converse),
			offering("mistral.large-9", converse),
			offering("openai.gpt-9-luna", "openai_responses/bedrock_mantle"),
			offering("eu.anthropic.claude-y", ""),
		},
		"gcp.vertex": {offering("claude-z@20261101", "anthropic_messages/vertex")},
		"meta":       {offering("muse-9", "openai_responses/direct")},
	}

	probes, err := plan([]string{vendorDir(t, offerings{}, patched)})
	require.NoError(t, err)

	assert.Equal(t, []string{
		"bedrock au.anthropic.claude-z-v1:0 @ap-southeast-2 via bedrock_converse",
		"bedrock eu.anthropic.claude-y skipped: no request for its wire APIs yet",
		"bedrock eu.anthropic.claude-z-v1:0 @eu-central-1 via bedrock_converse",
		"bedrock global.anthropic.claude-z-v1:0 skipped: CI's AWS account can't call global profiles",
		"bedrock in.anthropic.claude-z-v1:0 skipped: no source region for in profiles yet",
		"bedrock jp.anthropic.claude-z-v1:0 @ap-northeast-1 via bedrock_converse",
		"bedrock mistral.large-9 @us-east-1 via bedrock_converse",
		"bedrock openai.gpt-9-luna skipped: no request for its wire APIs yet",
		"bedrock us.anthropic.claude-z-v1:0 @us-east-1 via bedrock_converse",
		"google gemini-9-flash via gemini_generate_content",
		"meta muse-9 skipped: no CI credentials for this host yet",
		"vertex claude-z@20261101 skipped: no CI credentials for this host yet",
	}, planned(probes))
}

// TestPlanRefusesIDsOutsideThePattern: an ID the report's offering pattern
// wouldn't accept is neither called nor shown, since it would go into a URL
// and the PR body; its row says so.
func TestPlanRefusesIDsOutsideThePattern(t *testing.T) {
	t.Parallel()

	patched := offerings{"anthropic": {
		offering("claude-ok", "anthropic_messages/direct"),
		offering("Claude](https://example.com)", "anthropic_messages/direct"),
		offering("claude/../v1/files", "anthropic_messages/direct"),
	}}

	probes, err := plan([]string{vendorDir(t, offerings{}, patched)})
	require.NoError(t, err)

	assert.Equal(t, []string{
		"anthropic  skipped: its ID doesn't fit the offering pattern, so it isn't shown or called",
		"anthropic  skipped: its ID doesn't fit the offering pattern, so it isn't shown or called",
		"anthropic claude-ok via anthropic_messages",
	}, planned(probes))
}

// TestPlanCapsTheCalls: a run checks at most maxProbes offerings, so a
// patch that lists many can't spend much; the rest are listed as skipped.
func TestPlanCapsTheCalls(t *testing.T) {
	t.Parallel()

	many := make([]map[string]any, 0, maxProbes+3)
	for i := range maxProbes + 2 {
		many = append(many, offering(fmt.Sprintf("claude-%03d", i), "anthropic_messages/direct"))
	}

	many = append(many, offering("claude-vertex-only", "anthropic_messages/vertex"))

	probes, err := plan([]string{vendorDir(t, offerings{}, offerings{"anthropic": many})})
	require.NoError(t, err)

	lines := planned(probes)
	assert.Equal(t, "anthropic claude-049 via anthropic_messages", lines[maxProbes-1])
	assert.Equal(t, []string{
		"anthropic claude-050 skipped: over the run's cap of 50 offerings",
		"anthropic claude-051 skipped: over the run's cap of 50 offerings",
		"anthropic claude-vertex-only skipped: no request for its wire APIs yet",
	}, lines[maxProbes:])
}

// TestPlanCallsWhatThePRHoldsInTheEnd: the last folder's catalogue is the
// PR's, so an offering a later folder removes isn't called, and one it
// changes again is routed as it ends up. Example: a person's commit,
// carried after the vendors' (the carry job's folder comes last), fixes a
// wrong ID the agent's commit added; only the fixed ID is called.
func TestPlanCallsWhatThePRHoldsInTheEnd(t *testing.T) {
	t.Parallel()

	const messages = "anthropic_messages/direct"

	agent := offerings{"anthropic": {offering("claude-wrong", messages), offering("claude-kept", "anthropic_messages/vertex")}}
	carried := offerings{"anthropic": {offering("claude-right", messages), offering("claude-kept", messages)}}

	probes, err := plan([]string{vendorDir(t, offerings{}, agent), vendorDir(t, agent, carried)})
	require.NoError(t, err)

	assert.Equal(t, []string{
		"anthropic claude-kept via anthropic_messages",
		"anthropic claude-right via anthropic_messages",
	}, planned(probes))
}
