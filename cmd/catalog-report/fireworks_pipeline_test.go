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
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

func TestFireworksDiscoveryPatchReportSnapshot(t *testing.T) {
	t.Parallel()

	fixture := filepath.Join("testdata", "fireworks-reconciliation")
	source, err := os.ReadFile(filepath.Join(fixture, "model.go.txt"))
	require.NoError(t, err)

	rawCases, err := os.ReadFile(filepath.Join(fixture, "cases.json"))
	require.NoError(t, err)

	var cases []struct {
		Name                  string          `json:"name"`
		DiscoveredID          string          `json:"discovered_id"`
		Before                string          `json:"before"`
		After                 string          `json:"after"`
		ExpectedSnapshotField string          `json:"expected_snapshot_field"`
		ExpectedSnapshotValue string          `json:"expected_snapshot_value"`
		Report                json.RawMessage `json:"report"`
	}
	require.NoError(t, json.Unmarshal(rawCases, &cases))
	require.Len(t, cases, 4)

	rulesRaw, err := os.ReadFile(filepath.Join("..", "..", reconcile.RulesPath))
	require.NoError(t, err)

	rules, problems := reconcile.ParseRules(rulesRaw)
	require.Empty(t, problems)
	sources := namedSources(t)

	discoveryRaw, err := os.ReadFile(filepath.Join("..", "..", "providers", "fireworks", "testdata", "discovery", "inference-models.json"))
	require.NoError(t, err)

	var discovery struct {
		Response struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		} `json:"response"`
	}
	require.NoError(t, json.Unmarshal(discoveryRaw, &discovery))

	discovered := make(map[string]bool, len(discovery.Response.Data))
	for _, model := range discovery.Response.Data {
		discovered[model.ID] = true
	}

	for _, tt := range cases {
		t.Run(tt.Name, func(t *testing.T) {
			t.Parallel()

			require.Contains(t, discovered, tt.DiscoveredID)

			report, err := reconcile.ParseReport(tt.Report)
			require.NoError(t, err)
			require.Empty(t, Validate(report, "zai", sources["zai"], rules))
			require.Len(t, report.Changes, 1)
			assert.Equal(t, "fireworks:"+tt.DiscoveredID, report.Changes[0].Offering)

			repo := t.TempDir()
			root, err := os.OpenRoot(repo)
			require.NoError(t, err)

			defer root.Close()

			path := filepath.Join("providers", "fireworks", "models.go")
			file := filepath.Join(repo, path)
			require.NoError(t, root.MkdirAll(filepath.Dir(path), 0o700))
			require.NoError(t, root.WriteFile(path, source, 0o600))

			git := func(input string, args ...string) string {
				t.Helper()

				cmd := exec.CommandContext(t.Context(), "git", args...)
				cmd.Dir = repo
				cmd.Stdin = strings.NewReader(input)
				output, err := cmd.CombinedOutput()
				require.NoError(t, err, string(output))

				return string(output)
			}
			regenerate := func() reconcile.Snapshot {
				t.Helper()

				cmd := exec.CommandContext(t.Context(), "go", "run", file)
				cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "HOME=" + os.Getenv("HOME"), "GOPROXY=off", "GOTOOLCHAIN=local", "GOWORK=off"}
				output, err := cmd.CombinedOutput()
				require.NoError(t, err, string(output))

				snap, err := reconcile.ParseSnapshot(output)
				require.NoError(t, err)

				return snap
			}

			git("", "init", "-q")
			git("", "add", "--", path)

			before := regenerate()

			require.Equal(t, 1, strings.Count(string(source), tt.Before))
			require.NoError(t, root.WriteFile(path, []byte(strings.Replace(string(source), tt.Before, tt.After, 1)), 0o600))

			patch := git("", "diff", "--", path)
			require.NotEmpty(t, patch)
			require.NoError(t, root.WriteFile(path, source, 0o600))
			git(patch, "apply", "--check", "-")
			git(patch, "apply", "-")

			after := regenerate()

			base := Base{Before: &before, After: &after, Rules: rules, Today: "2026-11-04", Branch: "main"}
			require.Empty(t, base.ValidateAdded(report, "zai"))

			offering, ok := after.Offering("fireworks", tt.DiscoveredID)
			require.True(t, ok)

			value, err := json.Marshal(offering[tt.ExpectedSnapshotField])
			require.NoError(t, err)
			assert.Contains(t, string(value), tt.ExpectedSnapshotValue)

			body := Render([]Vendor{{Name: "zai", Report: report, Before: &before, After: &after}}, patch, nil, base)
			assert.Contains(t, body, "fireworks:"+tt.DiscoveredID)
			assert.NotContains(t, body, "Changed with no row")
			assert.Empty(t, reconcile.ReplayMismatches(before, after, before, after, report, "zai"))
			assert.NotEmpty(t, reconcile.ReplayMismatches(before, after, before, before, report, "zai"), "a dropped patch cannot replay")
			assert.NotEmpty(t, reconcile.ReplayMismatches(before, after, after, after, report, "zai"), "a changed base needs fresh reconciliation")
		})
	}
}
