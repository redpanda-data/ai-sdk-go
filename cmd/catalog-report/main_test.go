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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// priced is a report with one price change, from source.
func priced(offering, source string) Report {
	return Report{
		Changes:    []Change{{Offering: offering, Field: "price", Old: "input 2.00", New: "input 2.50", SourceURL: source, Rule: "price-follow"}},
		NeedsHuman: []NeedsHuman{}, Skipped: []Skipped{},
	}
}

// vendorFolder writes a vendor's folder, as its check job leaves it: the
// report and the named files.
func vendorFolder(t *testing.T, folder string, report Report, files map[string]string) string {
	t.Helper()

	require.NoError(t, os.MkdirAll(folder, 0o700))

	raw, err := json.Marshal(report)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(folder, "report.json"), raw, 0o600))

	for file, content := range files {
		require.NoError(t, os.WriteFile(filepath.Join(folder, file), []byte(content), 0o600))
	}

	return folder
}

// buildCatalogReport builds the command, as pr-job.sh does, and returns its path.
func buildCatalogReport(t *testing.T) string {
	t.Helper()

	bin := filepath.Join(t.TempDir(), "catalog-report")
	out, err := exec.CommandContext(t.Context(), "go", "build", "-o", bin, ".").CombinedOutput()
	require.NoError(t, err, string(out))

	return bin
}

// TestMainReadsEachVendorsFolder checks the command's inputs: one folder per
// vendor, each report checked against that vendor's own hosts in
// named-sources.json (a source on another vendor's host fails the run), and
// a folder with only one of its two catalogues refused.
func TestMainReadsEachVendorsFolder(t *testing.T) {
	t.Parallel()

	bin := buildCatalogReport(t)
	dir := t.TempDir()
	sources := filepath.Join(dir, "named-sources.json")
	require.NoError(t, os.WriteFile(sources, []byte(`{"anthropic": {"providers": ["anthropic"], "hosts": ["platform.claude.com"]},
		"openai": {"providers": ["openai"], "hosts": ["developers.openai.com"]}, "google": {"providers": ["google"], "hosts": ["ai.google.dev"]}}`), 0o600))

	folder := func(name string, report Report, files map[string]string) string {
		return vendorFolder(t, filepath.Join(dir, name), report, files)
	}
	empty := `{"providers": []}`
	both := map[string]string{"base-snapshot.json": empty, "snapshot.json": empty}

	// Anthropic adds an AU copy of Sonnet 5.5 at 10% more, and the routes move
	// Sydney's bare-name calls onto it.
	sonnet := func(copies string) string {
		return `{"providers": [{"provider": "aws.bedrock", "offerings": [` + copies + `]}]}`
	}
	const global = `{"id": "global.anthropic.claude-sonnet-5-5", "model": "anthropic/claude-sonnet-5-5", "pricing": {"default": {"base": {"input_microcents_per_million": 300000000}}}}`
	const au = `{"id": "au.anthropic.claude-sonnet-5-5", "model": "anthropic/claude-sonnet-5-5", "pricing": {"default": {"base": {"input_microcents_per_million": 330000000}}}}`
	anthropic := folder("anthropic", priced("claude-opus-5-5", "https://platform.claude.com/docs/en/about-claude/pricing.md"),
		map[string]string{"base-snapshot.json": sonnet(global), "snapshot.json": sonnet(global + "," + au)})
	routesBefore, routesAfter := filepath.Join(dir, "base-routes.txt"), filepath.Join(dir, "routes.txt")
	require.NoError(t, os.WriteFile(routesBefore, []byte("ap-southeast-2 anthropic.claude-sonnet-5-5 global.anthropic.claude-sonnet-5-5\n"), 0o600))
	require.NoError(t, os.WriteFile(routesAfter, []byte("ap-southeast-2 anthropic.claude-sonnet-5-5 au.anthropic.claude-sonnet-5-5\n"), 0o600))

	openai := folder("openai", priced("gpt-6-1", "https://developers.openai.com/api/docs/pricing"), both)
	quiet := folder("quiet", Report{Changes: []Change{}, NeedsHuman: []NeedsHuman{}, Skipped: []Skipped{}}, nil)
	borrowed := folder("borrowed", priced("gpt-6-1", "https://platform.claude.com/docs/en/about-claude/pricing.md"), both)
	// The catalogue already carries Sonnet 5.5, so its AU copy is a new_copy.
	carried := `"facts": {"anthropic/claude-sonnet-5-5": {"released": "2026-09-30"}}`
	mislabelled := folder("mislabelled", Report{
		Changes: []Change{{
			Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_model", New: "au.anthropic.claude-sonnet-5-5",
			SourceURL: "https://platform.claude.com/docs/en/about-claude/models/overview.md", Rule: "new-copy-offered-region",
		}},
		NeedsHuman: []NeedsHuman{}, Skipped: []Skipped{},
	}, map[string]string{
		"base-snapshot.json": `{` + carried + `, "providers": [{"provider": "aws.bedrock", "offerings": [` + global + `]}]}`,
		"snapshot.json":      `{` + carried + `, "providers": [{"provider": "aws.bedrock", "offerings": [` + global + `,` + au + `]}]}`,
	})
	halfCatalogue := folder("half", priced("gpt-6-1", "https://developers.openai.com/api/docs/pricing"), map[string]string{"snapshot.json": empty})
	noCatalogue := folder("none", priced("gpt-6-1", "https://developers.openai.com/api/docs/pricing"), nil)

	tests := []struct {
		name    string
		args    []string
		wantErr string
		want    []string
	}{
		{
			name: "one section per vendor folder, and a failed vendor",
			args: []string{"-vendor", "anthropic=" + anthropic, "-vendor", "openai=" + openai, "-failed", "google"},
			want: []string{"## Catalogue drift: anthropic, openai, google\n", "\n## anthropic\n", "\n## openai\n", "This vendor's run failed"},
		},
		{
			// The whole PR's catalogue is the first vendor's before and the
			// last changed vendor's after; a vendor that changed nothing has none.
			name: "routing prices come from the whole PR's catalogue",
			args: []string{"-vendor", "anthropic=" + anthropic, "-vendor", "openai=" + quiet, "-base-routes", routesBefore, "-routes", routesAfter},
			want: []string{"| `global.anthropic.claude-sonnet-5-5` | `au.anthropic.claude-sonnet-5-5` | +10% | Australia |"},
		},
		{
			name: "a replayed vendor names the agent run its change came from",
			args: []string{"-vendor", "anthropic=" + anthropic, "-vendor", "openai=" + openai, "-replayed", "anthropic=40504204648@2026-10-27T06:17:00Z"},
			want: []string{"\n## anthropic\n\nThis is the change the agent made in run 40504204648 on 2026-10-27, replayed with no model and checked to be exactly the same on this base.\n"},
		},
		{
			name: "a carried commit is listed under the checks line",
			args: []string{"-vendor", "anthropic=" + anthropic, "-carried", "1faeeba000000000000000000000000000000000\tt\ttest: swap the live test model"},
			want: []string{"**Carried from people:** `1faeeba` by t: `test: swap the live test model`."},
		},
		{
			name:    "a carried commit that doesn't start with a full SHA",
			args:    []string{"-vendor", "anthropic=" + anthropic, "-carried", "1faeeba\tt\ttest"},
			wantErr: `-carried "1faeeba\tt\ttest" doesn't start with a commit's full SHA`,
		},
		{
			name:    "a replay that isn't RUN@TIME",
			args:    []string{"-vendor", "anthropic=" + anthropic, "-replayed", "anthropic=yesterday"},
			wantErr: `-replayed "anthropic=yesterday" isn't NAME=RUN@TIME`,
		},
		{
			name:    "a replay of a vendor with no folder",
			args:    []string{"-vendor", "anthropic=" + anthropic, "-replayed", "openai=40504204648@2026-10-27T06:17:00Z"},
			wantErr: `-replayed names "openai", which has no -vendor folder`,
		},
		{
			name:    "a source on another vendor's host",
			args:    []string{"-vendor", "anthropic=" + anthropic, "-vendor", "openai=" + borrowed},
			wantErr: "openai: changes[0]: source not on a named-source host",
		},
		{
			name:    "a vendor named-sources.json doesn't list",
			args:    []string{"-vendor", "xai=" + openai},
			wantErr: `no named sources for vendor "xai"`,
		},
		{
			name:    "a vendor named twice",
			args:    []string{"-vendor", "openai=" + openai, "-failed", "openai"},
			wantErr: `vendor "openai" named twice`,
		},
		{
			name:    "a new_model row for a model the catalogue carries",
			args:    []string{"-vendor", "anthropic=" + mislabelled},
			wantErr: `the catalogue already carries "anthropic/claude-sonnet-5-5", so the field is new_copy, not new_model`,
		},
		{
			name:    "one catalogue without the other",
			args:    []string{"-vendor", "openai=" + halfCatalogue},
			wantErr: "needs both base-snapshot.json and snapshot.json, or neither",
		},
		{
			name:    "changes without the catalogues they changed",
			args:    []string{"-vendor", "openai=" + noCatalogue},
			wantErr: "has changes but no base-snapshot.json and snapshot.json",
		},
		{
			name:    "a vendor without its folder",
			args:    []string{"-vendor", "openai"},
			wantErr: `-vendor "openai" isn't NAME=DIR`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := filepath.Join(t.TempDir(), "body.md")
			args := append([]string{"-sources", sources, "-rules", filepath.Join("..", "..", ".github", "reconcile-models", "RULES.md"), "-out", body}, tt.args...)
			out, err := exec.CommandContext(t.Context(), bin, args...).CombinedOutput()

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, string(out), tt.wantErr)
				assert.NoFileExists(t, body, "an invalid input writes no body")

				return
			}

			require.NoError(t, err, string(out))

			raw, err := os.ReadFile(body)
			require.NoError(t, err)

			for _, want := range tt.want {
				assert.Contains(t, string(raw), want)
			}
		})
	}
}
