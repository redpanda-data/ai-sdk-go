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
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadPages(t *testing.T) {
	t.Parallel()

	const sources = `{"anthropic": {"providers": ["anthropic"], "hosts": ["platform.claude.com"], "gate": {
		"model_prefix": "anthropic/claude-", "must_contain": "Claude",
		"pages": [{"id": "anthropic-pricing", "type": "markdown", "url": "https://platform.claude.com/pricing.md"}],
		"per_model": [{"provider": "anthropic", "id": "anthropic-model-{slug}", "type": "markdown", "url": "https://platform.claude.com/models/{slug}.md"}]}}}`

	const snapshot = `{"providers": [
		{"provider": "anthropic", "offerings": [
			{"model": "anthropic/claude-opus-5-5", "lifecycle": {"stage": "ga"}},
			{"model": "anthropic/claude-opus-5-5", "lifecycle": {"stage": "ga"}},
			{"model": "anthropic/claude-haiku-3", "lifecycle": {"stage": "retired", "retires": "2026-01-05"}},
			{"model": "anthropic/claude-sonnet-4-5", "lifecycle": {"stage": "deprecated", "retires": "2027-01-05"}}]},
		{"provider": "openai", "offerings": [{"model": "openai/gpt-6", "lifecycle": {"stage": "ga"}}]},
		{"provider": "gcp.vertex", "offerings": [{"model": "anthropic/claude-opus-4", "lifecycle": {"stage": "ga"}}]}]}`

	pages, err := loadPages([]byte(sources), []byte(snapshot), "anthropic", "2026-10-15")
	require.NoError(t, err)

	ids := make([]string, 0, len(pages))
	for _, p := range pages {
		ids = append(ids, p.ID)
	}

	assert.Equal(t, []string{"anthropic-pricing", "anthropic-model-opus-5-5", "anthropic-model-sonnet-4-5"}, ids,
		"listed pages first; a retired model and other providers' offerings are left out; duplicates merged")
	assert.Equal(t, "https://platform.claude.com/models/sonnet-4-5.md", pages[2].URL)
	assert.Equal(t, "Claude", pages[2].MustContain, "pages take the vendor's content floor")

	_, err = loadPages([]byte(sources), []byte(snapshot), "openai", "2026-10-15")
	require.ErrorContains(t, err, "openai")

	duplicate := strings.Replace(sources, `"anthropic-model-{slug}"`, `"anthropic-pricing"`, 1)
	_, err = loadPages([]byte(duplicate), []byte(snapshot), "anthropic", "2026-10-15")
	require.ErrorContains(t, err, "duplicate page id")

	unsafe := strings.Replace(sources, `"anthropic-pricing"`, `"../escape"`, 1)
	_, err = loadPages([]byte(unsafe), []byte(snapshot), "anthropic", "2026-10-15")
	require.ErrorContains(t, err, "isn't safe as a file name", "an id names a file in the gate's output")
}

// TestSourcesFile checks the real page list against the real snapshot.
func TestSourcesFile(t *testing.T) {
	t.Parallel()

	sources, err := os.ReadFile(filepath.Join("..", "..", sourcesPath))
	require.NoError(t, err)

	snapshot, err := os.ReadFile(filepath.Join("..", "..", "catalog", "snapshot.json"))
	require.NoError(t, err)

	vendors, err := readSources(sources)
	require.NoError(t, err)

	for vendor, config := range vendors {
		pages, err := loadPages(sources, snapshot, vendor, time.Now().UTC().Format(time.DateOnly))
		require.NoError(t, err)

		for _, p := range pages {
			parsed, err := url.Parse(p.URL)
			require.NoError(t, err, p.ID)

			assert.Contains(t, config.Hosts, parsed.Hostname(), "%s: every page is on a named host", p.ID)
			assert.Contains(t, []string{typeMarkdown, typeHTML, typePriceList}, p.Type, p.ID)
			assert.NotEmpty(t, p.MustContain, "%s: every page has a content floor", p.ID)

			switch {
			case strings.HasSuffix(parsed.Hostname(), "cloud.google.com"):
				assert.Equal(t, "en", parsed.Query().Get("hl"), "%s: Google pages pin English", p.ID)
			case parsed.Hostname() == "docs.aws.amazon.com":
				assert.True(t, strings.HasSuffix(parsed.Path, ".md"), "%s: AWS serves .md; a missing .html redirects to a 200", p.ID)
			}

			if p.Type == typePriceList {
				assert.NotEmpty(t, p.Match, p.ID)
				assert.Positive(t, p.MinLines, p.ID)
			}
		}

		for _, tmpl := range config.Gate.PerModel {
			prefix, _, _ := strings.Cut(tmpl.ID, "{slug}")
			assert.True(t, slices.ContainsFunc(pages, func(p page) bool { return strings.HasPrefix(p.ID, prefix) }),
				"template %s yields no page", tmpl.ID)
		}
	}
}
