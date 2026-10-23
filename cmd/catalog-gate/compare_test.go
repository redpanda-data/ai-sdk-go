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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCompareRuns(t *testing.T) {
	t.Parallel()

	const priceRow = `{"offering": "claude-opus-5-5", "field": "price", "old": "4", "new": "5", "source_url": "https://platform.claude.com/docs/en/about-claude/pricing", "rule": "price-follow"}`
	// A baseline from before change rows named their rule.
	const oldPriceRow = `{"offering": "claude-opus-5-5", "field": "price", "old": "4", "new": "5", "source_url": "https://platform.claude.com/docs/en/about-claude/pricing"}`
	const askRow = `{"offering": "bedrock:eu.anthropic.claude-fable-5", "field": "lifecycle", "values": [], "rule": "copy-gone-no-eol", "source_urls": ["https://docs.aws.amazon.com/bedrock/latest/userguide/some-other-page.html"]}`
	report := func(changes, asks string) string {
		return `{"changes": [` + changes + `], "needs_human": [` + asks + `], "skipped": []}`
	}
	pages := fingerprints{Sources: map[string]source{
		"anthropic-pricing": {URL: "https://platform.claude.com/docs/en/about-claude/pricing.md"},
	}}

	tests := []struct {
		name        string
		baseline    string
		same        bool
		contains    []string
		notContains []string
	}{
		{
			name: "same findings", baseline: report(priceRow, askRow), same: true,
			contains: []string{
				"**A skip would have lost nothing:**",
				"**Cited sources the gate doesn't fingerprint (1):**", "`https://docs.aws.amazon.com/bedrock/latest/userguide/some-other-page.html`",
			},
			notContains: []string{"`https://platform.claude.com/docs/en/about-claude/pricing`"},
		},
		{
			name: "baseline from before change rules", baseline: report(oldPriceRow, askRow), same: true,
			contains: []string{"**A skip would have lost nothing:**"},
		},
		{
			name: "the baseline run missed a change", baseline: report("", askRow), same: true,
			contains: []string{"**A skip would have lost findings:**", "Found now, not by the baseline run: `claude-opus-5-5 price`"},
		},
		{
			name: "this run missed one", same: true,
			baseline: report(priceRow+`, {"offering": "claude-haiku-5-5", "field": "limit", "old": "1", "new": "2", "source_url": "https://platform.claude.com/x"}`, askRow),
			contains: []string{"Found by the baseline run, not now: `claude-haiku-5-5 limit`"},
		},
		{
			name: "same rows, different values", same: true,
			baseline:    report(`{"offering": "claude-opus-5-5", "field": "price", "old": "4", "new": "6", "source_url": "https://platform.claude.com/x", "rule": "price-follow"}`, askRow),
			contains:    []string{"**A skip would have lost findings:**", "Found by both runs, with different values: `claude-opus-5-5 price`"},
			notContains: []string{"lost nothing", "\"6\""},
		},
		{
			name: "pages or inputs changed", baseline: report("", ""), same: false,
			contains: []string{"The pages or inputs changed since the baseline run, so the two runs' findings aren't compared."},
		},
		{
			name: "no baseline report", baseline: "", same: true,
			contains: []string{"No report from the baseline run to compare with."},
		},
		{
			name: "a hostile offering in the baseline report is never printed", same: true,
			baseline:    report(`{"offering": "<img src=x>", "field": "price", "old": "", "new": "", "source_url": "https://x.test/"}`, askRow),
			notContains: []string{"<img"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			text, err := compareRuns([]byte(report(priceRow, askRow)), []byte(tt.baseline), pages, tt.same)
			require.NoError(t, err)

			for _, want := range tt.contains {
				assert.Contains(t, text, want)
			}

			for _, unwanted := range tt.notContains {
				assert.NotContains(t, text, unwanted)
			}
		})
	}

	_, err := compareRuns([]byte(`{"changes": []}`), nil, pages, true)
	require.ErrorContains(t, err, "required", "this run's report is parsed strictly, as catalog-report parses it")
}
