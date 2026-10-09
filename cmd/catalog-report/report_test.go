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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var hosts = []string{"platform.claude.com"}

func validReport() Report {
	return Report{
		Changes: []Change{{
			Offering: "claude-opus-5-5", Field: "price", Old: "4.00", New: "5.00",
			SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing.md",
		}},
		NeedsHuman: []NeedsHuman{{
			Offering: "claude-sonnet-5-5", Field: "limit", Values: []string{"65535", "65536"},
			SourceURLs: []string{"https://platform.claude.com/docs/en/about-claude/models/overview.md"},
		}},
		Skipped: []Skipped{{Offering: "claude-haiku-4-5", Reason: "conflicting_sources"}},
	}
}

func TestValidate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		mutate  func(r *Report)
		wantErr string
	}{
		{"valid", func(_ *Report) {}, ""},
		{"empty report", func(r *Report) { *r = Report{} }, ""},
		{"unknown field", func(r *Report) { r.Changes[0].Field = "notes" }, "field"},
		{"unknown reason", func(r *Report) { r.Skipped[0].Reason = "because" }, "reason"},
		{"bad offering", func(r *Report) { r.Changes[0].Offering = "Claude <b>" }, "offering"},
		{"markdown in value", func(r *Report) { r.Changes[0].New = "[x](https://evil.example)" }, "value"},
		{"html in value", func(r *Report) { r.NeedsHuman[0].Values[0] = "<img src=x>" }, "value"},
		{"backtick in value", func(r *Report) { r.Changes[0].Old = "`4`" }, "value"},
		{"http source", func(r *Report) { r.Changes[0].SourceURL = "http://platform.claude.com/x" }, "source"},
		{"foreign host", func(r *Report) { r.Changes[0].SourceURL = "https://evil.example/pricing" }, "source"},
		{"source with markdown", func(r *Report) {
			r.Changes[0].SourceURL = "https://platform.claude.com/x)|[click](https://evil.example)"
		}, "source"},
		{"source with pipe", func(r *Report) { r.Changes[0].SourceURL = "https://platform.claude.com/a|b" }, "source"},
		{"source with html", func(r *Report) { r.NeedsHuman[0].SourceURLs[0] = "https://platform.claude.com/<img src=x>" }, "source"},
		{"source with space", func(r *Report) { r.Changes[0].SourceURL = "https://platform.claude.com/a b" }, "source"},
		{"source with anchor and query", func(r *Report) {
			r.Changes[0].SourceURL = "https://platform.claude.com/docs/en/pricing?tab=batch&x=1#model-pricing"
		}, ""},
		{"missing needs_human source", func(r *Report) { r.NeedsHuman[0].SourceURLs = nil }, "source"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			report := validReport()
			tt.mutate(&report)

			problems := Validate(report, hosts)
			if tt.wantErr == "" {
				assert.Empty(t, problems)
				return
			}

			assert.Contains(t, strings.Join(problems, "\n"), tt.wantErr)
		})
	}
}

func TestParse(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"all three arrays", `{"changes": [], "needs_human": [], "skipped": []}`, false},
		{"truncated json", `{"changes": [`, true},
		{"unknown field", `{"changes": [], "needs_human": [], "skipped": [], "extra": 1}`, true},
		{"null", `null`, true},
		{"empty object", `{}`, true},
		{"missing changes", `{"needs_human": [], "skipped": []}`, true},
		{"missing needs_human", `{"changes": [], "skipped": []}`, true},
		{"missing skipped", `{"changes": [], "needs_human": []}`, true},
		{"null array", `{"changes": null, "needs_human": [], "skipped": []}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := Parse([]byte(tt.input))
			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
		})
	}
}

func TestRender(t *testing.T) {
	t.Parallel()

	diff := "-  \"input\": 4\n+  \"input\": 5\n"

	tests := []struct {
		name        string
		report      Report
		diff        string
		contains    []string
		notContains []string
	}{
		{
			name:   "full report",
			report: validReport(),
			diff:   diff,
			contains: []string{
				"## Catalogue drift: anthropic",
				"| `claude-opus-5-5` | price | `4.00` | `5.00` | [platform.claude.com](https://platform.claude.com/docs/en/about-claude/pricing.md) |",
				"`65535`, `65536`",
				"| `claude-haiku-4-5` | conflicting_sources |",
				"```diff",
			},
		},
		{
			name:        "no drift",
			report:      Report{},
			contains:    []string{"No drift found"},
			notContains: []string{"```diff"},
		},
		{
			name:        "empty report keeps diff",
			report:      Report{},
			diff:        diff,
			contains:    []string{"```diff", "+  \"input\": 5"},
			notContains: []string{"No drift found"},
		},
		{
			name:     "says not to push to the branch",
			report:   validReport(),
			diff:     diff,
			contains: []string{"Don't push commits to this branch"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := Render(tt.report, "anthropic", tt.diff)
			for _, want := range tt.contains {
				assert.Contains(t, body, want)
			}

			for _, unwanted := range tt.notContains {
				assert.NotContains(t, body, unwanted)
			}
		})
	}
}
