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

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

var hosts = []string{"platform.claude.com"}

// testRules is a RULES.md with four skip rules, one rule that asks a person,
// R4 for rows no rule covers, and Bedrock's offered regions.
var testRules = reconcile.Rules{
	Lines:     map[string]int{"T1": 40, "L3": 80, "R4": 120, "region-not-offered": 20, "copy-gone-no-eol": 22, "label-fact-differs": 24},
	Reasons:   map[string]string{"limited_availability": "the model needs a signup", "conflicting_sources": "no rule uses it yet"},
	Questions: map[string]string{"L3": "Which replacement?", "R4": "No rule covers this. Decide."},
	Produces: map[string][]string{
		"T1": {"skip:`limited_availability`"}, "L3": {"needs_human"}, "R4": {"needs_human"},
		"region-not-offered": {"skip:`region_not_offered`"}, "copy-gone-no-eol": {"skip:`copy_unlisted`"},
		"label-fact-differs": {"skip:`fixed_fact_differs`"},
	},
	Offered: map[string][]string{"bedrock": {"global", "us", "eu", "au", "jp"}},
}

func validReport() Report {
	return Report{
		Changes: []Change{{
			Offering: "claude-opus-5-5", Field: "price", Old: "4.00", New: "5.00",
			SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing.md",
		}},
		NeedsHuman: []NeedsHuman{{
			Offering: "claude-sonnet-5-5", Field: "limit", Values: []string{"65535", "65536"}, Rule: "R4",
			SourceURLs: []string{"https://platform.claude.com/docs/en/about-claude/models/overview.md"},
		}},
		Skipped: []Skipped{{Offering: "claude-haiku-4-5", Reason: "conflicting_sources", Rule: "R4"}},
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
		{"limited availability reason", func(r *Report) { r.Skipped[0].Reason = "limited_availability" }, ""},
		{"excluded vendor reason", func(r *Report) { r.Skipped[0].Reason = "vendor_excluded" }, ""},
		{"bad offering", func(r *Report) { r.Changes[0].Offering = "Claude <b>" }, "offering"},
		{"markdown in value", func(r *Report) { r.Changes[0].New = "[x](https://evil.example)" }, "value"},
		{"html in value", func(r *Report) { r.NeedsHuman[0].Values[0] = "<img src=x>" }, "value"},
		{"backtick in value", func(r *Report) { r.Changes[0].Old = "`4`" }, "value"},
		{"plain link in value", func(r *Report) { r.Changes[0].New = "https://evil.example/x" }, "value"},
		{"slash in value", func(r *Report) { r.Changes[0].New = "us/eu cache read 0.11" }, ""},
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
		{"rule RULES.md doesn't have", func(r *Report) { r.Skipped[0].Rule = "Z9" }, `skipped[0]: unknown rule "Z9"`},
		{"needs_human rule that never asks a person", func(r *Report) { r.NeedsHuman[0].Rule = "T1" }, "needs_human[0]: rule T1 doesn't produce needs_human"},
		{"skip rule for another reason", func(r *Report) { r.Skipped[0].Rule = "T1" }, "skipped[0]: rule T1 doesn't produce skip:`conflicting_sources`"},
		{"skip rule for this reason", func(r *Report) { r.Skipped[0] = Skipped{Offering: "x", Reason: "limited_availability", Rule: "T1"} }, ""},
		{"region not offered", func(r *Report) {
			r.Skipped[0] = Skipped{Offering: "bedrock:in.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"}
		}, ""},
		{"region not offered for an offered region", func(r *Report) {
			r.Skipped[0] = Skipped{Offering: "bedrock:us.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"}
		}, "skipped[0]: region us is offered on bedrock"},
		{"copy unlisted", func(r *Report) {
			r.Skipped[0] = Skipped{Offering: "bedrock:eu.anthropic.claude-fable-5", Reason: "copy_unlisted", Rule: "copy-gone-no-eol"}
		}, ""},
		{"fixed fact differs", func(r *Report) {
			r.Skipped[0] = Skipped{Offering: "claude-sonnet-5", Reason: "fixed_fact_differs", Rule: "label-fact-differs"}
		}, ""},
		{"region not offered on a host with no Offered row", func(r *Report) {
			r.Skipped[0] = Skipped{Offering: "vertex:in.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"}
		}, "skipped[0]: vertex has no Offered regions row"},
		{"region not offered without a package", func(r *Report) {
			r.Skipped[0] = Skipped{Offering: "in.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"}
		}, "skipped[0]: in.anthropic.claude-opus-5 names no package"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			report := validReport()
			tt.mutate(&report)

			problems := Validate(report, hosts, testRules)
			if tt.wantErr == "" {
				assert.Empty(t, problems)
				return
			}

			assert.Contains(t, strings.Join(problems, "\n"), tt.wantErr)
		})
	}
}

func TestRender(t *testing.T) {
	t.Parallel()

	diff := "-  \"input\": 4\n+  \"input\": 5\n"

	const baseURL = "https://example.test/blob/abc"

	decisionsURL := baseURL + "/" + reconcile.DecisionsPath
	rulesURL := baseURL + "/" + reconcile.RulesPath + "?plain=1"
	twoAsks := validReport()
	twoAsks.NeedsHuman = append(twoAsks.NeedsHuman, NeedsHuman{
		Offering: "claude-opus-4-1", Field: "replacement", Rule: "L3",
		SourceURLs: []string{"https://platform.claude.com/docs/en/about-claude/model-deprecations.md"},
	})
	keepLimit := reconcile.Decision{
		ID: "DEC-001", Offering: "anthropic:claude-sonnet-5-5", Field: "limit", Kind: reconcile.KindKeep,
		Reason: "Agreed in review.", ReviewBy: "2027-01-13", Line: 3,
	}
	holdOld := reconcile.Decision{
		ID: "DEC-002", Offering: "vertex:claude-haiku-4-5", Field: reconcile.FieldNewModel, Kind: reconcile.KindHold,
		Reason: "Not offered to users.", ReviewBy: "2027-01-13", Line: 12,
	}
	otherField := keepLimit
	otherField.Field = "price"
	// #259's copies in India, which Bedrock doesn't offer, and one in a second region.
	waiting := []Skipped{
		{Offering: "bedrock:in.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"},
		{Offering: "bedrock:in.anthropic.claude-sonnet-5", Reason: "region_not_offered", Rule: "region-not-offered"},
		{Offering: "bedrock:ca.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"},
		{Offering: "bedrock:in.anthropic.claude-haiku-4-5-20251001-v1:0", Reason: "region_not_offered", Rule: "region-not-offered"},
	}

	tests := []struct {
		name        string
		report      Report
		diff        string
		codeFiles   []string
		base        Base
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
				"| ⚠ `R4` | `conflicting_sources` | `claude-haiku-4-5` |",
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
			name:        "says not to push to the branch, and how to settle a row",
			report:      validReport(),
			diff:        diff,
			contains:    []string{"Don't push commits to this branch", "record a decision in `.github/reconcile-models/decisions.yaml`"},
			notContains: []string{"Comment on the PR instead", "Recorded decisions"},
		},
		{
			name:   "a needs_human row a decision answers is held, not asked",
			report: validReport(),
			base:   Base{URL: baseURL, Decisions: []reconcile.Decision{keepLimit}, Today: "2026-10-14"},
			contains: []string{
				"### Held by decisions (1)",
				"| [DEC-001](" + decisionsURL + "#L3) | `claude-sonnet-5-5` | limit | `65535`, `65536` | Agreed in review. |",
				"Recorded decisions: 1, in [decisions.yaml](" + decisionsURL + "). Past their review date: none.",
			},
			notContains: []string{"### Needs a human"},
		},
		{
			name:        "a skipped offering under a hold is held",
			report:      Report{Skipped: []Skipped{{Offering: "vertex:claude-haiku-4-5", Reason: "out_of_scope"}}},
			base:        Base{URL: baseURL, Decisions: []reconcile.Decision{holdOld}, Today: "2026-10-14"},
			contains:    []string{"| [DEC-002](" + decisionsURL + "#L12) | `vertex:claude-haiku-4-5` | new_model | out_of_scope | Not offered to users. |"},
			notContains: []string{"### Skipped"},
		},
		{
			name:   "needs a human is a block per row, marked when no rule covers it",
			report: twoAsks,
			base:   Base{URL: baseURL, Rules: testRules},
			contains: []string{
				"### Needs a human (2): 1 by a rule, 1 with no rule",
				"**1. `claude-sonnet-5-5` · limit · ⚠ no rule covers this ([R4](" + rulesURL + "#L120))**",
				"- **Question:** No rule covers this. Decide.",
				"- **Found:** `65535`, `65536`",
				"- **Sources:** [platform.claude.com](https://platform.claude.com/docs/en/about-claude/models/overview.md)",
				"**2. `claude-opus-4-1` · replacement · rule [L3](" + rulesURL + "#L80)**\n- **Question:** Which replacement?\n- **Sources:**",
			},
			notContains: []string{"| Offering | Field | Values |"},
		},
		{
			name: "skipped rows are grouped by rule",
			report: Report{Skipped: []Skipped{
				{Offering: "claude-mythos-5", Reason: "limited_availability", Rule: "T1"},
				{Offering: "claude-haiku-4-5", Reason: "conflicting_sources", Rule: "R4"},
				{Offering: "claude-mythos-5-1", Reason: "limited_availability", Rule: "T1"},
			}},
			base: Base{URL: baseURL, Rules: testRules},
			contains: []string{
				"### Skipped (3)\n\n| Rule | Why | Offerings |",
				"| [T1](" + rulesURL + "#L40) | the model needs a signup (`limited_availability`) | `claude-mythos-5`, `claude-mythos-5-1` |",
				"| ⚠ [R4](" + rulesURL + "#L120) | no rule uses it yet (`conflicting_sources`) | `claude-haiku-4-5` |",
			},
		},
		{
			name:   "one waiting line per region",
			report: Report{Skipped: waiting},
			base:   Base{URL: baseURL, Rules: testRules},
			contains: []string{
				"### Waiting on a product decision (2)\n\nModel cards list copies in regions this catalogue doesn't offer, so the bot left them out " +
					"(rule [region-not-offered](" + rulesURL + "#L20)). Nothing changes until someone decides, and they don't block the PR.\n\n",
				"- Offer `in` on bedrock? 3 copies: `bedrock:in.anthropic.claude-opus-5`, `bedrock:in.anthropic.claude-sonnet-5`, " +
					"`bedrock:in.anthropic.claude-haiku-4-5-20251001-v1:0`\n",
				"- Offer `ca` on bedrock? 1 copy: `bedrock:ca.anthropic.claude-opus-5`\n",
			},
		},
		{
			name:        "waiting rows aren't listed as skipped",
			report:      Report{Skipped: append([]Skipped{{Offering: "claude-mythos-5", Reason: "limited_availability", Rule: "T1"}}, waiting...)},
			base:        Base{URL: baseURL, Rules: testRules},
			contains:    []string{"### Skipped (1)"},
			notContains: []string{"`region_not_offered`"},
		},
		{
			name:        "a decision on another field holds nothing",
			report:      validReport(),
			base:        Base{URL: baseURL, Decisions: []reconcile.Decision{otherField}, Today: "2026-10-14"},
			contains:    []string{"### Needs a human (1)"},
			notContains: []string{"### Held by decisions"},
		},
		{
			name:     "an overdue decision is named even with no drift",
			report:   Report{},
			base:     Base{URL: baseURL, Decisions: []reconcile.Decision{keepLimit}, Today: "2027-02-01"},
			contains: []string{"No drift found", "Past their review date: DEC-001 (still enforced)."},
		},
		{
			name:        "no links without a base URL",
			report:      validReport(),
			base:        Base{Decisions: []reconcile.Decision{keepLimit}, Today: "2026-10-14"},
			contains:    []string{"| `DEC-001` | `claude-sonnet-5-5` |", "Recorded decisions: 1, in `decisions.yaml`."},
			notContains: []string{"](#L"},
		},
		{
			name:      "lists code files changed",
			report:    validReport(),
			diff:      diff,
			codeFiles: []string{"providers/anthropic/alias_test.go"},
			contains: []string{
				"### Code changed (1)", "- `providers/anthropic/alias_test.go`",
				"marking it ready runs them with secrets",
			},
		},
		{
			name: "empty old shows none",
			report: Report{Changes: []Change{{
				Offering: "claude-haiku-5-5", Field: "new_model", Old: "", New: "claude-haiku-5-5",
				SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing",
			}}},
			diff:        diff,
			contains:    []string{"| new_model | none | `claude-haiku-5-5` |"},
			notContains: []string{"| `` |"},
		},
		{
			name:        "no code section for a data-only patch",
			report:      validReport(),
			diff:        diff,
			notContains: []string{"Code changed"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := Render(tt.report, "anthropic", tt.diff, tt.codeFiles, tt.base)
			for _, want := range tt.contains {
				assert.Contains(t, body, want)
			}

			for _, unwanted := range tt.notContains {
				assert.NotContains(t, body, unwanted)
			}
		})
	}
}

func TestValidateCodeFiles(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		file    string
		wantErr bool
	}{
		{"provider test file", "providers/anthropic/alias_test.go", false},
		{"backtick", "providers/anthropic/a`b_test.go", true},
		{"markdown link", "providers/anthropic/[x](https://evil.example).go", true},
		{"space", "providers/anthropic/a b_test.go", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			problems := ValidateCodeFiles([]string{tt.file})
			if tt.wantErr {
				assert.NotEmpty(t, problems)
				return
			}

			assert.Empty(t, problems)
		})
	}
}

func TestRenderCapped(t *testing.T) {
	t.Parallel()

	lines := make([]string, 0, 200)
	for i := range 200 {
		lines = append(lines, fmt.Sprintf("+  \"line\": %03d, \"pad\": %q", i, strings.Repeat("x", 60)))
	}

	diff := strings.Join(lines, "\n")
	report := validReport()

	t.Run("fits: unchanged", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, Render(report, "anthropic", diff, nil, Base{}), RenderCapped(report, "anthropic", diff, nil, Base{}, 1<<20))
	})

	t.Run("too long: diff cut on a line, body within the limit", func(t *testing.T) {
		t.Parallel()

		const limit = 6000

		body := RenderCapped(report, "anthropic", diff, nil, Base{}, limit)
		assert.LessOrEqual(t, len(body), limit)
		assert.Contains(t, body, "### Changes (1)")
		assert.Contains(t, body, "truncated; see the workflow artifact")
		assert.Contains(t, body, lines[0]+"\n")
		assert.NotContains(t, body, lines[len(lines)-1])

		kept := body[strings.Index(body, lines[0]):strings.Index(body, "\n… (truncated")]
		for line := range strings.SplitSeq(kept, "\n") {
			assert.Contains(t, lines, line, "the diff is cut only between lines")
		}
	})
}

// The agent's schema and the validator must accept the same fields and skip
// reasons.
func TestSchemaMatchesValidator(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "report.schema.json"))
	require.NoError(t, err)

	var schema jsonschema.Schema
	require.NoError(t, json.Unmarshal(raw, &schema))

	items := func(array string) *jsonschema.Schema {
		items := schema.Properties[array].Items
		require.NotNil(t, items, "schema has no %s items", array)

		return items
	}
	enum := func(array, property string) []any { return items(array).Properties[property].Enum }

	assert.ElementsMatch(t, reasons, enum("skipped", "reason"), "skip reasons")
	assert.ElementsMatch(t, fields, enum("changes", "field"), "changes fields")
	assert.ElementsMatch(t, fields, enum("needs_human", "field"), "needs_human fields")

	// The agent can name only the rules the schema's pattern accepts.
	for _, array := range []string{"needs_human", "skipped"} {
		assert.Equal(t, reconcile.RuleIDPattern.String(), items(array).Properties["rule"].Pattern, "%s rule pattern", array)
	}
}
