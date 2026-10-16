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
	"slices"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

var hosts = []string{"platform.claude.com"}

// testRules is a RULES.md with a price Default, a copy Default, four skip
// Defaults, one Ask, no-rule for rows no rule covers, and Bedrock's offered
// regions, with `in` asked about.
var testRules = reconcile.Rules{
	Lines:      map[string]int{"price-follow": 30, "new-copy-offered-region": 32, "limited-availability": 40, "replacement-not-carried": 80, "no-rule": 120, "region-not-offered": 20, "copy-gone-no-eol": 22, "label-fact-differs": 24},
	Reasons:    map[string]string{"limited_availability": "the model needs a signup", "conflicting_sources": "no rule uses it yet"},
	Questions:  map[string]string{"replacement-not-carried": "Which replacement?", "no-rule": "No rule covers this. Decide."},
	RuleFields: map[string][]string{"price-follow": {"price"}, "new-copy-offered-region": {"new_copy"}},
	Outcomes: map[string][]string{
		"price-follow": {"change"}, "new-copy-offered-region": {"change"}, "limited-availability": {"skip:`limited_availability`"}, "replacement-not-carried": {"needs_human"}, "no-rule": {"needs_human"},
		"region-not-offered": {"skip:`region_not_offered`"}, "copy-gone-no-eol": {"skip:`copy_unlisted`"},
		"label-fact-differs": {"skip:`fixed_fact_differs`"},
	},
	Offered:    map[string][]string{"bedrock": {"global", "us", "eu", "au", "jp"}},
	NotOffered: map[string]map[string]string{"bedrock": {"in": "asked 2026-10-15"}},
}

func validReport() Report {
	return Report{
		Changes: []Change{{
			Offering: "claude-opus-5-5", Field: "price", Old: "4.00", New: "5.00", Rule: "price-follow",
			SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing.md",
		}},
		NeedsHuman: []NeedsHuman{{
			Offering: "claude-sonnet-5-5", Field: "limit", Values: []string{"65535", "65536"}, Rule: "no-rule",
			SourceURLs: []string{"https://platform.claude.com/docs/en/about-claude/models/overview.md"},
		}},
		Skipped: []Skipped{{Offering: "claude-haiku-4-5", Reason: "conflicting_sources", Rule: "no-rule"}},
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
		{"a question with both sides", func(r *Report) { r.NeedsHuman[0].Catalogue, r.NeedsHuman[0].Page = "65535", "65536" }, ""},
		{"html in the catalogue side", func(r *Report) { r.NeedsHuman[0].Catalogue = "<b>65535</b>" }, `needs_human[0]: bad value "<b>65535</b>"`},
		{"a link on the page side", func(r *Report) { r.NeedsHuman[0].Page = "https://evil.example/x" }, `needs_human[0]: bad value "https://evil.example/x"`},
		{"a long page side", func(r *Report) { r.NeedsHuman[0].Page = strings.Repeat("9", 65) }, "needs_human[0]: bad value"},
		{"rule RULES.md doesn't have", func(r *Report) { r.Skipped[0].Rule = "no-such-rule" }, `skipped[0]: unknown rule "no-such-rule"`},
		{"needs_human rule that never asks a person", func(r *Report) { r.NeedsHuman[0].Rule = "limited-availability" }, "needs_human[0]: rule limited-availability doesn't produce needs_human"},
		{"skip rule for another reason", func(r *Report) { r.Skipped[0].Rule = "limited-availability" }, "skipped[0]: rule limited-availability doesn't produce skip:`conflicting_sources`"},
		{"skip rule for this reason", func(r *Report) {
			r.Skipped[0] = Skipped{Offering: "x", Reason: "limited_availability", Rule: "limited-availability"}
		}, ""},
		{"change without a rule", func(r *Report) { r.Changes[0].Rule = "" }, `changes[0]: unknown rule ""`},
		{"change under a rule for another field", func(r *Report) { r.Changes[0].Field = "limit" }, "changes[0]: rule price-follow covers price, not limit"},
		{"change resting on judgment", func(r *Report) { r.Changes[0].Rule = "no-rule" }, "changes[0]: a change needs a Default, not no-rule"},
		{"new copy", func(r *Report) { r.Changes[0].Field, r.Changes[0].Rule = "new_copy", "new-copy-offered-region" }, ""},
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

// effectSnapshots are the catalogue before and after a patch, cut from #259:
// Sonnet 5.5's cache read halves, an AU copy is added, and Sonnet 4.5's JP
// and US copies retire with the global Sonnet 5.5 as their replacement.
func effectSnapshots(t *testing.T) (*reconcile.Snapshot, *reconcile.Snapshot) {
	t.Helper()

	sonnet45 := func(geo, life string) string {
		return `{"id": "` + geo + `.anthropic.claude-sonnet-4-5", "model": "anthropic/claude-sonnet-4-5",
			"attributes": [{"key": "inference_geo", "value": "` + geo + `"}],
			"capabilities": {"structured_output": true, "tools": true}, "modalities": {"input": ["text", "image"]},
			"constraints": {"supported_params": ["temperature", "top_p", "max_tokens", "stop"]},
			"pricing": {"default": {"base": {"input_microcents_per_million": 330000000}}}, "lifecycle": ` + life + `}`
	}
	sonnet55 := func(geo string, input int) string {
		return fmt.Sprintf(`{"id": "%s.anthropic.claude-sonnet-5-5", "model": "anthropic/claude-sonnet-5-5",
			"attributes": [{"key": "inference_geo", "value": "%s"}],
			"capabilities": {"structured_output": false, "tools": true}, "modalities": {"input": ["text"]},
			"constraints": {"supported_params": ["max_tokens", "stop"]},
			"pricing": {"default": {"base": {"input_microcents_per_million": %d}}}, "lifecycle": {"stage": "ga"}}`, geo, geo, input)
	}
	direct := func(cacheRead int) string {
		return fmt.Sprintf(`{"provider": "anthropic", "offerings": [{"id": "claude-sonnet-5-5", "model": "anthropic/claude-sonnet-5-5",
			"pricing": {"default": {"base": {"input_microcents_per_million": 200000000, "cached_input_microcents_per_million": %d}}}}]}`, cacheRead)
	}
	retiring := `{"stage": "ga", "deprecated": "2026-10-08", "retires": "2027-04-08", "replaced_by": "global.anthropic.claude-sonnet-5-5"}`

	before, err := reconcile.ParseSnapshot([]byte(`{"providers": [` + direct(20000000) + `, {"provider": "aws.bedrock", "offerings": [` +
		strings.Join([]string{sonnet45("jp", `{"stage": "ga"}`), sonnet45("us", `{"stage": "ga"}`), sonnet55("global", 200000000), sonnet55("us", 220000000)}, ",") + `]}]}`))
	require.NoError(t, err)

	after, err := reconcile.ParseSnapshot([]byte(`{"providers": [` + direct(10000000) + `, {"provider": "aws.bedrock", "offerings": [` +
		strings.Join([]string{sonnet45("jp", retiring), sonnet45("us", retiring), sonnet55("global", 200000000), sonnet55("us", 220000000), sonnet55("au", 220000000)}, ",") + `]}]}`))
	require.NoError(t, err)

	return &before, &after
}

// tampered is after with the AU copy's region set to markup, as patched code could.
func tampered(t *testing.T, after *reconcile.Snapshot) *reconcile.Snapshot {
	t.Helper()

	raw, err := json.Marshal(after)
	require.NoError(t, err)

	snap, err := reconcile.ParseSnapshot([]byte(strings.Replace(string(raw), `"value":"au"`, `"value":"[evil](https://evil.example)"`, 1)))
	require.NoError(t, err)

	return &snap
}

// addedSnapshots are the catalogue before and after a patch that adds an AU
// copy of Sonnet 5.5, which the catalogue carries, and Haiku 5.5, which it
// doesn't, on the Anthropic API and Bedrock.
func addedSnapshots(t *testing.T) (*reconcile.Snapshot, *reconcile.Snapshot) {
	t.Helper()

	const sonnet = `{"id": "global.anthropic.claude-sonnet-5-5", "model": "anthropic/claude-sonnet-5-5"}`

	before, err := reconcile.ParseSnapshot([]byte(`{"facts": {"anthropic/claude-sonnet-5-5": {"released": "2026-09-30"}},
		"providers": [{"provider": "aws.bedrock", "offerings": [` + sonnet + `]}]}`))
	require.NoError(t, err)

	after, err := reconcile.ParseSnapshot([]byte(`{"facts": {"anthropic/claude-sonnet-5-5": {"released": "2026-09-30"},
			"anthropic/claude-haiku-5-5": {"released": "2026-10-15"}},
		"providers": [{"provider": "anthropic", "offerings": [{"id": "claude-haiku-5-5", "model": "anthropic/claude-haiku-5-5"}]},
			{"provider": "aws.bedrock", "offerings": [` + sonnet + `,
				{"id": "au.anthropic.claude-sonnet-5-5", "model": "anthropic/claude-sonnet-5-5"},
				{"id": "global.anthropic.claude-haiku-5-5", "model": "anthropic/claude-haiku-5-5"}]}]}`))
	require.NoError(t, err)

	return &before, &after
}

func TestValidateAdded(t *testing.T) {
	t.Parallel()

	before, after := addedSnapshots(t)
	withSnapshots := Base{Before: before, After: after}

	tests := []struct {
		name        string
		change      Change
		base        Base
		wantProblem string
	}{
		{
			name:        "a copy reported as a new model",
			change:      Change{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_model"},
			base:        withSnapshots,
			wantProblem: `changes[0]: the catalogue already carries "anthropic/claude-sonnet-5-5", so the field is new_copy, not new_model`,
		},
		{
			name:        "a new model reported as a copy",
			change:      Change{Offering: "claude-haiku-5-5", Field: "new_copy"},
			base:        withSnapshots,
			wantProblem: `changes[0]: the catalogue doesn't carry "anthropic/claude-haiku-5-5", so the field is new_model, not new_copy`,
		},
		{name: "a copy", change: Change{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_copy"}, base: withSnapshots},
		{name: "a new model", change: Change{Offering: "claude-haiku-5-5", Field: "new_model"}, base: withSnapshots},
		{name: "a new model's Bedrock copy", change: Change{Offering: "bedrock:global.anthropic.claude-haiku-5-5", Field: "new_model"}, base: withSnapshots},
		{name: "an offering not in the patched catalogue", change: Change{Offering: "claude-opus-5-5", Field: "new_copy"}, base: withSnapshots},
		{name: "another field", change: Change{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "price"}, base: withSnapshots},
		{name: "no snapshots, no check", change: Change{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_model"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			problems := tt.base.ValidateAdded(Report{Changes: []Change{tt.change}}, "anthropic")
			if tt.wantProblem == "" {
				assert.Empty(t, problems)
				return
			}

			assert.Equal(t, []string{tt.wantProblem}, problems)
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
		Offering: "claude-opus-4-1", Field: "replacement", Rule: "replacement-not-carried",
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
	holdCopy := holdOld
	holdCopy.Field = reconcile.FieldNewCopy
	otherField := keepLimit
	otherField.Field = "price"
	// #259's copies in India, which Bedrock doesn't offer, and one in a second region.
	waiting := []Skipped{
		{Offering: "bedrock:in.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"},
		{Offering: "bedrock:in.anthropic.claude-sonnet-5", Reason: "region_not_offered", Rule: "region-not-offered"},
		{Offering: "bedrock:ca.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"},
		{Offering: "bedrock:in.anthropic.claude-haiku-4-5-20251001-v1:0", Reason: "region_not_offered", Rule: "region-not-offered"},
	}
	declinedCA := testRules
	declinedCA.NotOffered = map[string]map[string]string{"bedrock": {"in": "asked 2026-10-15", "ca": "declined 2026-10-20"}}
	withWaiting := validReport()
	withWaiting.Skipped = append(withWaiting.Skipped, waiting...)

	before, after := effectSnapshots(t)
	withEffects := Base{Before: before, After: after, Today: "2026-10-15"}
	addedBefore, addedAfter := addedSnapshots(t)
	routesBefore, err := ParseRoutes([]byte("ap-southeast-2 anthropic.claude-sonnet-5-5 global.anthropic.claude-sonnet-5-5\n" +
		"ap-southeast-4 anthropic.claude-sonnet-5-5 global.anthropic.claude-sonnet-5-5\nus-east-1 anthropic.claude-sonnet-5-5 us.anthropic.claude-sonnet-5-5\n"))
	require.NoError(t, err)
	routesAfter, err := ParseRoutes([]byte("ap-southeast-2 anthropic.claude-sonnet-5-5 au.anthropic.claude-sonnet-5-5\n" +
		"ap-southeast-4 anthropic.claude-sonnet-5-5 au.anthropic.claude-sonnet-5-5\nus-east-1 anthropic.claude-sonnet-5-5 us.anthropic.claude-sonnet-5-5\n"))
	require.NoError(t, err)

	withRoutes := withEffects
	withRoutes.RoutesBefore, withRoutes.RoutesAfter = routesBefore, routesAfter

	// A pair the SDK refused that now has a copy, and one the patch removes.
	failsBefore, err := ParseRoutes([]byte("us-east-1 anthropic.claude-sonnet-5-5 -\nus-west-2 anthropic.claude-haiku-5-5 us.anthropic.claude-haiku-5-5\n"))
	require.NoError(t, err)
	failsAfter, err := ParseRoutes([]byte("us-east-1 anthropic.claude-sonnet-5-5 global.anthropic.claude-sonnet-5-5\n"))
	require.NoError(t, err)

	withFails := withEffects
	withFails.RoutesBefore, withFails.RoutesAfter = failsBefore, failsAfter
	// Seven questions, and two waiting regions: one with a name, one without.
	manyAsks := Report{Skipped: []Skipped{
		{Offering: "bedrock:us-gov.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"},
		{Offering: "bedrock:zz.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"},
	}}

	for _, model := range []string{"claude-opus-5", "claude-opus-5-5", "claude-sonnet-5", "claude-sonnet-5-5", "claude-haiku-5", "claude-haiku-5-5", "claude-fable-5"} {
		manyAsks.NeedsHuman = append(manyAsks.NeedsHuman, NeedsHuman{Offering: model, Field: "limit", Rule: "no-rule", SourceURLs: []string{"https://platform.claude.com/a"}})
	}

	changes := func(rows ...Change) Report {
		for i := range rows {
			rows[i].SourceURL = "https://platform.claude.com/docs/en/about-claude/pricing"
		}

		return Report{Changes: rows}
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
			name:     "price note with percent",
			report:   changes(Change{Offering: "claude-sonnet-5-5", Field: "price", Old: "cache read 0.20", New: "cache read 0.10"}),
			base:     withEffects,
			contains: []string{"| Offering | Field | Old | New | Effect | Source |", "| `cache read 0.10` | cache read 0.2 → 0.1 (-50%) |"},
		},
		{
			name:     "notes ignore report values",
			report:   changes(Change{Offering: "claude-sonnet-5-5", Field: "price", Old: "input 9.00", New: "input 1.00"}),
			base:     withEffects,
			contains: []string{"| `input 1.00` | cache read 0.2 → 0.1 (-50%) |"},
		},
		{
			name:     "retirement note counts days",
			report:   changes(Change{Offering: "bedrock:us.anthropic.claude-sonnet-4-5", Field: "lifecycle", Old: "ga", New: "retires 2027-04-08"}),
			base:     withEffects,
			contains: []string{"| retires 2027-04-08, in 175 days |"},
		},
		{
			name:     "copies added per host",
			report:   changes(Change{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_copy", New: "au.anthropic.claude-sonnet-5-5"}),
			base:     withEffects,
			contains: []string{"| new on bedrock, region `au` |"},
		},
		{
			name: "jp replacement flagged",
			report: changes(Change{
				Offering: "bedrock:jp.anthropic.claude-sonnet-4-5", Field: "replacement", New: "global.anthropic.claude-sonnet-5-5",
			}),
			base: withEffects,
			contains: []string{"| ⚑ no `jp` copy of `anthropic/claude-sonnet-5-5`: callers who follow it leave `jp`; input -39%; " +
				"loses `image input`, `structured_output`, `temperature`, `top_p` |"},
		},
		{
			name: "changes grouped by rule",
			report: Report{Changes: []Change{
				{Offering: "vertex:claude-sonnet-5-5", Field: "price", Old: "0.22", New: "0.11", Rule: "price-follow", SourceURL: "https://platform.claude.com/a"},
				{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_copy", New: "au.anthropic.claude-sonnet-5-5", Rule: "new-copy-offered-region", SourceURL: "https://platform.claude.com/b"},
				{Offering: "claude-sonnet-5-5", Field: "price", Old: "0.20", New: "0.10", Rule: "price-follow", SourceURL: "https://platform.claude.com/c"},
			}},
			base: Base{URL: baseURL, Rules: testRules},
			contains: []string{
				"### Changes (3)\n\n**[price-follow](" + rulesURL + "#L30)** (2)\n\n| Offering | Field | Old | New | Source |\n|---|---|---|---|---|\n" +
					"| `claude-sonnet-5-5` | price | `0.20` | `0.10` | [platform.claude.com](https://platform.claude.com/c) |\n" +
					"| `vertex:claude-sonnet-5-5` | price | `0.22` | `0.11` | [platform.claude.com](https://platform.claude.com/a) |\n\n" +
					"**[new-copy-offered-region](" + rulesURL + "#L32)** (1)\n\n| Offering |",
			},
		},
		{
			name:   "a route that moves is flagged",
			report: changes(Change{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_copy", New: "au.anthropic.claude-sonnet-5-5"}),
			base:   withRoutes,
			contains: []string{
				"### Routing changes (1)\n\n⚑ Bare-name calls from these regions now go to another copy. Check the price and where the data is processed.\n\n" +
					"| Model | Callers in | Today | After | Input price | Data processed in |\n|---|---|---|---|---|---|\n" +
					"| `anthropic.claude-sonnet-5-5` | `ap-southeast-2`, `ap-southeast-4` | `global.anthropic.claude-sonnet-5-5` | `au.anthropic.claude-sonnet-5-5` | +10% | Australia |\n\n",
			},
			notContains: []string{"us-east-1"},
		},
		{
			name:   "a route that had no copy, or loses it, reads as a call that fails",
			report: changes(Change{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_copy", New: "au.anthropic.claude-sonnet-5-5"}),
			base:   withFails,
			contains: []string{
				"### Routing changes (2)\n\n",
				"|---|---|---|---|---|---|\n" +
					"| `anthropic.claude-haiku-5-5` | `us-west-2` | `us.anthropic.claude-haiku-5-5` | call fails (no copy) | — | — |\n" +
					"| `anthropic.claude-sonnet-5-5` | `us-east-1` | call fails (no copy) | `global.anthropic.claude-sonnet-5-5` | — | any region |\n\n",
			},
			notContains: []string{"refused"},
		},
		{
			name:        "unsafe snapshot strings stay out of the body",
			report:      changes(Change{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_copy", New: "au.anthropic.claude-sonnet-5-5"}),
			base:        Base{Before: before, After: tampered(t, after), Today: "2026-10-15"},
			contains:    []string{"| new on bedrock |"},
			notContains: []string{"evil"},
		},
		{
			name: "us replacement not flagged",
			report: changes(Change{
				Offering: "bedrock:us.anthropic.claude-sonnet-4-5", Field: "replacement", New: "global.anthropic.claude-sonnet-5-5",
			}),
			base:        withEffects,
			contains:    []string{"| `global.anthropic.claude-sonnet-5-5` |  |"},
			notContains: []string{"⚑"},
		},
		{
			name:   "full report",
			report: validReport(),
			diff:   diff,
			contains: []string{
				"## Catalogue drift: anthropic",
				"| `claude-opus-5-5` | price | `4.00` | `5.00` | [platform.claude.com](https://platform.claude.com/docs/en/about-claude/pricing.md) |",
				"`65535`, `65536`",
				"| ⚠ `no-rule` | `conflicting_sources` | `claude-haiku-4-5` |",
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
			name:   "says not to push to the branch, and how to settle a row",
			report: validReport(),
			diff:   diff,
			base:   Base{Branch: "main"},
			contains: []string{"Don't push commits to this branch: each run rebuilds it from main and force-pushes, so they would be lost. " +
				"To settle a row, merge a fix on main or record a decision in `.github/reconcile-models/decisions.yaml`."},
			notContains: []string{"Comment on the PR instead", "Recorded decisions"},
		},
		{
			name:        "names the branch a run from another branch rebuilds from",
			report:      validReport(),
			diff:        diff,
			base:        Base{Branch: "feature-x"},
			contains:    []string{"each run rebuilds it from feature-x and force-pushes", "merge a fix on feature-x or record a decision"},
			notContains: []string{"from main", "on main"},
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
			notContains: []string{"### Resolve before merging"},
		},
		{
			name:        "a skipped offering under a hold is held",
			report:      Report{Skipped: []Skipped{{Offering: "vertex:claude-haiku-4-5", Reason: "out_of_scope"}}},
			base:        Base{URL: baseURL, Decisions: []reconcile.Decision{holdOld}, Today: "2026-10-14"},
			contains:    []string{"| [DEC-002](" + decisionsURL + "#L12) | `vertex:claude-haiku-4-5` | new_model | out_of_scope | Not offered to users. |"},
			notContains: []string{"### Skipped"},
		},
		{
			name:        "a skipped copy under a hold on new_copy is held",
			report:      Report{Skipped: []Skipped{{Offering: "vertex:claude-haiku-4-5", Reason: "out_of_scope"}}},
			base:        Base{URL: baseURL, Decisions: []reconcile.Decision{holdCopy}, Today: "2026-10-14"},
			contains:    []string{"| [DEC-002](" + decisionsURL + "#L12) | `vertex:claude-haiku-4-5` | new_copy | out_of_scope | Not offered to users. |"},
			notContains: []string{"### Skipped"},
		},
		{
			name:     "a new model is new on its host",
			report:   changes(Change{Offering: "claude-haiku-5-5", Field: "new_model", New: "claude-haiku-5-5"}),
			base:     Base{Before: addedBefore, After: addedAfter},
			contains: []string{"| new_model | none | `claude-haiku-5-5` | new on anthropic |"},
		},
		{
			name:   "needs a human is a block per row, marked when no rule covers it",
			report: twoAsks,
			base:   Base{URL: baseURL, Rules: testRules},
			contains: []string{
				"### Resolve before merging (2): 1 by a rule, 1 with no rule",
				"**1. `claude-sonnet-5-5` · limit · ⚠ no rule covers this ([no-rule](" + rulesURL + "#L120))**",
				"- **Also found:** `65535`, `65536`\n- **Question:** No rule covers this. Decide.\n",
				"- **Sources:** [platform.claude.com](https://platform.claude.com/docs/en/about-claude/models/overview.md)",
				"**2. `claude-opus-4-1` · replacement · rule [replacement-not-carried](" + rulesURL + "#L80)**\n- **Question:** Which replacement?\n- **Sources:**",
			},
			notContains: []string{"| Offering | Field | Values |"},
		},
		{
			name: "skipped rows are grouped by rule",
			report: Report{Skipped: []Skipped{
				{Offering: "claude-mythos-5", Reason: "limited_availability", Rule: "limited-availability"},
				{Offering: "claude-haiku-4-5", Reason: "conflicting_sources", Rule: "no-rule"},
				{Offering: "claude-mythos-5-1", Reason: "limited_availability", Rule: "limited-availability"},
			}},
			base: Base{URL: baseURL, Rules: testRules},
			contains: []string{
				"### Skipped (3)\n\n| Rule | Why | Offerings |",
				"| [limited-availability](" + rulesURL + "#L40) | the model needs a signup (`limited_availability`) | `claude-mythos-5`, `claude-mythos-5-1` |",
				"| ⚠ [no-rule](" + rulesURL + "#L120) | no rule uses it yet (`conflicting_sources`) | `claude-haiku-4-5` |",
			},
		},
		{
			name:   "one waiting line per region, with since when it has been asked",
			report: Report{Skipped: waiting},
			base:   Base{URL: baseURL, Rules: testRules},
			contains: []string{
				"### Waiting on a product decision (2)\n\nModel cards list copies in regions this catalogue doesn't offer, so the bot left them out " +
					"(rule [region-not-offered](" + rulesURL + "#L20)). Nothing changes until someone decides, and they don't block the PR. " +
					"To answer, either offer the region (move its prefix to Offered in [RULES.md](" + baseURL + "/" + reconcile.RulesPath + "), with its code, in a reviewed PR) " +
					"or decline it (write it under Not offered as `` `<prefix>` (declined YYYY-MM-DD) ``, and its copies show under Skipped from then on).\n\n",
				"- Offer `in` on bedrock (asked since 2026-10-15)? 3 copies: `bedrock:in.anthropic.claude-opus-5`, `bedrock:in.anthropic.claude-sonnet-5`, " +
					"`bedrock:in.anthropic.claude-haiku-4-5-20251001-v1:0`\n",
				"- Offer `ca` on bedrock (not in the table yet)? 1 copy: `bedrock:ca.anthropic.claude-opus-5`\n",
				"> [!NOTE]\n> **2 product decisions waiting** (they don't block this PR): " +
					"offer `in` on bedrock (India, asked since 2026-10-15)? offer `ca` on bedrock (Canada, new)?\n\n",
			},
			notContains: []string{"[!IMPORTANT]"},
		},
		{
			name:   "a declined region's copies are skipped, not waiting",
			report: Report{Skipped: waiting},
			base:   Base{URL: baseURL, Rules: declinedCA},
			contains: []string{
				"### Waiting on a product decision (1)",
				"### Skipped (1)\n\n| Rule | Why | Offerings |\n|---|---|---|\n| [region-not-offered](" + rulesURL + "#L20) | `region_not_offered` | `bedrock:ca.anthropic.claude-opus-5` |\n",
				"> [!NOTE]\n> **1 product decision waiting** (it doesn't block this PR): offer `in` on bedrock (India, asked since 2026-10-15)?\n\n",
			},
			notContains: []string{"Offer `ca`", "`ca` on bedrock"},
		},
		{
			name:   "questions for people come after the intro, before the first section",
			report: withWaiting,
			base:   Base{Rules: testRules},
			contains: []string{
				"Later runs don't read PR comments.\n\n> [!IMPORTANT]\n> **Resolve 1 question before merging:** `claude-sonnet-5-5` limit. See Resolve before merging below.\n\n" +
					"> [!NOTE]\n> **2 product decisions waiting** (they don't block this PR): " +
					"offer `in` on bedrock (India, asked since 2026-10-15)? offer `ca` on bedrock (Canada, new)?\n\n### Changes (1)",
			},
		},
		{
			name:        "held rows aren't questions",
			report:      twoAsks,
			base:        Base{Decisions: []reconcile.Decision{keepLimit}, Today: "2026-10-14"},
			contains:    []string{"> [!IMPORTANT]\n> **Resolve 1 question before merging:** `claude-opus-4-1` replacement. See Resolve before merging below.\n\n"},
			notContains: []string{"`claude-sonnet-5-5` limit", "[!NOTE]"},
		},
		{
			name:   "more than five questions are cut, and a region with no name shows none",
			report: manyAsks,
			base:   Base{Rules: testRules},
			contains: []string{
				"> [!IMPORTANT]\n> **Resolve 7 questions before merging:** `claude-opus-5` limit, `claude-opus-5-5` limit, `claude-sonnet-5` limit, " +
					"`claude-sonnet-5-5` limit, `claude-haiku-5` limit and 2 more. See Resolve before merging below.\n\n",
				"> [!NOTE]\n> **2 product decisions waiting** (they don't block this PR): " +
					"offer `us-gov` on bedrock (GovCloud (US), new)? offer `zz` on bedrock (new)?\n\n",
			},
		},
		{
			name:        "no questions, no alerts",
			report:      Report{Skipped: waiting[2:3]},
			base:        Base{Rules: declinedCA},
			contains:    []string{"### Skipped (1)"},
			notContains: []string{"[!IMPORTANT]", "[!NOTE]", "### Waiting on a product decision"},
		},
		{
			name:        "waiting rows aren't listed as skipped",
			report:      Report{Skipped: append([]Skipped{{Offering: "claude-mythos-5", Reason: "limited_availability", Rule: "limited-availability"}}, waiting...)},
			base:        Base{URL: baseURL, Rules: testRules},
			contains:    []string{"### Skipped (1)"},
			notContains: []string{"`region_not_offered`"},
		},
		{
			name:        "a decision on another field holds nothing",
			report:      validReport(),
			base:        Base{URL: baseURL, Decisions: []reconcile.Decision{otherField}, Today: "2026-10-14"},
			contains:    []string{"### Resolve before merging (1)"},
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

	t.Run("notes stay when the diff is cut", func(t *testing.T) {
		t.Parallel()

		before, after := effectSnapshots(t)
		priced := Report{Changes: []Change{{
			Offering: "claude-sonnet-5-5", Field: "price", Old: "cache read 0.20", New: "cache read 0.10",
			SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing",
		}}}

		body := RenderCapped(priced, "anthropic", diff, nil, Base{Before: before, After: after}, 6000)
		assert.Contains(t, body, "truncated; see the workflow artifact")
		assert.Contains(t, body, "| cache read 0.2 → 0.1 (-50%) |")
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

	// Every row names a rule, and the agent can name only the rules the
	// schema's pattern accepts.
	for _, array := range []string{"changes", "needs_human", "skipped"} {
		assert.Contains(t, items(array).Required, "rule", "%s rows require a rule", array)
		require.Contains(t, items(array).Properties, "rule", "%s rows have no rule", array)
		assert.Equal(t, reconcile.RuleIDPattern.String(), items(array).Properties["rule"].Pattern, "%s rule pattern", array)
	}
}

// The schema takes a question's two sides as the same short values the
// validator takes, and the location field on a question.
func TestSchemaValues(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "report.schema.json"))
	require.NoError(t, err)

	var schema jsonschema.Schema
	require.NoError(t, json.Unmarshal(raw, &schema))

	resolved, err := schema.Resolve(nil)
	require.NoError(t, err)

	question := func(field, sides string) string {
		return `{"changes": [], "skipped": [], "needs_human": [{"offering": "claude-sonnet-5-5", "field": "` + field + `", "values": [], ` +
			`"source_urls": ["https://platform.claude.com/a"], "rule": "no-rule"` + sides + `}]}`
	}

	tests := []struct {
		name, report string
		wantErr      bool
	}{
		{"no sides", question("limit", ""), false},
		{"both sides", question("limit", `, "catalogue": "65535", "page": "65536"`), false},
		{"html in a side", question("limit", `, "catalogue": "<b>65535</b>"`), true},
		{"a long side", question("limit", `, "page": "`+strings.Repeat("9", 65)+`"`), true},
		{"a location question", question("location", `, "catalogue": "global us-east5", "page": "global us-east5 europe-west1"`), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var instance any
			require.NoError(t, json.Unmarshal([]byte(tt.report), &instance))

			if err := resolved.Validate(instance); tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestRenderQuestionBlock checks one Resolve before merging row: the two
// sides first, then other values, then the question, then the sources.
func TestRenderQuestionBlock(t *testing.T) {
	t.Parallel()

	const (
		noRule  = "**1. `claude-sonnet-5-5` · limit · ⚠ no rule covers this (`no-rule`)**\n"
		sources = "- **Sources:** [platform.claude.com](https://platform.claude.com/a)\n\n"
	)

	ask := func(rule, catalogue, page string, values ...string) NeedsHuman {
		return NeedsHuman{
			Offering: "claude-sonnet-5-5", Field: "limit", Catalogue: catalogue, Page: page, Values: values,
			Rule: rule, SourceURLs: []string{"https://platform.claude.com/a"},
		}
	}

	tests := []struct {
		name string
		item NeedsHuman
		want string
	}{
		{
			name: "no rule, both sides: keep or change",
			item: ask("no-rule", "65535", "65536", "providers/anthropic/models.go"),
			want: noRule + "- **Catalogue:** `65535`\n- **Page:** `65536`\n- **Also found:** `providers/anthropic/models.go`\n" +
				"- **Decide:** keep `65535`, or change to `65536`. If this can come back, add a rule.\n" + sources,
		},
		{
			name: "both sides and nothing else found",
			item: ask("no-rule", "65535", "65536"),
			want: noRule + "- **Catalogue:** `65535`\n- **Page:** `65536`\n- **Decide:** keep `65535`, or change to `65536`. If this can come back, add a rule.\n" + sources,
		},
		{
			name: "a rule's row keeps its question",
			item: ask("replacement-not-carried", "none", "claude-opus-5"),
			want: "**1. `claude-sonnet-5-5` · limit · rule `replacement-not-carried`**\n- **Catalogue:** `none`\n- **Page:** `claude-opus-5`\n" +
				"- **Question:** Which replacement?\n" + sources,
		},
		{
			name: "one side alone keeps the question",
			item: ask("no-rule", "", "65536"),
			want: noRule + "- **Page:** `65536`\n- **Question:** No rule covers this. Decide.\n" + sources,
		},
		{
			name: "no sides: also found, then the question",
			item: ask("no-rule", "", "", "65535", "65536"),
			want: noRule + "- **Also found:** `65535`, `65536`\n- **Question:** No rule covers this. Decide.\n" + sources,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := Render(Report{NeedsHuman: []NeedsHuman{tt.item}}, "anthropic", "", nil, Base{Rules: testRules})
			assert.Contains(t, body, tt.want)
		})
	}
}

// TestRenderNotesOnce checks the catalogue-written notes against the rows:
// two rows on one offering and field share one note, and an offering the
// patch changes with no row of its own is still named.
func TestRenderNotesOnce(t *testing.T) {
	t.Parallel()

	before, after := effectSnapshots(t)
	source := "https://platform.claude.com/docs/en/about-claude/pricing"
	report := Report{Changes: []Change{
		{Offering: "claude-sonnet-5-5", Field: "price", Old: "cache read 0.20", New: "cache read 0.10", Rule: "price-follow", SourceURL: source},
		{Offering: "claude-sonnet-5-5", Field: "price", Old: "input 2.00", New: "input 2.00", Rule: "price-follow", SourceURL: source},
	}}

	body := Render(report, "anthropic", "", nil, Base{Before: before, After: after, Today: "2026-10-15"})

	t.Run("a repeated note shows once", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, 1, strings.Count(body, "cache read 0.2 → 0.1 (-50%)"))
	})

	t.Run("an offering changed with no row is named", func(t *testing.T) {
		t.Parallel()
		assert.Contains(t, body, "### Changed with no row (3)\n\nThe patch changes these offerings, but the report has no row for them. "+
			"Check them in the snapshot diff.\n\n- `bedrock:au.anthropic.claude-sonnet-5-5`, `bedrock:jp.anthropic.claude-sonnet-4-5`, `bedrock:us.anthropic.claude-sonnet-4-5`\n")
		assert.NotContains(t, body, "- `claude-sonnet-5-5`", "it has a row")
	})
}

func TestUnreported(t *testing.T) {
	t.Parallel()

	snapshot := func(released, id string) *reconcile.Snapshot {
		snap, err := reconcile.ParseSnapshot([]byte(`{"facts": {"anthropic/claude-sonnet-5-5": {"released": "` + released + `"}},
			"providers": [{"provider": "anthropic", "offerings": [{"id": "claude-sonnet-5-5", "model": "anthropic/claude-sonnet-5-5"}, {"id": "` + id + `"}]}]}`))
		require.NoError(t, err)

		return &snap
	}

	t.Run("a model's facts changed with no row", func(t *testing.T) {
		t.Parallel()

		base := Base{Before: snapshot("2026-09-30", "x"), After: snapshot("2026-09-29", "x")}
		assert.Equal(t, []string{"anthropic/claude-sonnet-5-5 (facts)"}, base.unreported(Report{}, "anthropic"))
		assert.Empty(t, base.unreported(Report{Changes: []Change{{Offering: "claude-sonnet-5-5"}}}, "anthropic"), "a row on one of its offerings covers it")
	})

	t.Run("an unprintable ID is counted", func(t *testing.T) {
		t.Parallel()

		base := Base{Before: snapshot("2026-09-30", "x"), After: snapshot("2026-09-30", "<b>evil</b>")}
		assert.Equal(t, []string{"x (removed)", "1 offering with an ID that isn't safe to print"}, base.unreported(Report{}, "anthropic"))
	})
}

// TestRenderSuggestedRules uses #259's three Sonnet 4.5 lifecycle rows, which
// went to a person under no-rule before the lifecycle Defaults existed.
func TestRenderSuggestedRules(t *testing.T) {
	t.Parallel()

	found := []string{"Legacy 2026-10-08", "EOL 2027-04-08", "AWS names no replacement"}
	ask := func(geo string, values ...string) NeedsHuman {
		return NeedsHuman{
			Offering: "bedrock:" + geo + ".anthropic.claude-sonnet-4-5-20250929-v1:0", Field: "lifecycle", Rule: "no-rule",
			Values: values, SourceURLs: []string{"https://platform.claude.com/x"},
		}
	}
	held := reconcile.Decision{
		ID: "DEC-009", Offering: "bedrock:eu.anthropic.claude-sonnet-4-5-20250929-v1:0", Field: "lifecycle", Kind: reconcile.KindKeep,
		Reason: "Agreed in review.", ReviewBy: "2027-01-13",
	}

	tests := []struct {
		name        string
		asks        []NeedsHuman
		decisions   []reconcile.Decision
		contains    []string
		notContains []string
	}{
		{
			name: "two no-rule rows on one field suggest a rule",
			asks: []NeedsHuman{ask("global", found...), ask("us", found...), ask("eu", slices.Concat(found, []string{"only on eu"})...)},
			contains: []string{
				"### Suggested rules (1)\n\nSome rows under Resolve before merging went to a person under no-rule and share a field. If the case can come back, add a Default to RULES.md: give it a slug, write When as the condition, and replace each ?.\n\n" +
					"| Default | Field | When | Bot does | Why | Outcome |\n|---|---|---|---|---|---|\n" +
					"| new-default | `lifecycle` | 3 rows this run, all with `Legacy 2026-10-08`, `EOL 2027-04-08`, `AWS names no replacement` | ? | ? | ? |\n",
			},
		},
		{name: "one row suggests nothing", asks: []NeedsHuman{ask("global", found...)}, notContains: []string{"Suggested rules"}},
		{
			name: "held rows don't count", asks: []NeedsHuman{ask("global", found...), ask("eu", found...)}, decisions: []reconcile.Decision{held},
			notContains: []string{"Suggested rules"},
		},
	}

	t.Run("a pasted row fails until a person fills it in", func(t *testing.T) {
		t.Parallel()

		lines := suggestions([]NeedsHuman{ask("global", found...), ask("us", found...)})
		require.Len(t, lines, 1)

		_, problems := reconcile.ParseRules([]byte("- **Outcome.** `change` (a hunk)\n\n| Field | Covers |\n|---|---|\n| `lifecycle` | dates |\n\n" +
			"| Default | Field | When | Bot does | Why | Outcome |\n|---|---|---|---|---|---|\n" + lines[0]))
		assert.Contains(t, problems, "new-default: unknown outcome ?")
	})

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := Render(Report{NeedsHuman: tt.asks}, "anthropic", "", nil, Base{Rules: testRules, Decisions: tt.decisions, Today: "2026-10-15"})
			for _, want := range tt.contains {
				assert.Contains(t, body, want)
			}

			for _, unwanted := range tt.notContains {
				assert.NotContains(t, body, unwanted)
			}
		})
	}
}

func TestPercent(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "-50%", percent(20000000, 10000000))
	assert.Equal(t, "-39%", percent(330000000, 200000000), "rounded, not cut")
	assert.Equal(t, "+0.4%", percent(250000000, 251000000), "a small change keeps a decimal")
}

func TestParseRoutes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name, routes, wantErr string
	}{
		{"routes and refusals", "us-east-1 anthropic.claude-sonnet-5-5 us.anthropic.claude-sonnet-5-5\nus-gov-west-1 anthropic.claude-sonnet-5-5 -\n", ""},
		{"malformed routes fail closed", "us-east-1 anthropic.claude-sonnet-5-5\n", "route line 1"},
		{"markup in a route", "us-east-1 anthropic.claude-sonnet-5-5 [x](https://evil.example)\n", "route line 1"},
		{"a bad region", "Sydney anthropic.claude-sonnet-5-5 au.anthropic.claude-sonnet-5-5\n", "route line 1"},
		{"a pair twice", "us-east-1 a.b us.a.b\nus-east-1 a.b global.a.b\n", "route line 2"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := ParseRoutes([]byte(tt.routes))
			if tt.wantErr == "" {
				assert.NoError(t, err)
				return
			}

			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}
