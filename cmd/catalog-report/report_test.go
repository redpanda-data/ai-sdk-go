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
	"strconv"
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
	Lines:      map[string]int{"price-follow": 30, "new-copy-offered-region": 32, "limited-availability": 40, "out-of-reach": 80, "no-rule": 120, "region-not-offered": 20, "copy-gone-no-eol": 22, "label-fact-differs": 24},
	Reasons:    map[string]string{"limited_availability": "the model needs a signup", "vendor_excluded": "the vendor isn't tracked"},
	Questions:  map[string]string{"out-of-reach": "Which replacement?", "no-rule": "No rule covers this. Decide."},
	RuleFields: map[string][]string{"price-follow": {"price"}, "new-copy-offered-region": {"new_copy"}},
	Outcomes: map[string][]string{
		"price-follow": {"change"}, "new-copy-offered-region": {"change"}, "limited-availability": {"skip:`limited_availability`"}, "out-of-reach": {"needs_human"}, "no-rule": {"needs_human"},
		"region-not-offered": {"skip:`region_not_offered`"}, "copy-gone-no-eol": {"skip:`copy_unlisted`"},
		"label-fact-differs": {"skip:`fixed_fact_differs`"},
	},
	Offered:    map[string][]string{"bedrock": {"global", "us", "eu", "au", "jp"}},
	NotOffered: map[string]map[string]string{"bedrock": {"in": "asked 2026-10-15"}},
}

// oneVendor is the Anthropic-only vendor list the tests written before
// Render took several vendors use, with base's catalogue as Anthropic's.
func oneVendor(report Report, base Base) []Vendor {
	return []Vendor{{Name: "anthropic", Report: report, Before: base.Before, After: base.After}}
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
		Skipped: []Skipped{{Offering: "claude-haiku-4-5", Reason: "vendor_excluded", Rule: "no-rule"}},
	}
}

// keepLimit is a decision that answers validReport's limit question.
var keepLimit = reconcile.Decision{
	ID: "DEC-001", Offering: "anthropic:claude-sonnet-5-5", Field: "limit", Kind: reconcile.KindKeep,
	Reason: "Agreed in review.", ReviewBy: "2027-01-13", Line: 3,
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
		{"a versioned Vertex offering", func(r *Report) { r.Changes[0].Offering = "vertex:claude-opus-4-5@20251101" }, ""},
		{"a versioned Vertex ID as a value", func(r *Report) { r.Changes[0].New = "claude-sonnet-4-5@20250929" }, ""},
		{"an @ without a date version", func(r *Report) { r.Changes[0].Offering = "vertex:claude-opus-4-5@latest" }, "offering"},
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
		{"skip rule for another reason", func(r *Report) { r.Skipped[0].Rule = "limited-availability" }, "skipped[0]: rule limited-availability doesn't produce skip:`vendor_excluded`"},
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
		Offering: "claude-opus-4-1", Field: "replacement", Rule: "out-of-reach",
		SourceURLs: []string{"https://platform.claude.com/docs/en/about-claude/model-deprecations.md"},
	})
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
	pastRetirement := withEffects
	pastRetirement.Today = "2027-05-01"
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
		// Each page differs, so the rows stay seven questions.
		manyAsks.NeedsHuman = append(manyAsks.NeedsHuman, NeedsHuman{Offering: model, Field: "limit", Rule: "no-rule", Page: model, SourceURLs: []string{"https://platform.claude.com/a"}})
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
			name:     "retirement note after the date",
			report:   changes(Change{Offering: "bedrock:us.anthropic.claude-sonnet-4-5", Field: "lifecycle", Old: "ga", New: "retires 2027-04-08"}),
			base:     pastRetirement,
			contains: []string{"| retired 2027-04-08 |"},
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
				"| ⚠ `no-rule` | `vendor_excluded` | `claude-haiku-4-5` |",
				"\n## Snapshot diff\n\n<details><summary>Snapshot diff</summary>\n\n```diff",
			},
		},
		{
			name:        "no drift",
			report:      Report{},
			contains:    []string{"No drift found"},
			notContains: []string{"```diff", "## Snapshot diff"},
		},
		{
			name:        "empty report keeps diff",
			report:      Report{},
			diff:        diff,
			contains:    []string{"```diff", "+  \"input\": 5"},
			notContains: []string{"No drift found"},
		},
		{
			name:   "says what a pushed commit does, and how to settle a row",
			report: validReport(),
			diff:   diff,
			base:   Base{Branch: "main"},
			contains: []string{
				"Each run rebuilds this branch from main, leaving out a clean \"Update branch\" merge. A commit you push here is carried onto each rebuild while it still applies exactly; if it doesn't, or it's another merge commit or changes `.github/`, runs stop updating this PR and leave a comment saying why. " +
					"To settle a row, merge a fix on main or record a decision in `.github/reconcile-models/decisions.yaml`.",
			},
			notContains: []string{"Don't push", "Comment on the PR instead", "Recorded decisions"},
		},
		{
			name:     "links the review guide at the base commit",
			report:   validReport(),
			diff:     diff,
			base:     Base{URL: baseURL, Branch: "main"},
			contains: []string{"[REVIEW.md](" + baseURL + "/REVIEW.md#bot-catalog-prs) has the full list."},
		},
		{
			name:        "names the branch a run from another branch rebuilds from",
			report:      validReport(),
			diff:        diff,
			base:        Base{Branch: "feature-x"},
			contains:    []string{"Each run rebuilds this branch from feature-x,", "merge a fix on feature-x or record a decision"},
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
			notContains: []string{"### Open questions"},
		},
		{
			name: "a held two-sided question shows both sides",
			report: Report{NeedsHuman: []NeedsHuman{{
				Offering: "claude-sonnet-5-5", Field: "limit", Catalogue: "65535", Page: "65536", Values: []string{"providers/anthropic/models.go"},
				Rule: "no-rule", SourceURLs: []string{"https://platform.claude.com/docs/en/about-claude/models/overview.md"},
			}}},
			base: Base{URL: baseURL, Decisions: []reconcile.Decision{keepLimit}, Today: "2026-10-14"},
			contains: []string{
				"| [DEC-001](" + decisionsURL + "#L3) | `claude-sonnet-5-5` | limit | `65535`, `65536`, `providers/anthropic/models.go` | Agreed in review. |",
			},
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
				"### Open questions (2)\n\n**1. `claude-sonnet-5-5` · token limit** · ⚠ no rule covers this ([no-rule](" + rulesURL + "#L120))\n" +
					"- **Question:** No rule covers this. Decide.\n- **Also found:** `65535`, `65536`\n",
				"- **Sources:** [platform.claude.com](https://platform.claude.com/docs/en/about-claude/models/overview.md)",
				"**2. `claude-opus-4-1` · replacement** · rule [out-of-reach](" + rulesURL + "#L80)\n- **Question:** Which replacement?\n- **Sources:**",
			},
			notContains: []string{"| Offering | Field | Values |"},
		},
		{
			name: "skipped rows are grouped by rule",
			report: Report{Skipped: []Skipped{
				{Offering: "claude-mythos-5", Reason: "limited_availability", Rule: "limited-availability"},
				{Offering: "claude-haiku-4-5", Reason: "vendor_excluded", Rule: "no-rule"},
				{Offering: "claude-mythos-5-1", Reason: "limited_availability", Rule: "limited-availability"},
			}},
			base: Base{URL: baseURL, Rules: testRules},
			contains: []string{
				"### Skipped (3)\n\n| Rule | Why | Offerings |",
				"| [limited-availability](" + rulesURL + "#L40) | the model needs a signup (`limited_availability`) | `claude-mythos-5`, `claude-mythos-5-1` |",
				"| ⚠ [no-rule](" + rulesURL + "#L120) | the vendor isn't tracked (`vendor_excluded`) | `claude-haiku-4-5` |",
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
			name:   "questions for people come after the top lines, and Open questions before the first section",
			report: withWaiting,
			base:   Base{Rules: testRules, Branch: "main"},
			contains: []string{
				"has the full list.\n\n> [!IMPORTANT]\n> **1 open question,** listed under Open questions below. It doesn't block merging: merging keeps the current value, and a later run may ask again.\n>\n> 1. `claude-sonnet-5-5` · token limit\n\n**To answer a question,**",
				"add a rule to `RULES.md` instead.\n\n> [!NOTE]\n> **2 product decisions waiting** (they don't block this PR): " +
					"offer `in` on bedrock (India, asked since 2026-10-15)? offer `ca` on bedrock (Canada, new)?\n\nEach run rebuilds this branch",
				"### Open questions (1)\n\n**1. `claude-sonnet-5-5` · token limit**",
				"- **Sources:** [platform.claude.com](https://platform.claude.com/docs/en/about-claude/models/overview.md)\n\n## anthropic\n\n### Changes (1)",
			},
		},
		{
			name:        "held rows aren't questions",
			report:      twoAsks,
			base:        Base{Decisions: []reconcile.Decision{keepLimit}, Today: "2026-10-14"},
			contains:    []string{"> [!IMPORTANT]\n> **1 open question,** listed under Open questions below. It doesn't block merging: merging keeps the current value, and a later run may ask again.\n>\n> 1. `claude-opus-4-1` · replacement\n\n"},
			notContains: []string{"`claude-sonnet-5-5` · token limit", "[!NOTE]"},
		},
		{
			name:   "every question is listed, and a region with no name shows none",
			report: manyAsks,
			base:   Base{Rules: testRules},
			contains: []string{
				"> [!IMPORTANT]\n> **7 open questions,** listed under Open questions below. They don't block merging: merging keeps the current values, and a later run may ask again.\n>\n" +
					"> 1. `claude-opus-5` · token limit\n> 2. `claude-opus-5-5` · token limit\n> 3. `claude-sonnet-5` · token limit\n> 4. `claude-sonnet-5-5` · token limit\n" +
					"> 5. `claude-haiku-5` · token limit\n> 6. `claude-haiku-5-5` · token limit\n> 7. `claude-fable-5` · token limit\n\n",
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
			contains:    []string{"### Open questions (1)"},
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
			contains:  []string{"read `providers/anthropic/alias_test.go` as code; marking ready runs it with secrets."},
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

			body := Render(oneVendor(tt.report, tt.base), tt.diff, tt.codeFiles, tt.base)
			for _, want := range tt.contains {
				assert.Contains(t, body, want)
			}

			for _, unwanted := range tt.notContains {
				assert.NotContains(t, body, unwanted)
			}
		})
	}
}

// TestRenderSummaryLine checks the line under the title (proposal 2): what
// the PR changes, counted by kind across vendors, and what it leaves as it
// is.
func TestRenderSummaryLine(t *testing.T) {
	t.Parallel()

	price := func(offering string) Change {
		return Change{
			Offering: offering, Field: "price", Old: "input 2.00", New: "input 2.50", Rule: "price-follow",
			SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing.md",
		}
	}
	limit := Change{
		Offering: "claude-opus-5-5", Field: "limit", Old: "64000", New: "128000", Rule: "no-rule",
		SourceURL: "https://platform.claude.com/docs/en/about-claude/models/overview.md",
	}

	unchanged := validReport()
	unchanged.NeedsHuman = append(unchanged.NeedsHuman,
		NeedsHuman{Offering: "claude-opus-4-1", Field: "replacement", Rule: "out-of-reach", SourceURLs: []string{"https://platform.claude.com/docs/en/about-claude/model-deprecations.md"}},
		NeedsHuman{Offering: "claude-haiku-5-5", Field: "price", Page: "input 1.10", Rule: "no-rule", SourceURLs: []string{"https://platform.claude.com/docs/en/about-claude/pricing.md"}},
	)
	unchanged.Skipped = append(unchanged.Skipped,
		Skipped{Offering: "claude-haiku-4", Reason: "limited_availability", Rule: "limited-availability"},
		Skipped{Offering: "bedrock:in.anthropic.claude-sonnet-5-5", Reason: "region_not_offered", Rule: "region-not-offered"},
		Skipped{Offering: "bedrock:in.anthropic.claude-opus-5-5", Reason: "region_not_offered", Rule: "region-not-offered"},
	)

	tests := []struct {
		name    string
		vendors []Vendor
		base    Base
		want    string
	}{
		{
			name: "counts each kind of change across vendors",
			vendors: []Vendor{
				{Name: "anthropic", Report: Report{Changes: []Change{price("claude-opus-5-5"), price("claude-sonnet-5-5"), limit}}},
				{Name: "openai", Report: Report{Changes: []Change{price("gpt-6-1")}}},
			},
			want: "**4 changes** to the catalogue: 3 prices, 1 token limit.",
		},
		{
			name:    "one of a kind in the singular",
			vendors: oneVendor(validReport(), Base{}),
			want:    "**1 change** to the catalogue: 1 price. Not changed: 1 open question, 1 skipped offering.",
		},
		{
			name:    "no changes at all",
			vendors: oneVendor(Report{NeedsHuman: validReport().NeedsHuman}, Base{}),
			want:    "**No changes** to the catalogue. Not changed: 1 open question.",
		},
		{
			// The patch also retires both Bedrock Sonnet 4.5 copies and adds
			// an AU copy of Sonnet 5.5, and the report names none of them.
			name: "counts offerings changed with no row",
			vendors: func() []Vendor {
				before, after := effectSnapshots(t)
				cacheRead := Change{Offering: "claude-sonnet-5-5", Field: "price", Old: "cache read 0.20", New: "cache read 0.10", Rule: "price-follow"}

				return []Vendor{{Name: "anthropic", Report: Report{Changes: []Change{cacheRead}}, Before: before, After: after}}
			}(),
			want: "**4 changes** to the catalogue: 1 price, 3 offerings changed with no row.",
		},
		{
			name:    "counts what stays as it is",
			vendors: oneVendor(unchanged, Base{}),
			base:    Base{Decisions: []reconcile.Decision{keepLimit}, Today: "2026-10-14"},
			want:    "**1 change** to the catalogue: 1 price. Not changed: 2 open questions, 1 held by a decision, 2 copies waiting on a product decision, 2 skipped offerings.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// The line is the paragraph right after the title.
			_, rest, _ := strings.Cut(Render(tt.vendors, "", nil, tt.base), "\n\n")
			line, _, _ := strings.Cut(rest, "\n\n")
			assert.Equal(t, tt.want, line)
		})
	}
}

// TestRenderChecksLine checks the line under the summary (proposal 3): what a
// reviewer must do before marking the PR ready, naming up to three code files,
// and exactly what the bot checked.
func TestRenderChecksLine(t *testing.T) {
	t.Parallel()

	// Each claim is a step of pr-job.sh: render runs catalog-report, which
	// checks the report's format, each row's rule and the sources' hosts;
	// allowed() limits the files; the sandbox block runs the tests and lint.
	const checked = "Only the agent compared the values with their sources. For its own commits, the bot checked the report's format and the rule on each row, " +
		"that sources are on named hosts and that only allowed files change, and ran unit tests and lint without secrets. `REVIEW.md` has the full list."

	tests := []struct {
		name      string
		codeFiles []string
		want      string
		section   bool
	}{
		{
			name: "no code files",
			want: "**Before you mark it ready:** check each value in Changes against its source. " + checked,
		},
		{
			name:      "names one code file",
			codeFiles: []string{"providers/anthropic/alias_test.go"},
			want:      "**Before you mark it ready:** check each value in Changes against its source, and read `providers/anthropic/alias_test.go` as code; marking ready runs it with secrets. " + checked,
		},
		{
			name:      "names three code files",
			codeFiles: []string{"providers/anthropic/models.go", "providers/bedrock/models.go", "providers/openai/models.go"},
			want: "**Before you mark it ready:** check each value in Changes against its source, and read `providers/anthropic/models.go`, `providers/bedrock/models.go` and `providers/openai/models.go` as code; " +
				"marking ready runs them with secrets. " + checked,
		},
		{
			name:      "points to Code changed above three",
			codeFiles: []string{"go.mod", "providers/anthropic/models.go", "providers/bedrock/models.go", "providers/openai/models.go"},
			want:      "**Before you mark it ready:** check each value in Changes against its source, and read the 4 files under Code changed as code; marking ready runs them with secrets. " + checked,
			section:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := Render(oneVendor(validReport(), Base{}), "", tt.codeFiles, Base{})

			// The title, the summary line, then this line.
			paragraphs := strings.SplitN(body, "\n\n", 4)
			require.Len(t, paragraphs, 4)
			assert.Equal(t, tt.want, paragraphs[2])
			assert.Equal(t, tt.section, strings.Contains(body, "### Code changed"), "Code changed is a section only above three files")
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
		assert.Equal(t, Render(oneVendor(report, Base{}), diff, nil, Base{}), RenderCapped(oneVendor(report, Base{}), diff, nil, Base{}, 1<<20))
	})

	t.Run("tables alone too long: body cut on a line, within the limit", func(t *testing.T) {
		t.Parallel()

		const limit = 6000

		long := Report{}
		for i := range 100 {
			long.Changes = append(long.Changes, Change{
				Offering: fmt.Sprintf("bedrock:us.anthropic.claude-model-%03d", i), Field: "price", Old: "input 3.00", New: "input 3.30",
				SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing.md", Rule: "price-follow",
			})
		}

		body := RenderCapped(oneVendor(long, Base{}), diff, nil, Base{}, limit)
		assert.LessOrEqual(t, len(body), limit)

		top, _, _ := strings.Cut(Render(oneVendor(long, Base{}), diff, nil, Base{}), "\n## anthropic\n")
		assert.True(t, strings.HasPrefix(body, top), "the top of the body stays whole")
		assert.Contains(t, body, "cut to fit GitHub's size limit")
		assert.NotContains(t, body, "claude-model-099", "the rows past the limit are cut")
		assert.NotContains(t, body, "## Snapshot diff", "the diff's heading goes with the diff")
	})

	t.Run("too long: diff cut on a line, body within the limit", func(t *testing.T) {
		t.Parallel()

		const limit = 6000

		body := RenderCapped(oneVendor(report, Base{}), diff, nil, Base{}, limit)
		assert.LessOrEqual(t, len(body), limit)
		assert.Contains(t, body, "### Changes (1)")
		assert.Contains(t, body, "\n## Snapshot diff\n\n<details>", "a shortened diff keeps its heading")
		assert.Contains(t, body, "truncated; see the workflow artifact")
		assert.Contains(t, body, lines[0]+"\n")
		assert.NotContains(t, body, lines[len(lines)-1])

		kept := body[strings.Index(body, lines[0]):strings.Index(body, "\n… (truncated")]
		for line := range strings.SplitSeq(kept, "\n") {
			assert.Contains(t, lines, line, "the diff is cut only between lines")
		}
	})

	t.Run("many questions don't push the Changes table out", func(t *testing.T) {
		t.Parallel()

		const limit = 8000

		asks := validReport()
		for i := range 150 {
			// Each page differs, so the rows stay 150 questions.
			asks.NeedsHuman = append(asks.NeedsHuman, NeedsHuman{
				Offering: fmt.Sprintf("claude-model-%03d", i), Field: "limit", Page: strconv.Itoa(i), Rule: "no-rule",
				SourceURLs: []string{"https://platform.claude.com/a"},
			})
		}

		body := RenderCapped(oneVendor(asks, Base{}), "", nil, Base{}, limit)
		assert.LessOrEqual(t, len(body), limit)
		assert.Contains(t, body, "> 10. `claude-model-008` · token limit\n> …and 141 more, under Open questions.\n\n", "the box names ten")
		assert.Contains(t, body, "### Open questions (151)")
		assert.Contains(t, section(t, body, "anthropic"), "| `claude-opus-5-5` | price | `4.00` | `5.00` |", "the vendor's section keeps its share")
	})

	t.Run("notes stay when the diff is cut", func(t *testing.T) {
		t.Parallel()

		before, after := effectSnapshots(t)
		priced := Report{Changes: []Change{{
			Offering: "claude-sonnet-5-5", Field: "price", Old: "cache read 0.20", New: "cache read 0.10",
			SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing",
		}}}

		body := RenderCapped(oneVendor(priced, Base{Before: before, After: after}), diff, nil, Base{}, 6000)
		assert.Contains(t, body, "truncated; see the workflow artifact")
		assert.Contains(t, body, "| cache read 0.2 → 0.1 (-50%) |")
	})
}

// openaiSnapshots are an OpenAI catalogue before and after a patch that
// raises GPT-6.1's input price from 2.00 to 2.50.
func openaiSnapshots(t *testing.T) (*reconcile.Snapshot, *reconcile.Snapshot) {
	t.Helper()

	snapshot := func(input int) *reconcile.Snapshot {
		snap, err := reconcile.ParseSnapshot(fmt.Appendf(nil, `{"providers": [{"provider": "openai", "offerings": [{"id": "gpt-6-1", "model": "openai/gpt-6-1",
			"pricing": {"default": {"base": {"input_microcents_per_million": %d}}}}]}]}`, input))
		require.NoError(t, err)

		return &snap
	}

	return snapshot(200000000), snapshot(250000000)
}

// section is one vendor's part of a combined body: from its heading to the
// next vendor's, or to the snapshot diff.
func section(t *testing.T, body, vendor string) string {
	t.Helper()

	_, rest, found := strings.Cut(body, "\n## "+vendor+"\n")
	require.True(t, found, "the body has no section for %s", vendor)

	rest, _, _ = strings.Cut(rest, "\n## ")
	rest, _, _ = strings.Cut(rest, "<details>")

	return rest
}

// TestRenderVendors checks the combined PR body (AC4: one PR for every
// vendor): each vendor's rows sit in its own section, and its effect notes
// and "Changed with no row" come from its own catalogue.
func TestRenderVendors(t *testing.T) {
	t.Parallel()

	before, after := effectSnapshots(t)
	openaiBefore, openaiAfter := openaiSnapshots(t)

	cacheRead := Report{Changes: []Change{{
		Offering: "claude-sonnet-5-5", Field: "price", Old: "cache read 0.20", New: "cache read 0.10",
		SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing", Rule: "price-follow",
	}}}
	input := Report{Changes: []Change{{
		Offering: "gpt-6-1", Field: "price", Old: "input 2.00", New: "input 2.50",
		SourceURL: "https://developers.openai.com/api/docs/pricing", Rule: "price-follow",
	}}}

	body := Render([]Vendor{
		{Name: "anthropic", Report: cacheRead, Before: before, After: after},
		{Name: "openai", Report: input, Before: openaiBefore, After: openaiAfter},
	}, "", nil, Base{})

	assert.True(t, strings.HasPrefix(body, "## Catalogue drift: anthropic, openai\n"), "the title names every vendor")

	anthropic, openai := section(t, body, "anthropic"), section(t, body, "openai")
	assert.Contains(t, anthropic, "| cache read 0.2 → 0.1 (-50%) |")
	assert.Contains(t, anthropic, "### Changed with no row", "Anthropic's catalogue also adds an AU copy no row names")
	assert.NotContains(t, anthropic, "gpt-6-1")
	assert.Contains(t, openai, "| `gpt-6-1` | price | `input 2.00` | `input 2.50` |")
	assert.Contains(t, openai, "input 2 → 2.5 (+25%)")
	assert.NotContains(t, openai, "Changed with no row", "OpenAI's catalogue changes only what its row names")
}

// TestRenderVendorsShowsThePRPartsOnce checks that what covers the whole PR
// (code changed, routing changes, the snapshot diff) shows once, outside the
// vendor sections, and that a vendor with nothing to change, or whose run
// failed, still has a section that says so.
func TestRenderVendorsShowsThePRPartsOnce(t *testing.T) {
	t.Parallel()

	routesBefore, err := ParseRoutes([]byte("ap-southeast-2 anthropic.claude-sonnet-5-5 global.anthropic.claude-sonnet-5-5\n"))
	require.NoError(t, err)
	routesAfter, err := ParseRoutes([]byte("ap-southeast-2 anthropic.claude-sonnet-5-5 au.anthropic.claude-sonnet-5-5\n"))
	require.NoError(t, err)

	base := Base{RoutesBefore: routesBefore, RoutesAfter: routesAfter}
	body := Render([]Vendor{
		{Name: "anthropic", Report: validReport()},
		{Name: "openai", Report: Report{}},
		{Name: "google", Failed: true},
	}, "+  \"input\": 5", []string{"providers/bedrock/models.go", "providers/openai/models.go"}, base)

	for _, once := range []string{"read `providers/bedrock/models.go` and `providers/openai/models.go` as code", "### Routing changes (1)", "<details><summary>Snapshot diff</summary>"} {
		assert.Equal(t, 1, strings.Count(body, once), "%q shows once", once)
	}

	assert.Less(t, strings.Index(body, "### Routing changes"), strings.Index(body, "## anthropic"), "the PR's parts come before the vendors")
	assert.Greater(t, strings.Index(body, "<details>"), strings.Index(body, "## google"), "the snapshot diff comes last")
	assert.Equal(t, "The run found nothing to change.", strings.TrimSpace(section(t, body, "openai")))
	assert.Contains(t, section(t, body, "google"), "This vendor's run failed, so this PR has nothing from it.")
}

// TestRenderKeepsEveryVendorPart checks that moving what concerns the whole
// PR to the top leaves each vendor's own parts in its section: the replay
// note, Changes, Changed with no row, Held by decisions and Skipped; the
// found-nothing and failed notes; and the recorded-decisions line.
func TestRenderKeepsEveryVendorPart(t *testing.T) {
	t.Parallel()

	before, after := effectSnapshots(t)

	body := Render([]Vendor{
		{Name: "anthropic", Report: validReport(), Before: before, After: after, Replayed: Replay{Run: "123", Date: "2026-10-28"}},
		{Name: "openai", Report: Report{}},
		{Name: "google", Failed: true},
	}, "+  \"input\": 5", nil, Base{Decisions: []reconcile.Decision{keepLimit}, Today: "2026-10-14"})

	anthropic := section(t, body, "anthropic")
	for _, part := range []string{"> Replayed with no model", "### Changes (1)", "### Changed with no row (", "### Held by decisions (1)", "### Skipped (1)"} {
		assert.Contains(t, anthropic, part)
	}

	assert.Equal(t, "The run found nothing to change.", strings.TrimSpace(section(t, body, "openai")))
	assert.Contains(t, section(t, body, "google"), "This vendor's run failed, so this PR has nothing from it.")
	assert.Contains(t, body, "Recorded decisions: 1, in `decisions.yaml`.")
}

// TestRenderCappedVendors checks the byte budget per vendor: a vendor too big
// for its share is cut inside its own section, on a line, and the others,
// the top of the body and the size limit all hold.
func TestRenderCappedVendors(t *testing.T) {
	t.Parallel()

	const limit = 8000

	big := Report{}
	for i := range 200 {
		big.Changes = append(big.Changes, Change{
			Offering: fmt.Sprintf("bedrock:us.anthropic.claude-model-%03d", i), Field: "price", Old: "input 3.00", New: "input 3.30",
			SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing.md", Rule: "price-follow",
		})
	}

	small := Report{Changes: []Change{{
		Offering: "gpt-6-1", Field: "price", Old: "input 2.00", New: "input 2.50",
		SourceURL: "https://developers.openai.com/api/docs/pricing", Rule: "price-follow",
	}}}
	vendors := []Vendor{{Name: "anthropic", Report: big}, {Name: "openai", Report: small}, {Name: "google", Report: Report{}}}

	body := RenderCapped(vendors, "+  \"input\": 5", nil, Base{}, limit)
	assert.LessOrEqual(t, len(body), limit)

	top, _, _ := strings.Cut(Render(vendors, "+  \"input\": 5", nil, Base{}), "\n## anthropic\n")
	assert.True(t, strings.HasPrefix(body, top), "the top of the body stays whole")

	anthropic := section(t, body, "anthropic")
	assert.Contains(t, anthropic, "claude-model-000")
	assert.NotContains(t, anthropic, "claude-model-199", "the rows past the vendor's share are cut")
	assert.Contains(t, anthropic, "cut to fit GitHub's size limit")
	assert.True(t, strings.HasSuffix(strings.TrimSpace(strings.TrimSuffix(anthropic, strings.TrimSpace(cutNote)+"\n")), "|"), "the cut ends on a whole table row")

	assert.Equal(t, strings.TrimSpace(section(t, Render(vendors, "", nil, Base{}), "openai")), strings.TrimSpace(section(t, body, "openai")), "a vendor within its share stays whole")
	assert.Contains(t, section(t, body, "google"), "The run found nothing to change.")
}

// TestRenderVendorsLabelsQuestions checks that the box at the top and Open
// questions name each question's vendor once the body has more than one, and
// number them across vendors.
func TestRenderVendorsLabelsQuestions(t *testing.T) {
	t.Parallel()

	gpt := Report{NeedsHuman: []NeedsHuman{{
		Offering: "gpt-6-1", Field: "price", Page: "priority 4.00", Rule: "no-rule",
		SourceURLs: []string{"https://developers.openai.com/api/docs/pricing"},
	}}}
	body := Render([]Vendor{{Name: "anthropic", Report: validReport()}, {Name: "openai", Report: gpt}}, "", nil, Base{})

	assert.Contains(t, body, "> 1. anthropic · `claude-sonnet-5-5` · token limit\n> 2. openai · `gpt-6-1` · price\n\n")
	assert.Contains(t, body, "**2. openai · `gpt-6-1` · price** ·", "the numbers run across vendors")
	assert.Equal(t, "The run changed nothing; its questions are under Open questions above.", strings.TrimSpace(section(t, body, "openai")),
		"a vendor with only questions says where they went")
}

// TestRenderQuestionsBox checks the questions at the top (proposals 1 and 5):
// one box numbers them across vendors, names each so no two look alike, and
// says merging is allowed; the To answer paragraph says how to settle one;
// and Open questions repeats the numbers and names above the vendor sections.
func TestRenderQuestionsBox(t *testing.T) {
	t.Parallel()

	// Two copies in different regions, with different rates on the page, so
	// they are two questions.
	copyAsk := func(prefix, page string) NeedsHuman {
		return NeedsHuman{
			Offering: "bedrock:" + prefix + ".openai.gpt-6-1", Field: reconcile.FieldNewCopy, Page: page, Rule: "no-rule",
			SourceURLs: []string{"https://developers.openai.com/api/docs/pricing"},
		}
	}
	openai := Report{NeedsHuman: []NeedsHuman{copyAsk("us", "input 2.20"), copyAsk("global", "input 2.00")}}

	body := Render([]Vendor{{Name: "anthropic", Report: validReport()}, {Name: "openai", Report: openai}}, "", nil, Base{Branch: "main"})

	assert.Contains(t, body, "> [!IMPORTANT]\n> **3 open questions,** listed under Open questions below. "+
		"They don't block merging: merging keeps the current values, and a later run may ask again.\n>\n"+
		"> 1. anthropic · `claude-sonnet-5-5` · token limit\n> 2. openai · us copy: `bedrock:us.openai.gpt-6-1` · new copy\n"+
		"> 3. openai · global copy: `bedrock:global.openai.gpt-6-1` · new copy\n\n"+
		"**To answer a question,** open a PR into main that fixes the value, or that adds an entry for each offering to `.github/reconcile-models/decisions.yaml`: "+
		"`kind: keep` with a lock keeps a value, and `kind: hold` keeps a new model or copy out. If the case can come back, add a rule to `RULES.md` instead.\n\n")

	for _, block := range []string{
		"### Open questions (3)\n\n**1. anthropic · `claude-sonnet-5-5` · token limit** · ⚠ no rule covers this (`no-rule`)\n",
		"**2. openai · us copy: `bedrock:us.openai.gpt-6-1` · new copy** · ⚠ no rule covers this (`no-rule`)\n",
		"**3. openai · global copy: `bedrock:global.openai.gpt-6-1` · new copy** · ⚠ no rule covers this (`no-rule`)\n",
	} {
		assert.Contains(t, body, block)
	}

	assert.Less(t, strings.Index(body, "### Open questions"), strings.Index(body, "## anthropic"), "the questions come before the vendors")
	assert.NotContains(t, section(t, body, "anthropic"), "no rule covers this")
	assert.NotContains(t, section(t, body, "openai"), "no rule covers this")
}

// TestQuestionFile checks the file a Decide line names: the offering's
// provider package, or the facts file for a model's facts. Each file exists.
func TestQuestionFile(t *testing.T) {
	t.Parallel()

	tests := []struct{ field, offering, vendor, want string }{
		{"price", "bedrock:us.anthropic.claude-sonnet-5-5", "anthropic", "providers/bedrock/models.go"},
		{"limit", "claude-sonnet-5-5", "anthropic", "providers/anthropic/models.go"},
		{"capability", "gpt-6-1", "openai", "providers/openai/models.go"},
		{"release_date", "claude-sonnet-5-5", "anthropic", "catalog/facts_data.go"},
		{"knowledge_cutoff", "gpt-6-1", "openai", "catalog/facts_data.go"},
	}
	for _, tt := range tests {
		t.Run(tt.field+" on "+tt.offering, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tt.want, questionFile(tt.field, tt.offering, tt.vendor))
			assert.FileExists(t, filepath.Join("..", "..", tt.want))
		})
	}
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
		assert.Equal(t, reconcile.OfferingPattern.String(), items(array).Properties["offering"].Pattern, "%s offering pattern", array)
	}

	// Values match the validator's pattern wherever a row carries them.
	for array, properties := range map[string][]string{"changes": {"old", "new"}, "needs_human": {"catalogue", "page"}} {
		for _, property := range properties {
			assert.Equal(t, valuePattern.String(), items(array).Properties[property].Pattern, "%s %s pattern", array, property)
		}
	}

	assert.Equal(t, valuePattern.String(), items("needs_human").Properties["values"].Items.Pattern, "needs_human values pattern")
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
		{"a versioned Vertex offering", strings.Replace(question("limit", `, "page": "claude-opus-4-5@20251101"`),
			`"claude-sonnet-5-5"`, `"vertex:claude-opus-4-5@20251101"`, 1), false},
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

// TestRenderQuestionBlock checks one Open questions row: a rule's fixed
// question first, then the two sides and other values, then a no-rule row's
// Decide line naming the file to edit, then the sources.
func TestRenderQuestionBlock(t *testing.T) {
	t.Parallel()

	const (
		noRule  = "**1. `claude-sonnet-5-5` · token limit** · ⚠ no rule covers this (`no-rule`)\n"
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
				"- **Decide:** keep `65535`, or edit `providers/anthropic/models.go` to match the page (`65536`).\n" + sources,
		},
		{
			name: "both sides and nothing else found",
			item: ask("no-rule", "65535", "65536"),
			want: noRule + "- **Catalogue:** `65535`\n- **Page:** `65536`\n- **Decide:** keep `65535`, or edit `providers/anthropic/models.go` to match the page (`65536`).\n" + sources,
		},
		{
			name: "a new copy with both sides: leave it out or add it",
			item: NeedsHuman{
				Offering: "bedrock:us.anthropic.claude-sonnet-5-5", Field: reconcile.FieldNewCopy, Catalogue: "none", Page: "input 3.30",
				Rule: "no-rule", SourceURLs: []string{"https://platform.claude.com/a"},
			},
			want: "**1. us copy: `bedrock:us.anthropic.claude-sonnet-5-5` · new copy** · ⚠ no rule covers this (`no-rule`)\n" +
				"- **Catalogue:** `none`\n- **Page:** `input 3.30`\n- **Decide:** leave it out, or add it in `providers/bedrock/models.go` with the page's rates.\n" + sources,
		},
		{
			name: "a rule's row keeps its question",
			item: ask("out-of-reach", "none", "claude-opus-5"),
			want: "**1. `claude-sonnet-5-5` · token limit** · rule `out-of-reach`\n- **Question:** Which replacement?\n" +
				"- **Catalogue:** `none`\n- **Page:** `claude-opus-5`\n" + sources,
		},
		{
			name: "one side alone keeps the question",
			item: ask("no-rule", "", "65536"),
			want: noRule + "- **Question:** No rule covers this. Decide.\n- **Page:** `65536`\n" + sources,
		},
		{
			name: "no sides: the question, then also found",
			item: ask("no-rule", "", "", "65535", "65536"),
			want: noRule + "- **Question:** No rule covers this. Decide.\n- **Also found:** `65535`, `65536`\n" + sources,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := Render(oneVendor(Report{NeedsHuman: []NeedsHuman{tt.item}}, Base{}), "", nil, Base{Rules: testRules})
			assert.Contains(t, body, tt.want)
		})
	}
}

// TestRenderGroupsSameQuestion uses #287's structured-output rows, which asked
// one question 12 times: rows that differ only by offering and sources are
// one question, and a row with any other difference stays its own.
func TestRenderGroupsSameQuestion(t *testing.T) {
	t.Parallel()

	ask := func(offering, page, source string) NeedsHuman {
		return NeedsHuman{
			Offering: "bedrock:" + offering, Field: "capability", Rule: "no-rule",
			Catalogue: "structured_output false", Page: page,
			Values: []string{"providers/bedrock/capabilities_test.go", "claudeNoSchemaCaps"}, SourceURLs: []string{source},
		}
	}
	report := Report{NeedsHuman: []NeedsHuman{
		ask("global.anthropic.claude-fable-5-1", "Structured outputs supported", "https://docs.aws.amazon.com/a"),
		ask("us.anthropic.claude-fable-5-1", "Structured outputs supported", "https://docs.aws.amazon.com/a"),
		ask("global.anthropic.claude-haiku-5-5", "Structured outputs not listed", "https://docs.aws.amazon.com/c"),
		ask("global.anthropic.claude-opus-5-5", "Structured outputs supported", "https://docs.aws.amazon.com/b"),
	}}

	body := Render(oneVendor(report, Base{}), "", nil, Base{Rules: testRules})

	for _, want := range []string{
		"> [!IMPORTANT]\n> **2 open questions,** listed under Open questions below. They don't block merging: merging keeps the current values, and a later run may ask again.\n>\n" +
			"> 1. `bedrock:global.anthropic.claude-fable-5-1` and 2 more · capability\n> 2. `bedrock:global.anthropic.claude-haiku-5-5` · capability\n\n",
		"### Open questions (2)\n\n" +
			"**1. `bedrock:global.anthropic.claude-fable-5-1` and 2 more · capability** · ⚠ no rule covers this (`no-rule`)\n" +
			"- **Offerings:** `bedrock:global.anthropic.claude-fable-5-1`, `bedrock:us.anthropic.claude-fable-5-1`, `bedrock:global.anthropic.claude-opus-5-5`\n" +
			"- **Catalogue:** `structured_output false`\n- **Page:** `Structured outputs supported`\n" +
			"- **Also found:** `providers/bedrock/capabilities_test.go`, `claudeNoSchemaCaps`\n" +
			"- **Decide:** keep `structured_output false`, or edit `providers/bedrock/models.go` to match the page (`Structured outputs supported`).\n" +
			"- **Sources:** [docs.aws.amazon.com](https://docs.aws.amazon.com/a), [docs.aws.amazon.com](https://docs.aws.amazon.com/b)\n\n" +
			"**2. `bedrock:global.anthropic.claude-haiku-5-5` · capability** · ⚠ no rule covers this (`no-rule`)\n",
	} {
		assert.Contains(t, body, want)
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

	body := Render(oneVendor(report, Base{Before: before, After: after}), "", nil, Base{Today: "2026-10-15"})

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
	twoSided := func(geo, catalogue string) NeedsHuman {
		return NeedsHuman{
			Offering: "bedrock:" + geo + ".anthropic.claude-sonnet-4-5-20250929-v1:0", Field: "limit", Rule: "no-rule",
			Catalogue: catalogue, Page: "200", Values: []string{"providers/bedrock/models.go"}, SourceURLs: []string{"https://platform.claude.com/x"},
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
				"### Suggested rules (1)\n\nSome questions under Open questions went to a person under no-rule and share a field. If the case can come back, add a Default to RULES.md: give it a slug, write When as the condition, and replace each ?.\n\n" +
					"| Default | Field | When | Bot does | Why | Outcome |\n|---|---|---|---|---|---|\n" +
					"| new-default | `lifecycle` | 3 rows this run, all with `Legacy 2026-10-08`, `EOL 2027-04-08`, `AWS names no replacement` | ? | ? | ? |\n",
			},
		},
		{
			name: "two-sided rows suggest the side they share",
			asks: []NeedsHuman{twoSided("global", "100"), twoSided("us", "150")},
			contains: []string{
				"| new-default | `limit` | 2 rows this run, all with `200`, `providers/bedrock/models.go` | ? | ? | ? |\n",
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

			body := Render(oneVendor(Report{NeedsHuman: tt.asks}, Base{}), "", nil, Base{Rules: testRules, Decisions: tt.decisions, Today: "2026-10-15"})
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

// TestRouteInputChange checks that a moved route's price compares what its
// callers paid before the patch with what they pay after it, when the same
// patch also reprices the copy they leave.
func TestRouteInputChange(t *testing.T) {
	t.Parallel()

	copyAt := func(id string, input int) string {
		return fmt.Sprintf(`{"id": "%s", "model": "anthropic/claude-sonnet-5-5", "pricing": {"default": {"base": {"input_microcents_per_million": %d}}}}`, id, input)
	}
	before, err := reconcile.ParseSnapshot([]byte(`{"providers": [{"provider": "aws.bedrock", "offerings": [` +
		copyAt("global.anthropic.claude-sonnet-5-5", 300000000) + `]}]}`))
	require.NoError(t, err)

	after, err := reconcile.ParseSnapshot([]byte(`{"providers": [{"provider": "aws.bedrock", "offerings": [` +
		copyAt("global.anthropic.claude-sonnet-5-5", 600000000) + `, ` + copyAt("us.anthropic.claude-sonnet-5-5", 660000000) + `]}]}`))
	require.NoError(t, err)

	base := Base{Before: &before, After: &after}
	assert.Equal(t, "+120%", base.routeInputChange("global.anthropic.claude-sonnet-5-5", "us.anthropic.claude-sonnet-5-5"))
}
