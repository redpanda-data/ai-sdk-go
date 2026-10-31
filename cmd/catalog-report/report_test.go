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
		{"a why in plain words", func(r *Report) {
			r.NeedsHuman[0].Why = "AWS prints 65536 (the API's limit), but we list 65535; which is right?"
		}, ""},
		{"markup in the why", func(r *Report) { r.NeedsHuman[0].Why = "Safe to **approve**" }, `needs_human[0]: bad sentence "Safe to **approve**"`},
		{"a bare link in the why", func(r *Report) { r.NeedsHuman[0].Why = "See www.evil.example for the price" }, "needs_human[0]: bad sentence"},
		{"an issue link in the why", func(r *Report) { r.NeedsHuman[0].Why = "As agreed in GH-12" }, "needs_human[0]: bad sentence"},
		{"a mention in the why", func(r *Report) { r.NeedsHuman[0].Why = "Ask @admins" }, "needs_human[0]: bad sentence"},
		{"a long why", func(r *Report) { r.NeedsHuman[0].Why = strings.Repeat("a", 301) }, "needs_human[0]: bad sentence"},
		{"a rule draft", func(r *Report) {
			r.NeedsHuman[0].RuleWhen, r.NeedsHuman[0].RuleDoes = "a model page lists a limit one token above ours", "keeps ours"
		}, ""},
		{"half a rule draft", func(r *Report) { r.NeedsHuman[0].RuleWhen = "a model page lists a limit one token above ours" }, "needs_human[0]: a rule draft needs both rule_when and rule_does"},
		{"markup in a rule draft", func(r *Report) {
			r.NeedsHuman[0].RuleWhen, r.NeedsHuman[0].RuleDoes = "always", "[approves](x)"
		}, `needs_human[0]: bad sentence "[approves](x)"`},
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
					"| `claude-sonnet-5-5` | price | `0.20` | `0.10` | [c](https://platform.claude.com/c) |\n" +
					"| `vertex:claude-sonnet-5-5` | price | `0.22` | `0.11` | [a](https://platform.claude.com/a) |\n\n" +
					"**[new-copy-offered-region](" + rulesURL + "#L32)** (1)\n\n| Offering |",
			},
		},
		{
			name:   "a route that moves is flagged",
			report: changes(Change{Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_copy", New: "au.anthropic.claude-sonnet-5-5"}),
			base:   withRoutes,
			contains: []string{
				"### Routing changes (1)\n\n⚑ Bare-name calls from these regions now go to another copy. Check the price and where the data is processed.\n\n",
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
				"| `claude-opus-5-5` | price | `4.00` | `5.00` | [pricing](https://platform.claude.com/docs/en/about-claude/pricing.md) |",
				"`65535`, `65536`",
				"| `no-rule` | `vendor_excluded` | `claude-haiku-4-5` |",
			},
			notContains: []string{"Snapshot diff", "```diff", "⚠"},
		},
		{
			name:        "no drift",
			report:      Report{},
			contains:    []string{"No drift found"},
			notContains: []string{"```diff", "## Snapshot diff"},
		},
		{
			name:        "a diff with no rows isn't no drift, and stays in Files changed",
			report:      Report{},
			diff:        diff,
			notContains: []string{"No drift found", "```diff", "+  \"input\": 5"},
		},
		{
			name:   "says what a pushed commit does",
			report: validReport(),
			diff:   diff,
			base:   Base{Branch: "main"},
			contains: []string{
				"You can push commits here: the bot never removes them. It doesn't read comments.\n\n",
			},
			notContains: []string{"Don't push", "Comment on the PR instead", "Recorded decisions"},
		},
		{
			name:     "links the review guide at the base commit",
			report:   validReport(),
			diff:     diff,
			base:     Base{URL: baseURL, Branch: "main"},
			contains: []string{"[REVIEW.md](" + baseURL + "/REVIEW.md#bot-catalog-prs) says how to review a bot PR."},
		},
		{
			name:        "names the branch a run from another branch rebuilds from",
			report:      validReport(),
			diff:        diff,
			base:        Base{Branch: "feature-x"},
			contains:    []string{"open a PR into feature-x that does what its Decide line says."},
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
			notContains: []string{"Skipped ("},
		},
		{
			name:        "a skipped copy under a hold on new_copy is held",
			report:      Report{Skipped: []Skipped{{Offering: "vertex:claude-haiku-4-5", Reason: "out_of_scope"}}},
			base:        Base{URL: baseURL, Decisions: []reconcile.Decision{holdCopy}, Today: "2026-10-14"},
			contains:    []string{"| [DEC-002](" + decisionsURL + "#L12) | `vertex:claude-haiku-4-5` | new_copy | out_of_scope | Not offered to users. |"},
			notContains: []string{"Skipped ("},
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
				"### Open questions (2)\n\n**1. `claude-sonnet-5-5` · token limit** · no rule covers this ([no-rule](" + rulesURL + "#L120))\n" +
					"- **Agent's notes:** `65535`, `65536`\n- **Decide:** No rule covers this. Decide.\n",
				"- **Sources:** [models](https://platform.claude.com/docs/en/about-claude/models/overview.md)",
				"**2. `claude-opus-4-1` · replacement** · rule [out-of-reach](" + rulesURL + "#L80)\n- **Decide:** Which replacement?\n- **Sources:**",
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
				"<details><summary>Skipped (3): left out by a rule</summary>\n\n| Rule | Why | Offerings |",
				"| [limited-availability](" + rulesURL + "#L40) | the model needs a signup (`limited_availability`) | `claude-mythos-5`, `claude-mythos-5-1` |",
				"| [no-rule](" + rulesURL + "#L120) | the vendor isn't tracked (`vendor_excluded`) | `claude-haiku-4-5` |",
			},
		},
		{
			name:   "a question per region, with since when it has been asked",
			report: Report{Skipped: waiting},
			base:   Base{URL: baseURL, Rules: testRules},
			contains: []string{
				"> [!IMPORTANT]\n> **2 open questions,** listed under Open questions below. They don't block merging: merging changes nothing they ask about, and a later run asks again.\n>\n" +
					"> 1. bedrock · offer region `in` (India)? · 3 copies · asked since 2026-10-15\n> 2. bedrock · offer region `ca` (Canada)? · 1 copy · new\n\n",
				"### Open questions (2)\n\n**1. bedrock · offer region `in` (India)? · 3 copies · asked since 2026-10-15** · rule [region-not-offered](" + rulesURL + "#L20)\n" +
					"- **Copies waiting:** `bedrock:in.anthropic.claude-opus-5`, `bedrock:in.anthropic.claude-sonnet-5`, `bedrock:in.anthropic.claude-haiku-4-5-20251001-v1:0`\n" +
					"- **Why the bot asks:** The vendor's pages list these copies in India, a region the catalogue doesn't offer, so the bot left them out. A new region changes routing, price and where data is processed for callers there, so offering one is a product call.\n" +
					"- **Decide:** offer it, or decline it? To offer it, follow the [add-model skill](" + baseURL + "/.claude/skills/add-model/SKILL.md)'s steps for a new region: the provider's code, the copies and [RULES.md](" + baseURL + "/" + reconcile.RulesPath + ")'s Offered list, in one PR. To decline it, list it under Not offered in RULES.md as `` `in` (declined YYYY-MM-DD) ``.\n\n",
				"**2. bedrock · offer region `ca` (Canada)? · 1 copy · new** · rule [region-not-offered](" + rulesURL + "#L20)\n- **Copies waiting:** `bedrock:ca.anthropic.claude-opus-5`\n" +
					"- **Why the bot asks:** The vendor's pages list this copy in Canada, a region the catalogue doesn't offer, so the bot left it out.",
			},
			notContains: []string{"[!NOTE]", "Waiting on a product decision", "product decision"},
		},
		{
			name:   "a declined region's copies are skipped, not waiting",
			report: Report{Skipped: waiting},
			base:   Base{URL: baseURL, Rules: declinedCA},
			contains: []string{
				"### Open questions (1)\n\n**1. bedrock · offer region `in` (India)?",
				"<details><summary>Skipped (1): left out by a rule</summary>\n\n| Rule | Why | Offerings |\n|---|---|---|\n| [region-not-offered](" + rulesURL + "#L20) | `region_not_offered` | `bedrock:ca.anthropic.claude-opus-5` |\n",
			},
			notContains: []string{"offer region `ca`", "[!NOTE]"},
		},
		{
			name:   "questions for people come after the top lines, and Open questions before the first section",
			report: withWaiting,
			base:   Base{Rules: testRules, Branch: "main"},
			contains: []string{
				"because no automated check does.\n\n> [!IMPORTANT]\n> **3 open questions,** listed under Open questions below. They don't block merging: merging changes nothing they ask about, and a later run asks again.\n>\n" +
					"> 1. `claude-sonnet-5-5` · token limit\n> 2. bedrock · offer region `in` (India)? · 3 copies · asked since 2026-10-15\n> 3. bedrock · offer region `ca` (Canada)? · 1 copy · new\n\n**To answer a question,**",
				"add a rule to `RULES.md` instead.\n\nYou can push commits here",
				"### Open questions (3)\n\n**1. `claude-sonnet-5-5` · token limit**",
				"- **Sources:** [models](https://platform.claude.com/docs/en/about-claude/models/overview.md)\n\n**2. bedrock · offer region `in` (India)?",
				"- **Decide:** offer it, or decline it? To offer it, follow the `add-model skill`'s steps for a new region: the provider's code, the copies and `RULES.md`'s Offered list, in one PR. To decline it, list it under Not offered in RULES.md as `` `ca` (declined YYYY-MM-DD) ``.\n\n## anthropic\n\n### Changes (1)",
			},
		},
		{
			name:        "held rows aren't questions",
			report:      twoAsks,
			base:        Base{Decisions: []reconcile.Decision{keepLimit}, Today: "2026-10-14"},
			contains:    []string{"> [!IMPORTANT]\n> **1 open question,** listed under Open questions below. It doesn't block merging: merging changes nothing it asks about, and a later run asks again.\n>\n> 1. `claude-opus-4-1` · replacement\n\n"},
			notContains: []string{"`claude-sonnet-5-5` · token limit", "product decision"},
		},
		{
			name:   "every question is listed, and a region with no name shows none",
			report: manyAsks,
			base:   Base{Rules: testRules},
			contains: []string{
				"> [!IMPORTANT]\n> **9 open questions,** listed under Open questions below. They don't block merging: merging changes nothing they ask about, and a later run asks again.\n>\n" +
					"> 1. `claude-opus-5` · token limit\n> 2. `claude-opus-5-5` · token limit\n> 3. `claude-sonnet-5` · token limit\n> 4. `claude-sonnet-5-5` · token limit\n" +
					"> 5. `claude-haiku-5` · token limit\n> 6. `claude-haiku-5-5` · token limit\n> 7. `claude-fable-5` · token limit\n" +
					"> 8. bedrock · offer region `us-gov` (GovCloud (US))? · 1 copy · new\n> 9. bedrock · offer region `zz`? · 1 copy · new\n\n",
				"- **Why the bot asks:** The vendor's pages list this copy in region `zz`, a region the catalogue doesn't offer,",
			},
			notContains: []string{"[!NOTE]"},
		},
		{
			name:        "no questions, no box",
			report:      Report{Skipped: waiting[2:3]},
			base:        Base{Rules: declinedCA},
			contains:    []string{"<summary>Skipped (1)"},
			notContains: []string{"[!IMPORTANT]", "[!NOTE]", "offer region"},
		},
		{
			name:        "waiting rows aren't listed as skipped",
			report:      Report{Skipped: append([]Skipped{{Offering: "claude-mythos-5", Reason: "limited_availability", Rule: "limited-availability"}}, waiting...)},
			base:        Base{URL: baseURL, Rules: testRules},
			contains:    []string{"<summary>Skipped (1)"},
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
			contains:  []string{"Also read the file under Code changed as code: marking ready runs it with secrets.", "### Code changed (1)\n\n- `providers/anthropic/alias_test.go`\n\n"},
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

// TestRenderCarried checks the line under the checks line (proposal 12): each
// commit people pushed that the carry job carried, and that such commits get
// fewer checks than the bot's own (carry.sh).
func TestRenderCarried(t *testing.T) {
	t.Parallel()

	sha := func(first string) string { return first + strings.Repeat("0", 40-len(first)) }

	tests := []struct {
		name    string
		carried []Commit
		want    string
	}{
		{
			name:    "one commit",
			carried: []Commit{{SHA: sha("1faeeba"), Author: "Adam Wownysz"}},
			want:    "**Not from the bot:** 1 commit: `1faeeba` by `Adam Wownysz`.",
		},
		{
			name: "an author or subject that isn't safe to print is left out",
			carried: []Commit{
				{SHA: sha("aaaaaaa"), Author: "Łukasz"},
				{SHA: sha("bbbbbbb"), Author: "t"},
			},
			want: "**Not from the bot:** 2 commits: `aaaaaaa`; `bbbbbbb` by `t`.",
		},
		{
			// In a code span, a name can't link, mention or start markup.
			name: "an author name with markup stays in its code span",
			carried: []Commit{
				{SHA: sha("aaaaaaa"), Author: "[Approve](https://x)"},
				{SHA: sha("bbbbbbb"), Author: "www.example.com @org/team"},
				{SHA: sha("ccccccc"), Author: "Mary `O'Neil`"},
			},
			want: "**Not from the bot:** 3 commits: `aaaaaaa` by `[Approve](https://x)`; `bbbbbbb` by `www.example.com @org/team`; `ccccccc`.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// The title, the Context line, the summary line, the checks line,
			// then this line.
			paragraphs := strings.SplitN(Render(oneVendor(validReport(), Base{}), "", nil, Base{Carried: tt.carried}), "\n\n", 6)
			require.Len(t, paragraphs, 6)
			assert.Equal(t, tt.want, paragraphs[4])
		})
	}

	assert.NotContains(t, Render(oneVendor(validReport(), Base{}), "", nil, Base{}), "Not from the bot", "no line when nothing was carried")
}

// TestLinkNamesThePage checks a source link's text (proposal 8): the page's
// last path segment, without "model-card-" and its extension, or the host
// when there is none or it isn't safe to print.
func TestLinkNamesThePage(t *testing.T) {
	t.Parallel()

	tests := []struct{ url, want string }{
		{"https://platform.claude.com/docs/en/about-claude/pricing.md", "[pricing](https://platform.claude.com/docs/en/about-claude/pricing.md)"},
		{"https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5-1.html", "[anthropic-claude-fable-5-1](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5-1.html)"},
		{"https://developers.openai.com/api/docs/pricing", "[pricing](https://developers.openai.com/api/docs/pricing)"},
		{"https://developers.openai.com/", "[developers.openai.com](https://developers.openai.com/)"},
		{"https://developers.openai.com", "[developers.openai.com](https://developers.openai.com)"},
		{"https://example.com/docs/a%5Db", "[example.com](https://example.com/docs/a%5Db)"},
		{"https://developers.openai.com/api/docs/models/gpt-6.1-sol", "[gpt-6.1-sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)"},
		{"https://platform.claude.com/docs/en/models/claude-opus-5/overview.md", "[claude-opus-5](https://platform.claude.com/docs/en/models/claude-opus-5/overview.md)"},
	}
	for _, tt := range tests {
		t.Run(tt.url, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, link(tt.url))
		})
	}
}

// TestRenderContextLine checks the first line under the title: what the bot
// is, in plain words, for a reviewer who hasn't met it, and where the guide is.
func TestRenderContextLine(t *testing.T) {
	t.Parallel()

	const baseURL = "https://example.test/blob/abc"

	paragraphs := strings.SplitN(Render(oneVendor(validReport(), Base{}), "", nil, Base{URL: baseURL}), "\n\n", 3)
	require.Len(t, paragraphs, 3)
	assert.Equal(t, "**Context:** an AI agent per vendor checks the SDK's model catalog against the vendor's docs, and this PR holds the fixes it proposes. "+
		"[REVIEW.md]("+baseURL+"/REVIEW.md#bot-catalog-prs) says how to review a bot PR.", paragraphs[1])
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
			want: "**4 changes** to the catalogue: 1 price, 3 changes with no row.",
		},
		{
			name:    "counts what stays as it is",
			vendors: oneVendor(unchanged, Base{}),
			base:    Base{Decisions: []reconcile.Decision{keepLimit}, Today: "2026-10-14"},
			want:    "**1 change** to the catalogue: 1 price. Not changed: 3 open questions, 1 held by a decision, 2 skipped offerings.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// The title, the Context line, then this line.
			paragraphs := strings.SplitN(Render(tt.vendors, "", nil, tt.base), "\n\n", 4)
			require.Len(t, paragraphs, 4)
			assert.Equal(t, tt.want, paragraphs[2])
		})
	}
}

// TestRenderChecksLine checks the line under the summary (proposal 3): what a
// reviewer must do before marking the PR ready, and why: no automated check
// compares a value with its source.
func TestRenderChecksLine(t *testing.T) {
	t.Parallel()

	const check = "**Before you mark it ready:** check each value in Changes against its source, because no automated check does."

	tests := []struct {
		name      string
		codeFiles []string
		want      string
		section   bool
	}{
		{
			name: "no code files",
			want: check,
		},
		{
			name:      "one code file",
			codeFiles: []string{"providers/anthropic/alias_test.go"},
			want:      check + " Also read the file under Code changed as code: marking ready runs it with secrets.",
			section:   true,
		},
		{
			name:      "several code files",
			codeFiles: []string{"go.mod", "providers/anthropic/models.go", "providers/bedrock/models.go", "providers/openai/models.go"},
			want:      check + " Also read the 4 files under Code changed as code: marking ready runs them with secrets.",
			section:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := Render(oneVendor(validReport(), Base{}), "", tt.codeFiles, Base{})

			// The title, the Context line, the summary line, then this line.
			paragraphs := strings.SplitN(body, "\n\n", 5)
			require.Len(t, paragraphs, 5)
			assert.Equal(t, tt.want, paragraphs[3])
			assert.Equal(t, tt.section, strings.Contains(body, "### Code changed"), "Code changed is a section whenever code changed")
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

	t.Run("a cut never leaves a fold open", func(t *testing.T) {
		t.Parallel()

		skips := Report{}
		for i := range 40 {
			skips.Skipped = append(skips.Skipped, Skipped{Offering: fmt.Sprintf("claude-model-%03d", i), Reason: "vendor_excluded", Rule: "no-rule"})
		}

		for i := range 20 {
			skips.Changes = append(skips.Changes, Change{
				Offering: fmt.Sprintf("bedrock:us.anthropic.claude-model-%03d", i), Field: "price", Old: "input 3.00", New: "input 3.30",
				SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing.md", Rule: "price-follow",
			})
		}

		vendors := []Vendor{{Name: "anthropic", Report: skips}, {Name: "openai", Report: Report{Changes: validReport().Changes}}}
		for limit := 3000; limit <= 20000; limit += 25 {
			body := RenderCapped(vendors, "", nil, Base{}, limit)
			require.LessOrEqual(t, len(body), limit)
			require.Equal(t, strings.Count(body, "<details>"), strings.Count(body, "</details>"), "limit %d leaves a fold open", limit)
		}
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
// next vendor's, or to the snapshot diff's.
func section(t *testing.T, body, vendor string) string {
	t.Helper()

	_, rest, found := strings.Cut(body, "\n## "+vendor+"\n")
	require.True(t, found, "the body has no section for %s", vendor)

	rest, _, _ = strings.Cut(rest, "\n## ")

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
// (code changed, routing changes) shows once, outside the
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

	for _, once := range []string{"### Code changed (2)", "### Routing changes (1)"} {
		assert.Equal(t, 1, strings.Count(body, once), "%q shows once", once)
	}

	assert.Less(t, strings.Index(body, "### Routing changes"), strings.Index(body, "## anthropic"), "the PR's parts come before the vendors")
	assert.NotContains(t, body, "Snapshot diff", "the diff stays in Files changed")
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
	for _, part := range []string{"Reused from the agent's run on 2026-10-28 (run 123); no new agent run this time.", "### Changes (1)", "### Changed with no row (", "### Held by decisions (1)", "<summary>Skipped (1)"} {
		assert.Contains(t, anthropic, part)
	}

	assert.Equal(t, "The run found nothing to change.", strings.TrimSpace(section(t, body, "openai")))
	assert.Contains(t, section(t, body, "google"), "This vendor's run failed, so this PR has nothing from it.")
	assert.Contains(t, body, "Recorded decisions: 1, in `decisions.yaml`.")
}

// TestRenderUnreportedOnly checks a vendor whose report has no rows but whose
// patch changes offerings: it lists them under Changed with no row instead of
// saying the run found nothing.
func TestRenderUnreportedOnly(t *testing.T) {
	t.Parallel()

	before, after := effectSnapshots(t)
	body := Render([]Vendor{{Name: "anthropic", Report: Report{}, Before: before, After: after}}, "+  \"input\": 5", nil, Base{})

	assert.Contains(t, section(t, body, "anthropic"), "### Changed with no row (4)")
	assert.NotContains(t, body, "The run found nothing to change.")
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
	assert.Equal(t, "The run changed nothing; its rows are under Open questions above.", strings.TrimSpace(section(t, body, "openai")),
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
		"They don't block merging: merging changes nothing they ask about, and a later run asks again.\n>\n"+
		"> 1. anthropic · `claude-sonnet-5-5` · token limit\n> 2. openai · us copy: `bedrock:us.openai.gpt-6-1` · new copy\n"+
		"> 3. openai · global copy: `bedrock:global.openai.gpt-6-1` · new copy\n\n"+
		"**To answer a question,** open a PR into main that does what its Decide line says. If the same case can come up again, add a rule to `RULES.md` instead.\n\n")

	for _, block := range []string{
		"### Open questions (3)\n\n**1. anthropic · `claude-sonnet-5-5` · token limit** · no rule covers this (`no-rule`)\n",
		"**2. openai · us copy: `bedrock:us.openai.gpt-6-1` · new copy** · no rule covers this (`no-rule`)\n",
		"**3. openai · global copy: `bedrock:global.openai.gpt-6-1` · new copy** · no rule covers this (`no-rule`)\n",
	} {
		assert.Contains(t, body, block)
	}

	assert.Less(t, strings.Index(body, "### Open questions"), strings.Index(body, "## anthropic"), "the questions come before the vendors")
	assert.NotContains(t, section(t, body, "anthropic"), "no rule covers this")
	assert.NotContains(t, section(t, body, "openai"), "no rule covers this")
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

	// The agent's own words: a question's why, required, and an optional
	// rule draft.
	for _, property := range []string{"why", "rule_when", "rule_does"} {
		require.Contains(t, items("needs_human").Properties, property)
		assert.Equal(t, sentencePattern.String(), items("needs_human").Properties[property].Pattern, "needs_human %s pattern", property)
	}

	assert.Contains(t, items("needs_human").Required, "why", "every question says why the bot asks")
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
			`"source_urls": ["https://platform.claude.com/a"], "rule": "no-rule", "why": "The page and the catalogue differ."` + sides + `}]}`
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
		{"no why", strings.Replace(question("limit", ""), `, "why": "The page and the catalogue differ."`, "", 1), true},
		{"an empty why", strings.Replace(question("limit", ""), `"The page and the catalogue differ."`, `""`, 1), true},
		{"markup in the why", strings.Replace(question("limit", ""), `"The page and the catalogue differ."`, `"Safe to **approve**"`, 1), true},
		{"a rule draft", question("limit", `, "rule_when": "the page lists one token more", "rule_does": "keeps ours"`), false},
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

// TestRenderQuestionBlock checks one Open questions row: the two sides and the
// agent's notes, then the Decide line (a rule's fixed question, or for a
// no-rule row with both sides keep or change, and add or leave out for a new
// model or copy, with the decision that stops the question), then the sources.
func TestRenderQuestionBlock(t *testing.T) {
	t.Parallel()

	const (
		noRule  = "**1. `claude-sonnet-5-5` · token limit** · no rule covers this (`no-rule`)\n"
		sources = "- **Sources:** [a](https://platform.claude.com/a)\n\n"
		KEEP1   = "- **Decide:** keep what we have, or change it to match the vendor's page? To keep it and stop the question coming back, add a `keep` for it to `decisions.yaml`.\n"
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
			want: noRule + "- **We have:** `65535`\n- **Vendor's page:** `65536`\n- **Agent's notes:** `providers/anthropic/models.go`\n" +
				"- **Decide:** keep what we have, or change it to match the vendor's page? To keep it and stop the question coming back, add a `keep` for it to `decisions.yaml`.\n" + sources,
		},
		{
			name: "both sides and nothing else found",
			item: ask("no-rule", "65535", "65536"),
			want: noRule + "- **We have:** `65535`\n- **Vendor's page:** `65536`\n- **Decide:** keep what we have, or change it to match the vendor's page? To keep it and stop the question coming back, add a `keep` for it to `decisions.yaml`.\n" + sources,
		},
		{
			name: "a new copy with both sides: leave it out or add it",
			item: NeedsHuman{
				Offering: "bedrock:us.anthropic.claude-sonnet-5-5", Field: reconcile.FieldNewCopy, Catalogue: "none", Page: "input 3.30",
				Rule: "no-rule", SourceURLs: []string{"https://platform.claude.com/a"},
			},
			want: "**1. us copy: `bedrock:us.anthropic.claude-sonnet-5-5` · new copy** · no rule covers this (`no-rule`)\n" +
				"- **We have:** `none`\n- **Vendor's page:** `input 3.30`\n- **Decide:** add this copy, or leave it out? To leave it out and stop the question coming back, add a `hold` for it to `decisions.yaml`.\n" + sources,
		},
		{
			name: "a rule's row keeps its question",
			item: ask("out-of-reach", "none", "claude-opus-5"),
			want: "**1. `claude-sonnet-5-5` · token limit** · rule `out-of-reach`\n" +
				"- **We have:** `none`\n- **Vendor's page:** `claude-opus-5`\n- **Decide:** Which replacement?\n" + sources,
		},
		{
			name: "one side alone keeps the question",
			item: ask("no-rule", "", "65536"),
			want: noRule + "- **Vendor's page:** `65536`\n- **Decide:** No rule covers this. Decide.\n" + sources,
		},
		{
			name: "the agent's why, after the two sides",
			item: NeedsHuman{
				Offering: "claude-sonnet-5-5", Field: "limit", Catalogue: "65535", Page: "65536", Values: []string{"providers/anthropic/models.go"},
				Why: "The page lists one token more than we do; it may count the stop token.", Rule: "no-rule", SourceURLs: []string{"https://platform.claude.com/a"},
			},
			want: noRule + "- **We have:** `65535`\n- **Vendor's page:** `65536`\n" +
				"- **Why the bot asks:** The page lists one token more than we do; it may count the stop token.\n" +
				"- **Agent's notes:** `providers/anthropic/models.go`\n" + KEEP1 + sources,
		},
		{
			name: "no sides: the notes, then the question",
			item: ask("no-rule", "", "", "65535", "65536"),
			want: noRule + "- **Agent's notes:** `65535`, `65536`\n- **Decide:** No rule covers this. Decide.\n" + sources,
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
		"> [!IMPORTANT]\n> **2 open questions,** listed under Open questions below. They don't block merging: merging changes nothing they ask about, and a later run asks again.\n>\n" +
			"> 1. `bedrock:global.anthropic.claude-fable-5-1` and 2 more · capability\n> 2. `bedrock:global.anthropic.claude-haiku-5-5` · capability\n\n",
		"### Open questions (2)\n\n" +
			"**1. `bedrock:global.anthropic.claude-fable-5-1` and 2 more · capability** · no rule covers this (`no-rule`)\n" +
			"- **Offerings:** `bedrock:global.anthropic.claude-fable-5-1`, `bedrock:us.anthropic.claude-fable-5-1`, `bedrock:global.anthropic.claude-opus-5-5`\n" +
			"- **We have:** `structured_output false`\n- **Vendor's page:** `Structured outputs supported`\n" +
			"- **Agent's notes:** `providers/bedrock/capabilities_test.go`, `claudeNoSchemaCaps`\n" +
			"- **Decide:** keep what we have, or change it to match the vendor's page? To keep it and stop the question coming back, add a `keep` for each to `decisions.yaml`.\n" +
			"- **Sources:** [a](https://docs.aws.amazon.com/a), [b](https://docs.aws.amazon.com/b)\n\n" +
			"**2. `bedrock:global.anthropic.claude-haiku-5-5` · capability** · no rule covers this (`no-rule`)\n",
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
			"Check them in `catalog/snapshot.json` under Files changed.\n\n- `bedrock:au.anthropic.claude-sonnet-5-5`, `bedrock:jp.anthropic.claude-sonnet-4-5`, `bedrock:us.anthropic.claude-sonnet-4-5`\n")
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
		names, n := base.unreported(Report{}, "anthropic")
		assert.Equal(t, []string{"anthropic/claude-sonnet-5-5 (facts)"}, names)
		assert.Equal(t, 1, n)

		names, _ = base.unreported(Report{Changes: []Change{{Offering: "claude-sonnet-5-5"}}}, "anthropic")
		assert.Empty(t, names, "a row on one of its offerings covers it")
	})

	t.Run("an unprintable ID is counted", func(t *testing.T) {
		t.Parallel()

		base := Base{Before: snapshot("2026-09-30", "x"), After: snapshot("2026-09-30", "<b>evil</b>")}
		names, n := base.unreported(Report{}, "anthropic")
		assert.Equal(t, []string{"x (removed)", "1 offering with an ID that isn't safe to print"}, names)
		assert.Equal(t, 2, n, "each unprintable offering counts")
	})
}

// TestRenderRuleDrafts checks the agent's rule drafts (Adam, 31 Oct): a draft
// shows once it is on two or more no-rule offerings across the PR's vendors,
// with the questions it would settle; one offering's draft, a rule's row and
// a held row show none.
func TestRenderRuleDrafts(t *testing.T) {
	t.Parallel()

	const (
		when = "a Bedrock card prints a model's price only for calls within one region"
		does = "leaves its us. and global. copies out until AWS prints their price"
	)

	ask := func(prefix, model, rule, ruleWhen, ruleDoes string) NeedsHuman {
		return NeedsHuman{
			Offering: "bedrock:" + prefix + ".openai." + model, Field: reconcile.FieldNewCopy, Rule: rule, Page: "In-Region rate only " + prefix + " " + model,
			Why: "AWS prints no price for this copy.", RuleWhen: ruleWhen, RuleDoes: ruleDoes, SourceURLs: []string{"https://docs.aws.amazon.com/" + model},
		}
	}
	held := reconcile.Decision{
		ID: "DEC-009", Offering: "bedrock:us.openai.gpt-5.4", Field: reconcile.FieldNewCopy, Kind: reconcile.KindHold, Reason: "Agreed in review.", ReviewBy: "2027-01-13",
	}
	draft := "**Rules that would settle these** (the agent's drafts, for cases on two or more offerings; to use one, add it to RULES.md in a reviewed PR):\n\n"

	tests := []struct {
		name        string
		vendors     []Vendor
		decisions   []reconcile.Decision
		contains    []string
		notContains []string
	}{
		{
			name: "one draft on three offerings, across two vendors' questions",
			vendors: []Vendor{
				{Name: "openai", Report: Report{NeedsHuman: []NeedsHuman{ask("us", "gpt-5.5", "no-rule", when, does), ask("global", "gpt-5.5", "no-rule", when, does)}}},
				{Name: "anthropic", Report: Report{NeedsHuman: []NeedsHuman{ask("us", "claude-x", "no-rule", " "+when+" ", does+".")}}},
			},
			contains: []string{draft + "- Questions 1-3 (3 offerings): When " + when + ", the bot " + does + ".\n\n"},
		},
		{
			name:        "a draft on one offering shows none",
			vendors:     []Vendor{{Name: "openai", Report: Report{NeedsHuman: []NeedsHuman{ask("us", "gpt-5.5", "no-rule", when, does), ask("us", "gpt-5.4", "no-rule", "another case", does)}}}},
			notContains: []string{"Rules that would settle these"},
		},
		{
			name:        "a rule's rows show none",
			vendors:     []Vendor{{Name: "openai", Report: Report{NeedsHuman: []NeedsHuman{ask("us", "gpt-5.5", "out-of-reach", when, does), ask("global", "gpt-5.5", "out-of-reach", when, does)}}}},
			notContains: []string{"Rules that would settle these"},
		},
		{
			name:        "held rows don't count",
			vendors:     []Vendor{{Name: "openai", Report: Report{NeedsHuman: []NeedsHuman{ask("us", "gpt-5.5", "no-rule", when, does), ask("us", "gpt-5.4", "no-rule", when, does)}}}},
			decisions:   []reconcile.Decision{held},
			notContains: []string{"Rules that would settle these"},
		},
		{
			name: "two drafts, each with its questions",
			vendors: []Vendor{{Name: "openai", Report: Report{NeedsHuman: []NeedsHuman{
				ask("us", "gpt-5.5", "no-rule", when, does), ask("us", "gpt-5.6", "no-rule", "another case", "asks a person"),
				ask("global", "gpt-5.5", "no-rule", when, does), ask("global", "gpt-5.6", "no-rule", "another case", "asks a person"),
			}}}},
			contains: []string{draft + "- Questions 1 and 3 (2 offerings): When " + when + ", the bot " + does + ".\n" +
				"- Questions 2 and 4 (2 offerings): When another case, the bot asks a person.\n\n"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body := Render(tt.vendors, "", nil, Base{Rules: testRules, Decisions: tt.decisions, Today: "2026-10-15"})
			for _, want := range tt.contains {
				assert.Contains(t, body, want)
			}

			for _, unwanted := range tt.notContains {
				assert.NotContains(t, body, unwanted)
			}

			assert.NotContains(t, body, "Suggested rules")
		})
	}
}

// TestRenderFolded checks what sits behind <details> (proposal 6): each
// vendor's Skipped; the region questions sit once for every vendor, above
// the vendor sections.
func TestRenderFolded(t *testing.T) {
	t.Parallel()

	limit := func(offering string) NeedsHuman {
		return NeedsHuman{Offering: offering, Field: "limit", Rule: "no-rule", Values: []string{"65536"}, SourceURLs: []string{"https://platform.claude.com/x"}}
	}
	waiting := Skipped{Offering: "bedrock:in.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"}

	body := Render([]Vendor{
		{Name: "anthropic", Report: Report{NeedsHuman: []NeedsHuman{limit("claude-opus-5")}, Skipped: []Skipped{waiting, {Offering: "claude-haiku-4-5", Reason: "vendor_excluded", Rule: "no-rule"}}}},
		{Name: "openai", Report: Report{NeedsHuman: []NeedsHuman{limit("gpt-6-1")}, Changes: validReport().Changes}},
	}, "", nil, Base{})

	assert.Contains(t, section(t, body, "anthropic"), "<details><summary>Skipped (1): left out by a rule</summary>\n\n| Rule | Why | Offerings |\n|---|---|---|\n| `no-rule` | `vendor_excluded` | `claude-haiku-4-5` |\n\n</details>\n")
	assert.NotContains(t, body, "Suggested rules")

	for _, once := range []string{"**3. bedrock · offer region `in` (India)?"} {
		assert.Equal(t, 1, strings.Count(body, once), "%q shows once", once)
		assert.Less(t, strings.Index(body, once), strings.Index(body, "## anthropic"), "%q comes before the vendors", once)
	}
}

// TestRenderRegionQuestion checks that a region question's way to decline the
// region names that region and today (proposal 9).
func TestRenderRegionQuestion(t *testing.T) {
	t.Parallel()

	report := Report{Skipped: []Skipped{{Offering: "bedrock:us-gov.anthropic.claude-opus-5", Reason: "region_not_offered", Rule: "region-not-offered"}}}
	body := Render(oneVendor(report, Base{}), "", nil, Base{Today: "2026-10-30"})

	assert.Contains(t, body, "list it under Not offered in RULES.md as `` `us-gov` (declined 2026-10-30) ``")
	assert.FileExists(t, filepath.Join("..", "..", addModelSkill), "the Decide line links the add-model skill")
	assert.Equal(t, "The run changed nothing; its rows are under Open questions above.", strings.TrimSpace(section(t, body, "anthropic")))
}

// TestChangeKinds checks the kind of each change row (proposal 4), read from
// the catalogue before and after, never from the report's values.
func TestChangeKinds(t *testing.T) {
	t.Parallel()

	parse := func(raw string) map[string]any {
		decoder := json.NewDecoder(strings.NewReader(raw))
		decoder.UseNumber()

		var offering map[string]any
		require.NoError(t, decoder.Decode(&offering))

		return offering
	}
	rates := func(input, output int) string {
		return fmt.Sprintf(`{"pricing": {"default": {"base": {"input_microcents_per_million": %d, "output_microcents_per_million": %d}}}}`, input, output)
	}

	tests := []struct {
		name, field, before, after string
		want                       []string
	}{
		{"a price goes down", "price", rates(300, 900), rates(200, 900), []string{"price down"}},
		{"a price goes up", "price", rates(200, 900), rates(300, 900), []string{"price up"}},
		{"one rate up, one down", "price", rates(300, 900), rates(200, 1000), []string{"price"}},
		{"only a tier or bracket changes", "price", rates(300, 900), rates(300, 900), []string{"price"}},
		{"a rate set to $0, kept as 0", "price", rates(300, 900), rates(0, 900), []string{"price down", "rate to zero"}},
		{"a rate set to $0, left out", "price", rates(300, 900), `{"pricing": {"default": {"base": {"output_microcents_per_million": 900}}}}`, []string{"price", "rate to zero"}},
		{"max input", "limit", `{"constraints": {"max_input_tokens": 100}}`, `{"constraints": {"max_input_tokens": 200}}`, []string{"max input"}},
		{"max output", "limit", `{"constraints": {"max_output_tokens": 200}}`, `{"constraints": {"max_output_tokens": 100}}`, []string{"max output"}},
		{
			"a flag off, and the modalities", "capability", `{"capabilities": {"tools": true}, "modalities": {"input": ["text", "image"]}}`,
			`{"capabilities": {"tools": false}, "modalities": {"input": ["text"]}}`,
			[]string{"flag off", "modality"},
		},
		{"a flag on", "capability", `{"capabilities": {"tools": false}}`, `{"capabilities": {"tools": true}}`, []string{"flag on"}},
		{"reasoning efforts", "capability", `{"reasoning": {"efforts": ["low", "high"]}}`, `{"reasoning": {"efforts": ["low"]}}`, []string{"reasoning"}},
		{"launch date", "lifecycle", `{"lifecycle": {"stage": "ga"}}`, `{"lifecycle": {"stage": "ga", "available": "2026-10-01"}}`, []string{"launch date"}},
		{"deprecated date", "lifecycle", `{"lifecycle": {"stage": "ga"}}`, `{"lifecycle": {"stage": "ga", "deprecated": "2026-10-01"}}`, []string{"deprecation"}},
		{"stage moves to deprecated", "lifecycle", `{"lifecycle": {"stage": "ga"}}`, `{"lifecycle": {"stage": "deprecated"}}`, []string{"deprecation"}},
		{"shutdown date", "lifecycle", `{"lifecycle": {"stage": "ga"}}`, `{"lifecycle": {"stage": "ga", "retires": "2027-04-08"}}`, []string{"retirement"}},
		{"a new model", reconcile.FieldNewModel, `{}`, `{"id": "x"}`, []string{"new offering"}},
		{"a new copy", reconcile.FieldNewCopy, `{}`, `{"id": "x"}`, []string{"new offering"}},
		{"replacement", "replacement", `{}`, `{"lifecycle": {"replaced_by": "y"}}`, []string{"replacement"}},
		{"a stage the table doesn't know", "lifecycle", `{"lifecycle": {"stage": "preview"}}`, `{"lifecycle": {"stage": "ga"}}`, []string{"other"}},
		{"a field the table doesn't know", "location", `{}`, `{"locations": ["us"]}`, []string{"other"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, kinds(tt.field, parse(tt.before), parse(tt.after)))
		})
	}
}

// TestEveryKindHasAnImpact checks the impact table's 16 kinds: each has a
// name and a sentence each for SDK users and ADP, and kinds() names only
// these.
func TestEveryKindHasAnImpact(t *testing.T) {
	t.Parallel()

	require.Len(t, impacts, 16)

	for _, kind := range impacts {
		assert.NotEmpty(t, kind.name, kind.key)
		assert.NotEmpty(t, kind.sdk, kind.key)
		assert.NotEmpty(t, kind.adp, kind.key)
	}

	for _, field := range fields {
		for _, kind := range kinds(field, map[string]any{}, map[string]any{"id": "x"}) {
			assert.GreaterOrEqual(t, kindIndex(kind), 0, "%s gives kind %q, which has no impact", field, kind)
		}
	}
}

// TestRenderImpactTable checks "What changes for customers" (proposal 4): one
// row per kind, how many offerings it touches, and what it means for SDK users
// and for ADP, each once they get the new catalogue; and each row's Effect cell.
func TestRenderImpactTable(t *testing.T) {
	t.Parallel()

	t.Run("a column per vendor", func(t *testing.T) {
		t.Parallel()

		before, after := effectSnapshots(t)
		openaiBefore, openaiAfter := openaiSnapshots(t)
		cacheRead := Report{Changes: []Change{{Offering: "claude-sonnet-5-5", Field: "price", Old: "cache read 0.20", New: "cache read 0.10", Rule: "price-follow"}}}
		input := Report{Changes: []Change{{Offering: "gpt-6-1", Field: "price", Old: "input 2.00", New: "input 2.50", Rule: "price-follow"}}}

		body := Render([]Vendor{
			{Name: "anthropic", Report: cacheRead, Before: before, After: after},
			{Name: "openai", Report: input, Before: openaiBefore, After: openaiAfter},
		}, "", nil, Base{})

		assert.Contains(t, body, "### What changes for customers\n\n| Change | Offerings | SDK users, once they upgrade | ADP, once it moves to this SDK and redeploys |\n|---|---|---|---|\n"+
			"| price cut | 1 | Cost estimates for new calls drop. Bills don't change: vendors bill their own prices. | New calls record less spend, so budgets fill more slowly. Past spend stays. |\n"+
			"| price rise | 1 | Cost estimates for new calls rise. Bills don't change: vendors bill their own prices. | New calls record more spend, so budgets reach their limit sooner. Past spend stays. |\n\n"+
			"You can push commits here: the bot never removes them. It doesn't read comments.\n\n")
		assert.Less(t, strings.Index(body, "### What changes for customers"), strings.Index(body, "You can push commits here"), "how pushing works comes after it")
		assert.Less(t, strings.Index(body, "You can push commits here"), strings.Index(body, "## anthropic"), "the table is at the top")
	})

	t.Run("one vendor, and the Effect cell's short form", func(t *testing.T) {
		t.Parallel()

		snapshot := func(maxInput int) *reconcile.Snapshot {
			snap, err := reconcile.ParseSnapshot(fmt.Appendf(nil, `{"providers": [{"provider": "anthropic", "offerings": [{"id": "claude-opus-5-5", "model": "anthropic/claude-opus-5-5",
				"constraints": {"max_input_tokens": %d}, "modalities": {"input": ["text"]}}]}]}`, maxInput))
			require.NoError(t, err)

			return &snap
		}
		limit := Report{Changes: []Change{{Offering: "claude-opus-5-5", Field: "limit", Old: "200000", New: "1000000", Rule: "price-follow"}}}

		body := Render(oneVendor(limit, Base{Before: snapshot(200000), After: snapshot(1000000)}), "", nil, Base{})

		assert.Contains(t, body, "| max input tokens | 1 | Agents trim older turns only when a conversation nears the new window, and send requests up to it. "+
			"If the vendor's real limit is lower, a call fails once and is retried shorter. | ADP's agents are built on the SDK's, so they change the same way. The console shows the new window. |\n\n")
		assert.Contains(t, body, "| `claude-opus-5-5` | limit | `200000` | `1000000` | agents use the new window |")
	})

	// #299: gpt-6.1-sol's input and output limits both changed, one row each,
	// and the table counted the offering twice in both kinds.
	t.Run("an offering counts once per kind", func(t *testing.T) {
		t.Parallel()

		snapshot := func(maxInput, maxOutput int) *reconcile.Snapshot {
			snap, err := reconcile.ParseSnapshot(fmt.Appendf(nil, `{"providers": [{"provider": "anthropic", "offerings": [{"id": "claude-opus-5-5", "model": "anthropic/claude-opus-5-5",
				"constraints": {"max_input_tokens": %d, "max_output_tokens": %d}}]}]}`, maxInput, maxOutput))
			require.NoError(t, err)

			return &snap
		}
		limits := Report{Changes: []Change{
			{Offering: "claude-opus-5-5", Field: "limit", Old: "MaxInputTokens 200000", New: "MaxInputTokens 1000000", Rule: "price-follow"},
			{Offering: "claude-opus-5-5", Field: "limit", Old: "MaxOutputTokens 64000", New: "MaxOutputTokens 128000", Rule: "price-follow"},
		}}

		body := Render(oneVendor(limits, Base{Before: snapshot(200000, 64000), After: snapshot(1000000, 128000)}), "", nil, Base{})

		assert.Contains(t, body, "| max input tokens | 1 |")
		assert.Contains(t, body, "| max output tokens | 1 |")
	})

	assert.NotContains(t, Render(oneVendor(validReport(), Base{}), "", nil, Base{}), "What changes for customers", "no catalogue, no table")
}

// routeBase is a Base whose routes move Sydney's and Melbourne's bare-name
// Sonnet 5.5 calls from the global copy to the new AU one, which costs 10%
// more, and keep Virginia's on the US copy; after is the AU copy's ID.
func routeBase(t *testing.T, after string) Base {
	t.Helper()

	before, afterSnapshot := effectSnapshots(t)
	routesBefore, err := ParseRoutes([]byte("ap-southeast-2 anthropic.claude-sonnet-5-5 global.anthropic.claude-sonnet-5-5\n" +
		"ap-southeast-4 anthropic.claude-sonnet-5-5 global.anthropic.claude-sonnet-5-5\nus-east-1 anthropic.claude-sonnet-5-5 us.anthropic.claude-sonnet-5-5\n"))
	require.NoError(t, err)
	routesAfter, err := ParseRoutes([]byte("ap-southeast-2 anthropic.claude-sonnet-5-5 " + after + "\n" +
		"ap-southeast-4 anthropic.claude-sonnet-5-5 " + after + "\nus-east-1 anthropic.claude-sonnet-5-5 us.anthropic.claude-sonnet-5-5\n"))
	require.NoError(t, err)

	return Base{Before: before, After: afterSnapshot, RoutesBefore: routesBefore, RoutesAfter: routesAfter}
}

// TestRenderRoutingDiagram checks the diagram above the Routing changes table
// (proposal 11): each move's bare name and callers, the copy before, and the
// copy after with the input price change and where it processes data.
func TestRenderRoutingDiagram(t *testing.T) {
	t.Parallel()

	auCopy := Report{Changes: []Change{{
		Offering: "bedrock:au.anthropic.claude-sonnet-5-5", Field: "new_copy", New: "au.anthropic.claude-sonnet-5-5",
		SourceURL: "https://platform.claude.com/docs/en/about-claude/pricing", Rule: "new-copy-offered-region",
	}}}
	base := routeBase(t, "au.anthropic.claude-sonnet-5-5")

	assert.Contains(t, Render(oneVendor(auCopy, base), "", nil, base), "Check the price and where the data is processed.\n\n"+
		"```mermaid\nflowchart TB\n"+
		"  n1[\"anthropic.claude-sonnet-5-5<br>from ap-southeast-2, ap-southeast-4\"]\n"+
		"  n1 -. before .-> b1[\"global.anthropic.claude-sonnet-5-5<br>any region\"]\n"+
		"  n1 == \"after, input +10%\" ==> a1[\"au.anthropic.claude-sonnet-5-5<br>Australia\"]\n"+
		"```\n\n| Model | Callers in |")
}

// TestRoutingDiagramRefusesUnsafeIDs checks that the diagram is left out when
// an ID from catalog-routes, which runs the patched code, has a character
// other than letters, digits and .:-_/; the table still shows.
func TestRoutingDiagramRefusesUnsafeIDs(t *testing.T) {
	t.Parallel()

	// safeValue lets + through, so ParseRoutes accepts it.
	base := routeBase(t, "au.anthropic.claude-sonnet-5-5+x")
	body := Render(oneVendor(validReport(), base), "", nil, base)

	assert.Contains(t, body, "| `au.anthropic.claude-sonnet-5-5+x` |")
	assert.NotContains(t, body, "```mermaid")
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
