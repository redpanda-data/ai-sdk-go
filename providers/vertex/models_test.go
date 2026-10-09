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

package vertex_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

// TestCatalogModelSet pins the catalog to exactly the Gemini + Claude chat
// scope, keyed by bare publisher ID; no image-generation, embedding, or
// live-audio model belongs here.
func TestCatalogModelSet(t *testing.T) {
	t.Parallel()

	cat := vertex.Catalog()
	require.NotNil(t, cat)

	got := make([]string, 0, cat.Len())
	for _, o := range cat.All() {
		got = append(got, o.ID)
	}

	assert.ElementsMatch(t, []string{
		"gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
		"gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
		"gemini-3.1-pro-preview", "gemini-3-flash-preview",
		"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
		"claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5",
		"claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5@20251101",
		"claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-4-6", "claude-sonnet-4-5@20250929",
		"claude-haiku-5-5", "claude-haiku-4-5",
	}, got)
}

// TestCatalogProviderName pins the literal key in-package so a rename has
// to be deliberate; cmd/catalog-snapshot cross-checks it against Name().
func TestCatalogProviderName(t *testing.T) {
	t.Parallel()

	assert.Equal(t, llm.ProviderID("gcp.vertex"), vertex.Catalog().Provider())
}

// TestOfferingAttributes checks every offering carries its publisher.
// The publisher is the one attribute the request path needs that the
// offering ID does not already carry, so a missing or drifted value is a
// routing bug.
func TestOfferingAttributes(t *testing.T) {
	t.Parallel()

	wantPublisher := map[string]catalog.Publisher{
		vertex.ModelGemini38Flash:       "google",
		vertex.ModelGemini37Flash:       "google",
		vertex.ModelGemini36Flash:       "google",
		vertex.ModelGemini35Flash:       "google",
		vertex.ModelGemini35FlashLite:   "google",
		vertex.ModelGemini31FlashLite:   "google",
		vertex.ModelGemini31ProPreview:  "google",
		vertex.ModelGemini3FlashPreview: "google",
		vertex.ModelGemini25Pro:         "google",
		vertex.ModelGemini25Flash:       "google",
		vertex.ModelGemini25FlashLite:   "google",
		vertex.ModelClaudeFable51:       "anthropic",
		vertex.ModelClaudeFable5:        "anthropic",
		vertex.ModelClaudeOpus55:        "anthropic",
		vertex.ModelClaudeOpus5:         "anthropic",
		vertex.ModelClaudeOpus48:        "anthropic",
		vertex.ModelClaudeOpus47:        "anthropic",
		vertex.ModelClaudeOpus46:        "anthropic",
		vertex.ModelClaudeOpus45:        "anthropic",
		vertex.ModelClaudeSonnet55:      "anthropic",
		vertex.ModelClaudeSonnet5:       "anthropic",
		vertex.ModelClaudeSonnet46:      "anthropic",
		vertex.ModelClaudeSonnet45:      "anthropic",
		vertex.ModelClaudeHaiku55:       "anthropic",
		vertex.ModelClaudeHaiku45:       "anthropic",
	}

	require.Len(t, vertex.Catalog().All(), len(wantPublisher),
		"wantPublisher covers every offering, so a new one lands here before its publisher is checked")

	for _, o := range vertex.Catalog().All() {
		require.Containsf(t, wantPublisher, o.ID, "%s has no expected publisher", o.ID)
		assert.Equalf(t, wantPublisher[o.ID], o.Facts().Publisher, "%s publisher", o.ID)
	}
}

func TestNoNamespacedPricingKey(t *testing.T) {
	t.Parallel()

	for id := range vertex.Catalog().PricingByID() {
		assert.Falsef(t, strings.HasPrefix(id, "vertex."), "pricing key %q must be a bare model ID, not vertex.-prefixed", id)
	}
}

func TestGeminiRegionalOverride(t *testing.T) {
	t.Parallel()

	flash := struct{ global, regional pricing.Rates }{
		pricing.NewRates(0.75, 3.75, 0.075), pricing.NewRates(0.825, 4.125, 0.0825),
	}
	cases := map[string]struct{ global, regional pricing.Rates }{
		vertex.ModelGemini38Flash: flash,
		vertex.ModelGemini37Flash: flash,
		vertex.ModelGemini36Flash: flash,
		vertex.ModelGemini35Flash: {
			pricing.NewRates(1.50, 9.00, 0.15), pricing.NewRates(1.65, 9.90, 0.165),
		},
		vertex.ModelGemini35FlashLite: {
			pricing.NewRates(0.30, 2.50, 0.03), pricing.NewRates(0.33, 2.75, 0.033),
		},
		vertex.ModelGemini31FlashLite: {
			pricing.NewRates(0.25, 1.50, 0.025), pricing.NewRates(0.275, 1.65, 0.0275),
		},
	}

	for id, want := range cases {
		info, ok := vertex.Catalog().PricingByID()[id]
		require.Truef(t, ok, "no pricing for %s", id)

		assert.Equalf(t, want.global, info.Default.Base, "%s global default rate", id)

		served := vertex.LocationsForModel(id)
		require.NotEmptyf(t, served, "expected served locations for %s", id)

		require.NotEmptyf(t, info.Overrides, "expected at least one non-global rate override for %s", id)

		for _, ov := range info.Overrides {
			require.NotEmptyf(t, ov.Match.Region, "override has empty Region (would also match global): %+v", ov.Match)
			assert.Equalf(t, want.regional, ov.RateCard.Base, "%s region %q rate", id, ov.Match.Region)
			assert.Containsf(t, served, ov.Match.Region, "priced region %q is not a served location", ov.Match.Region)
			assert.NotEqualf(t, vertex.LocationGlobal, ov.Match.Region, "override region must be non-global")
		}

		// The reverse guard: every served non-global region must carry an
		// override. Without it, adding a served region to the matrix without a
		// rate silently bills that region at the global default, ~10% under the
		// published non-global rate, and nothing goes red.
		assertEveryNonGlobalRegionPriced(t, served, info.Overrides)
	}
}

// assertEveryNonGlobalRegionPriced checks the served-implies-priced
// direction: every served location other than global has a matching rate
// override. It is the reverse of the Containsf guard in the override loops,
// which only checks priced-implies-served.
func assertEveryNonGlobalRegionPriced(t *testing.T, served []string, overrides []pricing.Override) {
	t.Helper()

	for _, loc := range served {
		if loc == vertex.LocationGlobal {
			continue
		}

		priced := false

		for _, ov := range overrides {
			if ov.Match.Region == loc {
				priced = true
				break
			}
		}

		assert.Truef(t, priced, "served non-global region %q has no rate override", loc)
	}
}

// TestClaudeHaiku55PromptLengthTier pins Haiku 5.5's bracket above 100K input
// tokens on the global card and on every regional override.
func TestClaudeHaiku55PromptLengthTier(t *testing.T) {
	t.Parallel()

	info, ok := vertex.Catalog().PricingByID()[vertex.ModelClaudeHaiku55]
	require.True(t, ok)

	require.Len(t, info.Default.Brackets, 1)
	assert.Equal(t, int64(100_001), info.Default.Brackets[0].MinContextTokens)
	assert.Equal(t, pricing.NewRates(0.50, 2.50, 0.05).WithCacheCreation(0.625, 1.00, 0), info.Default.Brackets[0].Rates)

	require.Len(t, info.Overrides, 2)

	for _, ov := range info.Overrides {
		require.Lenf(t, ov.RateCard.Brackets, 1, "region %q", ov.Match.Region)
		assert.Equal(t, int64(100_001), ov.RateCard.Brackets[0].MinContextTokens)
		assert.Equalf(t, pricing.NewRates(0.55, 2.75, 0.055).WithCacheCreation(0.6875, 1.10, 0),
			ov.RateCard.Brackets[0].Rates, "region %q", ov.Match.Region)
	}
}

// TestClaudeRegionalOverride checks the Claude rates. Google's Agent Platform
// pricing page groups Sonnet 5 and Haiku 4.5 under "Models with regional
// pricing" (read 2026-09-08).
func TestClaudeRegionalOverride(t *testing.T) {
	t.Parallel()

	opus := struct{ global, regional pricing.Rates }{
		global:   pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		regional: pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
	}
	sonnet4 := struct{ global, regional pricing.Rates }{
		global:   pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
		regional: pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
	}
	cases := map[string]struct {
		global, regional pricing.Rates
	}{
		vertex.ModelClaudeFable51: {
			global:   pricing.NewRates(10.00, 50.00, 0.25).WithCacheCreation(12.50, 20.00, 0),
			regional: pricing.NewRates(11.00, 55.00, 0.275).WithCacheCreation(13.75, 22.00, 0),
		},
		vertex.ModelClaudeFable5: {
			global:   pricing.NewRates(10.00, 50.00, 1.00).WithCacheCreation(12.50, 20.00, 0),
			regional: pricing.NewRates(11.00, 55.00, 1.10).WithCacheCreation(13.75, 22.00, 0),
		},
		vertex.ModelClaudeOpus55: {
			global:   pricing.NewRates(4.00, 20.00, 0.20).WithCacheCreation(5.00, 8.00, 0),
			regional: pricing.NewRates(4.40, 22.00, 0.22).WithCacheCreation(5.50, 8.80, 0),
		},
		vertex.ModelClaudeOpus5:  opus,
		vertex.ModelClaudeOpus48: opus,
		vertex.ModelClaudeOpus47: opus,
		vertex.ModelClaudeOpus46: opus,
		vertex.ModelClaudeOpus45: opus,
		vertex.ModelClaudeSonnet55: {
			global:   pricing.NewRates(2.00, 10.00, 0.20).WithCacheCreation(2.50, 4.00, 0),
			regional: pricing.NewRates(2.20, 11.00, 0.22).WithCacheCreation(2.75, 4.40, 0),
		},
		vertex.ModelClaudeSonnet5: {
			global:   pricing.NewRates(2.00, 10.00, 0.20).WithCacheCreation(2.50, 4.00, 0),
			regional: pricing.NewRates(2.20, 11.00, 0.22).WithCacheCreation(2.75, 4.40, 0),
		},
		vertex.ModelClaudeSonnet46: sonnet4,
		vertex.ModelClaudeSonnet45: sonnet4,
		vertex.ModelClaudeHaiku55: {
			global:   pricing.NewRates(0.10, 0.50, 0.01).WithCacheCreation(0.125, 0.20, 0),
			regional: pricing.NewRates(0.11, 0.55, 0.011).WithCacheCreation(0.1375, 0.22, 0),
		},
		vertex.ModelClaudeHaiku45: {
			global:   pricing.NewRates(1.00, 5.00, 0.10).WithCacheCreation(1.25, 2.00, 0),
			regional: pricing.NewRates(1.10, 5.50, 0.11).WithCacheCreation(1.375, 2.20, 0),
		},
	}

	prices := vertex.Catalog().PricingByID()
	for id, want := range cases {
		info, ok := prices[id]
		require.Truef(t, ok, "no pricing for %s", id)
		assert.Equalf(t, want.global, info.Default.Base, "%s global default rate", id)

		served := vertex.LocationsForModel(id)
		require.NotEmptyf(t, served, "expected served locations for %s", id)

		require.NotEmptyf(t, info.Overrides, "%s must carry a non-global rate override", id)

		for _, ov := range info.Overrides {
			require.NotEmptyf(t, ov.Match.Region, "%s override has empty Region (would also match global): %+v", id, ov.Match)
			assert.Equalf(t, want.regional, ov.RateCard.Base, "%s region %q rate", id, ov.Match.Region)
			assert.Containsf(t, served, ov.Match.Region, "%s priced region %q is not a served location", id, ov.Match.Region)
			assert.NotEqualf(t, vertex.LocationGlobal, ov.Match.Region, "%s override region must be non-global", id)
		}

		// The reverse guard: every served non-global region must be priced,
		// so a matrix row added without an override goes red instead of
		// silently billing the global rate.
		assertEveryNonGlobalRegionPriced(t, served, info.Overrides)
	}
}

// TestEveryOfferingHasLocations guards entries() against servedLocations.
// The two are independent literals: an offering added to entries() but
// missing from servedLocations would silently price global-only and read
// unavailable at every location, with no build or existing test failing.
// Every offering must have a non-empty location set that includes global.
func TestEveryOfferingHasLocations(t *testing.T) {
	t.Parallel()

	for _, o := range vertex.Catalog().All() {
		locs := vertex.LocationsForModel(o.ID)
		require.NotEmptyf(t, locs, "%s has no servedLocations row", o.ID)
		assert.Containsf(t, locs, vertex.LocationGlobal, "%s must be served at global", o.ID)
	}
}

// flatPriced is every offering Google prices the same at every location it
// serves.
var flatPriced = map[string]bool{
	vertex.ModelGemini25Pro:         true,
	vertex.ModelGemini25Flash:       true,
	vertex.ModelGemini25FlashLite:   true,
	vertex.ModelGemini31ProPreview:  true,
	vertex.ModelGemini3FlashPreview: true,
}

// TestGlobalToNonGlobalRatio pins the single invariant the whole Vertex
// pricing rests on: every non-global rate is exactly global x 1.10, on every
// column, for every offering. Google's Agent Platform page publishes the
// premium as a flat 10% markup (read from the region tabs on 2026-09-08), so
// a future rate edit that breaks the ratio - a fat-fingered override, a
// global rate changed without its non-global sibling - is a transcription
// bug this test must catch. It mirrors bedrock's TestGeoGlobalRatio.
//
// The check is the exact-integer form 11*global == 10*geo, which avoids
// float rounding in the int64 micro-cent values pricing.NewRates produces.
// Columns a model does not carry (Gemini has no cache-creation rate) are
// zero on both sides and pass trivially.
func TestGlobalToNonGlobalRatio(t *testing.T) {
	t.Parallel()

	for id, info := range vertex.Catalog().PricingByID() {
		global := info.Default.Base

		if flatPriced[id] {
			continue
		}

		require.NotEmptyf(t, info.Overrides, "%s carries no non-global override to compare", id)

		for _, ov := range info.Overrides {
			geo := ov.RateCard.Base

			t.Run(id+"/"+ov.Match.Region, func(t *testing.T) {
				t.Parallel()

				check := func(col string, globalVal, geoVal int64) {
					assert.Equalf(t, 11*globalVal, 10*geoVal,
						"%s/%s %s: non-global (%d) must be exactly 1.10x global (%d)",
						id, ov.Match.Region, col, geoVal, globalVal)
				}

				check("input", global.InputPerMillion, geo.InputPerMillion)
				check("output", global.OutputPerMillion, geo.OutputPerMillion)
				check("cache read", global.CachedInputPerMillion, geo.CachedInputPerMillion)
				check("cache 5m write", global.CacheCreation5mPerMillion, geo.CacheCreation5mPerMillion)
				check("cache 1h write", global.CacheCreation1hPerMillion, geo.CacheCreation1hPerMillion)
			})
		}
	}
}

func TestFlatPricedModelsCarryNoOverride(t *testing.T) {
	t.Parallel()

	prices := vertex.Catalog().PricingByID()
	for id := range flatPriced {
		info, ok := prices[id]
		require.Truef(t, ok, "no pricing for %s", id)
		assert.Emptyf(t, info.Overrides, "%s is priced flat and must carry no override", id)
	}
}

func TestEveryServedRegionPriced(t *testing.T) {
	t.Parallel()

	for id, info := range vertex.Catalog().PricingByID() {
		if flatPriced[id] {
			continue
		}

		assertEveryNonGlobalRegionPriced(t, vertex.LocationsForModel(id), info.Overrides)
	}
}

// TestSonnet45LongContextBracket checks the one Claude model Vertex bills
// higher above 200K input tokens.
func TestSonnet45LongContextBracket(t *testing.T) {
	t.Parallel()

	info := vertex.Catalog().PricingByID()[vertex.ModelClaudeSonnet45]

	wantGlobal := pricing.Bracket{
		MinContextTokens: 200_001,
		Rates:            pricing.NewRates(6.00, 22.50, 0.60).WithCacheCreation(7.50, 12.00, 0),
	}
	wantRegional := pricing.Bracket{
		MinContextTokens: 200_001,
		Rates:            pricing.NewRates(6.60, 24.75, 0.66).WithCacheCreation(8.25, 13.20, 0),
	}

	assert.Equal(t, []pricing.Bracket{wantGlobal}, info.Default.Brackets)

	require.NotEmpty(t, info.Overrides)

	for _, ov := range info.Overrides {
		assert.Equalf(t, []pricing.Bracket{wantRegional}, ov.RateCard.Brackets, "region %q bracket", ov.Match.Region)
	}
}

func TestLifecycle(t *testing.T) {
	t.Parallel()

	wantRetires := map[string]string{
		vertex.ModelGemini25FlashLite: "2026-10-20",
		vertex.ModelGemini25Flash:     "2026-10-20",
		vertex.ModelGemini25Pro:       "2026-10-20",
	}
	wantPreview := map[string]bool{
		vertex.ModelGemini31ProPreview:  true,
		vertex.ModelGemini3FlashPreview: true,
	}

	for _, o := range vertex.Catalog().All() {
		if want, ok := wantRetires[o.ID]; ok {
			assert.Equalf(t, want, o.Life.Retires.Format("2006-01-02"), "%s Retires", o.ID)
			assert.NotEmptyf(t, o.Life.ReplacedBy, "%s retires, so it must name a replacement", o.ID)
		} else {
			assert.Truef(t, o.Life.Retires.IsZero(), "%s has no announced Vertex shutdown date", o.ID)
		}

		wantStage := catalog.StageGA
		if wantPreview[o.ID] {
			wantStage = catalog.StagePreview
		}

		assert.Equalf(t, wantStage, o.Life.Stage, "%s stage", o.ID)
	}
}

// TestManualThinkingBudget covers Claude 4.5 and 4.6, which take
// budget_tokens, and Claude 4.7 and later, which reject thinking.type enabled
// with a 400.
func TestManualThinkingBudget(t *testing.T) {
	t.Parallel()

	wantBudget := map[string]bool{
		vertex.ModelClaudeFable51:  false,
		vertex.ModelClaudeFable5:   false,
		vertex.ModelClaudeOpus5:    false,
		vertex.ModelClaudeOpus48:   false,
		vertex.ModelClaudeOpus47:   false,
		vertex.ModelClaudeOpus46:   true,
		vertex.ModelClaudeOpus45:   true,
		vertex.ModelClaudeSonnet46: true,
		vertex.ModelClaudeSonnet45: true,
		vertex.ModelClaudeHaiku55:  false,
		vertex.ModelClaudeHaiku45:  true,
	}

	for id, want := range wantBudget {
		o, ok := vertex.Catalog().Lookup(id)
		require.Truef(t, ok, "%s missing", id)
		assert.Equalf(t, want, o.Reasoning.Budget, "%s Reasoning.Budget", id)
		assert.Equalf(t, want, slices.Contains(o.Constraints.SupportedParams, "thinking_budget"), "%s thinking_budget param", id)
	}
}

// TestGeminiPenaltyParams covers Gemini 3.6 Flash and later, and 3.5
// Flash-Lite, which return an error on a custom penalty value.
func TestGeminiPenaltyParams(t *testing.T) {
	t.Parallel()

	noPenalty := map[string]bool{
		vertex.ModelGemini38Flash:     true,
		vertex.ModelGemini37Flash:     true,
		vertex.ModelGemini36Flash:     true,
		vertex.ModelGemini35FlashLite: true,
	}

	for _, o := range vertex.Catalog().All() {
		if !strings.HasPrefix(o.ID, "gemini-") {
			continue
		}

		for _, p := range []string{"presence_penalty", "frequency_penalty"} {
			assert.Equalf(t, !noPenalty[o.ID], slices.Contains(o.Constraints.SupportedParams, p), "%s %s param", o.ID, p)
		}
	}
}

// TestExtendedThinkingOnly covers Claude models on which Anthropic's thinking
// table rejects thinking.type adaptive with a 400.
func TestExtendedThinkingOnly(t *testing.T) {
	t.Parallel()

	for _, id := range []string{vertex.ModelClaudeSonnet45, vertex.ModelClaudeHaiku45} {
		o, ok := vertex.Catalog().Lookup(id)
		require.Truef(t, ok, "%s missing", id)
		assert.Falsef(t, o.Reasoning.Adaptive, "%s Reasoning.Adaptive", id)
		assert.Emptyf(t, o.Reasoning.Efforts, "%s Reasoning.Efforts", id)
		assert.NotContainsf(t, o.Constraints.SupportedParams, "reasoning_effort", "%s reasoning_effort param", id)
	}
}

// TestBareClaudeIDAliases covers Claude IDs that Vertex serves in bare and
// @-dated form as the same version.
func TestBareClaudeIDAliases(t *testing.T) {
	t.Parallel()

	cat := vertex.Catalog()
	prices := cat.PricingByID()

	for bare, dated := range map[string]string{
		"claude-opus-4-5":   vertex.ModelClaudeOpus45,
		"claude-sonnet-4-5": vertex.ModelClaudeSonnet45,
	} {
		id, ok := cat.ResolveID(bare)
		require.Truef(t, ok, "%s does not resolve", bare)
		assert.Equalf(t, dated, id, "%s resolves to", bare)

		info, ok := prices[bare]
		require.Truef(t, ok, "%s is unpriced", bare)
		assert.Equalf(t, prices[dated], info, "%s pricing", bare)
	}
}

func TestClaudeToolSearchMirrorsAnthropic(t *testing.T) {
	t.Parallel()

	for _, o := range vertex.Catalog().All() {
		if !strings.HasPrefix(o.ID, "claude-") {
			continue
		}

		bare, _, _ := strings.Cut(o.ID, "@")
		direct, ok := anthropic.Catalog().Lookup(bare)
		require.Truef(t, ok, "%s has no Anthropic-direct entry", o.ID)
		assert.Equalf(t, direct.Capabilities.ToolSearch, o.Capabilities.ToolSearch, "%s ToolSearch", o.ID)
	}
}
