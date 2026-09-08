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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

const (
	offeringGemini36Flash = "vertex.gemini-3.6-flash"
	offeringClaudeSonnet5 = "vertex.claude-sonnet-5"
	offeringClaudeHaiku45 = "vertex.claude-haiku-4-5"
)

// TestCatalogBuildsWithDayOneModels pins the day-one catalog to exactly
// the Gemini + Claude scope: three offerings, each namespaced vertex.*.
func TestCatalogBuildsWithDayOneModels(t *testing.T) {
	t.Parallel()

	cat := vertex.Catalog()
	require.NotNil(t, cat)

	got := make([]string, 0, cat.Len())
	for _, o := range cat.All() {
		got = append(got, o.ID)
	}

	assert.ElementsMatch(t, []string{offeringGemini36Flash, offeringClaudeSonnet5, offeringClaudeHaiku45}, got)
}

// TestOfferingForModel checks the bridge that lets a caller holding a bare
// publisher model ID reach the namespaced offering, and that passing the
// already-namespaced ID resolves to the same entry. An unknown model
// returns ok false.
func TestOfferingForModel(t *testing.T) {
	t.Parallel()

	bare, ok := vertex.OfferingForModel(vertex.ModelClaudeSonnet5)
	require.True(t, ok, "bare model %q should resolve", vertex.ModelClaudeSonnet5)
	assert.Equal(t, offeringClaudeSonnet5, bare.ID)

	prefixed, ok := vertex.OfferingForModel(offeringClaudeSonnet5)
	require.True(t, ok, "namespaced id %q should resolve", offeringClaudeSonnet5)
	assert.Equal(t, offeringClaudeSonnet5, prefixed.ID)

	_, ok = vertex.OfferingForModel("gemini-99-ultra")
	assert.False(t, ok, "unknown model must not resolve")
}

// TestOfferingAttributes checks every offering carries the publisher and
// the bare wire model, and that the bare model is the offering ID minus
// the vertex. prefix. The runtime provider builds the request path from
// these, so a missing or drifted value is a routing bug.
func TestOfferingAttributes(t *testing.T) {
	t.Parallel()

	wantPublisher := map[string]string{
		offeringGemini36Flash: "google",
		offeringClaudeSonnet5: "anthropic",
		offeringClaudeHaiku45: "anthropic",
	}

	for _, o := range vertex.Catalog().All() {
		assert.Equalf(t, wantPublisher[o.ID], o.Attributes[vertex.AttrPublisher], "%s publisher", o.ID)
		assert.Equalf(t, strings.TrimPrefix(o.ID, "vertex."), o.Attributes[vertex.AttrVertexModel], "%s vertex_model", o.ID)
	}
}

// TestNoBarePricingKey is the collision guard. A merged pricing catalog
// rejects two providers registering the same model ID, so every Vertex
// pricing key must be namespaced and no bare publisher ID (or alias for
// one) may leak in.
func TestNoBarePricingKey(t *testing.T) {
	t.Parallel()

	for id := range vertex.Catalog().PricingByID() {
		assert.Truef(t, strings.HasPrefix(id, "vertex."), "pricing key %q is not namespaced with the vertex. prefix", id)
	}
}

// TestGeminiRegionalOverride checks the one interface-shaped requirement:
// the non-global Gemini rate is a Region override on the global default,
// not a separate model entry.
//
// Priced regions and served locations are separate facts, so this does not
// assert the two sets are equal. It asserts the guard direction that
// matters: every priced region must be a served location (a price at a
// location the model is not served would be dead), and at least one
// override must exist at the regional rate. A served location with no
// override simply falls back to the global default, which is intended.
func TestGeminiRegionalOverride(t *testing.T) {
	t.Parallel()

	info, ok := vertex.Catalog().PricingByID()[offeringGemini36Flash]
	require.True(t, ok, "no pricing for %s", offeringGemini36Flash)

	assert.Equal(t, pricing.NewRates(1.50, 7.50, 0.15), info.Default.Base, "global default rate")

	wantRegional := pricing.NewRates(1.65, 8.25, 0.165)

	served := vertex.LocationsForModel(vertex.ModelGemini36Flash)
	require.NotEmpty(t, served, "expected served locations for Gemini")

	require.NotEmpty(t, info.Overrides, "expected at least one non-global Gemini rate override")

	for _, ov := range info.Overrides {
		require.NotEmptyf(t, ov.Match.Region, "override has empty Region (would also match global): %+v", ov.Match)
		assert.Equalf(t, wantRegional, ov.RateCard.Base, "region %q rate", ov.Match.Region)
		assert.Containsf(t, served, ov.Match.Region, "priced region %q is not a served location", ov.Match.Region)
		assert.NotEqualf(t, vertex.LocationGlobal, ov.Match.Region, "override region must be non-global")
	}
}

// TestClaudeRegionalOverride checks the Claude rates and that Claude
// carries the same ~10% non-global premium Gemini does. Google's Agent
// Platform pricing page groups Sonnet 5 and Haiku 4.5 under "Models with
// regional pricing" and publishes every non-global rate at exactly global
// x 1.10 (read from the page's region tabs on 2026-09-08).
//
// Like TestGeminiRegionalOverride, it asserts the global default rate, the
// exact non-global rate, and the guard direction that matters: every priced
// region is a served location, and the override region is non-global. The
// two models differ in which regions carry the premium - Sonnet on the
// us/eu multi-regions, Haiku on us-east5/europe-west1 - so each is checked
// against its own served set.
func TestClaudeRegionalOverride(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		global, regional pricing.Rates
	}{
		offeringClaudeSonnet5: {
			global:   pricing.NewRates(2.00, 10.00, 0.20).WithCacheCreation(2.50, 4.00, 0),
			regional: pricing.NewRates(2.20, 11.00, 0.22).WithCacheCreation(2.75, 4.40, 0),
		},
		offeringClaudeHaiku45: {
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
		bare := o.Attributes[vertex.AttrVertexModel]
		locs := vertex.LocationsForModel(bare)
		require.NotEmptyf(t, locs, "%s has no servedLocations row for %q", o.ID, bare)
		assert.Containsf(t, locs, vertex.LocationGlobal, "%s must be served at global", o.ID)
	}
}
