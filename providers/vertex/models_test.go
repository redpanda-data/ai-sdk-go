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
// not a separate model entry. The override set must cover exactly the
// non-global locations the model is served at, so pricing and
// availability never disagree.
func TestGeminiRegionalOverride(t *testing.T) {
	t.Parallel()

	info, ok := vertex.Catalog().PricingByID()[offeringGemini36Flash]
	require.True(t, ok, "no pricing for %s", offeringGemini36Flash)

	assert.Equal(t, pricing.NewRates(1.50, 7.50, 0.15), info.Default.Base, "global default rate")

	wantRegional := pricing.NewRates(1.65, 8.25, 0.165)

	// Every non-global served location gets one override at the regional
	// rate, and nothing else does.
	var wantRegions []string

	for _, loc := range vertex.LocationsForModel(vertex.ModelGemini36Flash) {
		if loc != vertex.LocationGlobal {
			wantRegions = append(wantRegions, loc)
		}
	}

	require.NotEmpty(t, wantRegions, "expected at least one non-global Gemini location to override")

	gotRegions := make([]string, 0, len(info.Overrides))
	for _, ov := range info.Overrides {
		require.NotEmptyf(t, ov.Match.Region, "override has empty Region (would also match global): %+v", ov.Match)
		assert.Equalf(t, wantRegional, ov.RateCard.Base, "region %q rate", ov.Match.Region)
		gotRegions = append(gotRegions, ov.Match.Region)
	}

	assert.ElementsMatch(t, wantRegions, gotRegions, "override regions must equal the non-global served locations")
}

// TestClaudeFlatPricing checks the Claude rates and that Claude carries no
// regional premium: the non-global markup is a Gemini-only fact.
func TestClaudeFlatPricing(t *testing.T) {
	t.Parallel()

	cases := map[string]pricing.Rates{
		offeringClaudeSonnet5: pricing.NewRates(2.00, 10.00, 0.20).WithCacheCreation(2.50, 4.00, 0),
		offeringClaudeHaiku45: pricing.NewRates(1.00, 5.00, 0.10).WithCacheCreation(1.25, 2.00, 0),
	}

	prices := vertex.Catalog().PricingByID()
	for id, want := range cases {
		info, ok := prices[id]
		require.Truef(t, ok, "no pricing for %s", id)
		assert.Equalf(t, want, info.Default.Base, "%s base rate", id)
		assert.Emptyf(t, info.Overrides, "%s must have no region overrides (Claude has no regional premium)", id)
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
