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

package bedrock

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/pricing"
)

func TestAllModelsHavePricing(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			assert.Positive(t, def.Pricing.Default.Base.InputPerMillion,
				"model %s missing input pricing — add Pricing to its ModelDefinition", id)
			assert.Positive(t, def.Pricing.Default.Base.OutputPerMillion,
				"model %s missing output pricing — add Pricing to its ModelDefinition", id)
			assert.Positive(t, def.Pricing.Default.Base.CachedInputPerMillion,
				"model %s missing cached pricing — add CachedInputPerMillion to its ModelDefinition", id)
			assert.Positive(t, def.Pricing.Default.Base.CacheCreation5mPerMillion,
				"model %s missing 5m cache write pricing", id)
			assert.Positive(t, def.Pricing.Default.Base.CacheCreation1hPerMillion,
				"model %s missing 1h cache write pricing", id)
		})
	}
}

func TestModelPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	assert.Len(t, pricingMap, len(supportedModels),
		"ModelPricing should return exactly one entry per supported model")
}

// TestGeoGlobalRatio pins the relationship between the Geo and Global rate
// cards at exactly 1.10x per column. AWS publishes the Global tier at a
// 10% discount to the Geo/In-region tier; an earlier version of this catalog
// inverted the relationship (cf. revert a7f0410), so we encode the direction
// here to fail loud on any future drift.
func TestGeoGlobalRatio(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		geo, global pricing.Rates
	}{
		{"Opus 4.x", claudeOpus4xGeo, claudeOpus4xGlobal},
		{"Sonnet 4.x", claudeSonnet4xGeo, claudeSonnet4xGlobal},
		{"Haiku 4.5", claudeHaiku45Geo, claudeHaiku45Global},
	}

	// 10*geo == 11*global is the exact integer form of geo = 1.10 * global
	// in the int64-microcent representation pricing.NewRates produces.
	check := func(t *testing.T, col string, geo, global int64) {
		t.Helper()
		assert.Equal(t, 11*global, 10*geo, "%s: geo should be exactly 1.10x global", col)
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			check(t, "input", tc.geo.InputPerMillion, tc.global.InputPerMillion)
			check(t, "output", tc.geo.OutputPerMillion, tc.global.OutputPerMillion)
			check(t, "cache read", tc.geo.CachedInputPerMillion, tc.global.CachedInputPerMillion)
			check(t, "cache 5m write", tc.geo.CacheCreation5mPerMillion, tc.global.CacheCreation5mPerMillion)
			check(t, "cache 1h write", tc.geo.CacheCreation1hPerMillion, tc.global.CacheCreation1hPerMillion)
		})
	}
}
