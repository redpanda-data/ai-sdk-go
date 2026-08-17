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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// freeCacheWriteModels lists Bedrock models AWS documents as charging nothing
// to populate the cache (their cache-write usagetype is priced at $0.00). For
// these, cache-write is expected to be exactly zero; every other model must
// carry a positive cache-write rate.
var freeCacheWriteModels = map[string]bool{
	ModelNova2LiteGlobal: true,
	ModelNova2LiteUS:     true,
	ModelNova2LiteEU:     true,
	ModelNova2LiteJP:     true,
}

// noCacheModels lists Bedrock models that do not support prompt caching at all
// (AWS bills them with no cache-read or cache-write usagetype). For these,
// every cache rate — read AND write — is expected to be exactly zero, which is
// the documented shape for a non-caching provider (see pricing.Rates). This is
// distinct from freeCacheWriteModels, which DO bill cache reads but populate
// the cache for free.
var noCacheModels = map[string]bool{
	ModelMistralLarge3: true,
	ModelGemma431B:     true,
	ModelGemma426BA4B:  true,
	ModelGemma4E2B:     true,
}

// unknownTTLCacheModels report aggregate cache-write tokens without a
// TTL-specific usage bucket. Their write rate must therefore live only in
// CacheCreationUnknownTTLPerMillion.
var unknownTTLCacheModels = map[string]bool{
	ModelGPT56Sol:   true,
	ModelGPT56Terra: true,
	ModelGPT56Luna:  true,
}

func TestAllModelsHavePricing(t *testing.T) {
	t.Parallel()

	for _, def := range Catalog().All() {
		id := def.ID
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			assert.Positive(t, def.Pricing.Default.Base.InputPerMillion,
				"model %s missing input pricing", id)
			assert.Positive(t, def.Pricing.Default.Base.OutputPerMillion,
				"model %s missing output pricing", id)

			base := def.Pricing.Default.Base

			switch {
			case noCacheModels[id]:
				// No prompt caching at all: every cache rate is zero. This is
				// the documented shape for a non-caching provider (Mistral
				// Large 3 — AWS bills no cache usagetype for it).
				assert.Zero(t, base.CachedInputPerMillion,
					"model %s is documented no-cache but has a cache-read rate", id)
				assert.Zero(t, base.CacheCreation5mPerMillion,
					"model %s is documented no-cache but has a 5m cache-write rate", id)
				assert.Zero(t, base.CacheCreation1hPerMillion,
					"model %s is documented no-cache but has a 1h cache-write rate", id)
			case unknownTTLCacheModels[id]:
				assert.Positive(t, base.CachedInputPerMillion,
					"model %s missing cached pricing", id)
				assert.Zero(t, base.CacheCreation5mPerMillion,
					"model %s reports aggregate cache writes but has a 5m rate", id)
				assert.Zero(t, base.CacheCreation1hPerMillion,
					"model %s reports aggregate cache writes but has a 1h rate", id)
				assert.Positive(t, base.CacheCreationUnknownTTLPerMillion,
					"model %s missing aggregate cache-write pricing", id)
			case freeCacheWriteModels[id]:
				// Cache reads are billed, but populating the cache is free —
				// their cache-write usagetype is $0.00 (Amazon Nova 2 Lite).
				assert.Positive(t, base.CachedInputPerMillion,
					"model %s missing cached pricing — add CachedInputPerMillion to its ModelDefinition", id)
				assert.Zero(t, base.CacheCreation5mPerMillion,
					"model %s is documented free-cache-write but has a 5m rate", id)
				assert.Zero(t, base.CacheCreation1hPerMillion,
					"model %s is documented free-cache-write but has a 1h rate", id)
			default:
				// Full caching (Claude): read + write rates all positive. This
				// keeps the "forgot WithCacheCreation" check intact.
				assert.Positive(t, base.CachedInputPerMillion,
					"model %s missing cached pricing — add CachedInputPerMillion to its ModelDefinition", id)
				assert.Positive(t, base.CacheCreation5mPerMillion,
					"model %s missing 5m cache write pricing", id)
				assert.Positive(t, base.CacheCreation1hPerMillion,
					"model %s missing 1h cache write pricing", id)
			}
		})
	}
}

func TestGPT56Pricing(t *testing.T) {
	t.Parallel()

	tests := []struct {
		modelID string
		rates   pricing.Rates
	}{
		{
			modelID: ModelGPT56Sol,
			rates:   pricing.NewRates(5.50, 33.00, 0.55).WithCacheCreation(0, 0, 6.875),
		},
		{
			modelID: ModelGPT56Terra,
			rates:   pricing.NewRates(2.75, 16.50, 0.275).WithCacheCreation(0, 0, 3.4375),
		},
		{
			modelID: ModelGPT56Luna,
			rates:   pricing.NewRates(1.10, 6.60, 0.11).WithCacheCreation(0, 0, 1.375),
		},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			t.Parallel()

			def, ok := Catalog().Lookup(tt.modelID)
			require.True(t, ok)
			assert.Equal(t, tt.rates, def.Pricing.Default.Base)
		})
	}
}

func TestClaudeOpus5Pricing(t *testing.T) {
	t.Parallel()

	global, globalOK := Catalog().Lookup(ModelClaudeOpus5Global)
	require.True(t, globalOK)
	assert.Equal(t,
		pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		global.Pricing.Default.Base,
	)

	geoRates := pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0)

	for _, id := range []string{
		ModelClaudeOpus5US,
		ModelClaudeOpus5EU,
		ModelClaudeOpus5AU,
	} {
		def, ok := Catalog().Lookup(id)
		require.True(t, ok)
		assert.Equal(t, geoRates, def.Pricing.Default.Base)
	}
}

// TestGeoGlobalRatio pins, per logical model, the relationship between the
// catalog's global. variant and any of its non-global siblings (bare /
// us. / eu. / au. / jp.): geo == 1.10 * global, exactly, in every priced
// column. AWS publishes the Global tier at a 10% discount to the
// Geo/In-region tier; an earlier version of this catalog inverted the
// relationship (cf. revert a7f0410), so we encode the direction here to
// fail loud on any future drift.
//
// The check walks the catalog rather than the family declarations, so it
// guards the expander's output: authored rates are per family, and this
// tripwire fails loud if AWS ever breaks the 1.10x relationship for one
// model (making it a data edit) or if the expander mispairs rate cards.
func TestGeoGlobalRatio(t *testing.T) {
	t.Parallel()

	for _, def := range Catalog().All() {
		id := def.ID
		if !strings.HasPrefix(id, "global.") {
			continue
		}

		bare := strings.TrimPrefix(id, "global.")

		// Pick any non-global sibling to compare against. Prefer the bare
		// entry when it exists, otherwise fall back to the us. profile
		// (every model in the catalog has at least one geo profile).
		sibling, ok := Catalog().Lookup(bare)
		if !ok {
			sibling, ok = Catalog().Lookup("us." + bare)
		}

		require.True(t, ok, "global. variant %s has no non-global sibling to compare against", id)

		t.Run(id, func(t *testing.T) {
			t.Parallel()

			gl := def.Pricing.Default.Base
			geo := sibling.Pricing.Default.Base

			// 10*geo == 11*global is the exact integer form of
			// geo = 1.10 * global in the int64-microcent form
			// pricing.NewRates produces.
			check := func(col string, geoVal, globalVal int64) {
				assert.Equal(t, 11*globalVal, 10*geoVal,
					"%s/%s: geo (%d) should be exactly 1.10x global (%d)",
					id, col, geoVal, globalVal)
			}

			check("input", geo.InputPerMillion, gl.InputPerMillion)
			check("output", geo.OutputPerMillion, gl.OutputPerMillion)
			check("cache read", geo.CachedInputPerMillion, gl.CachedInputPerMillion)
			check("cache 5m write", geo.CacheCreation5mPerMillion, gl.CacheCreation5mPerMillion)
			check("cache 1h write", geo.CacheCreation1hPerMillion, gl.CacheCreation1hPerMillion)
		})
	}
}
