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
			// Cache-write: every model must carry a positive rate, EXCEPT models
			// AWS documents as charging nothing to populate the cache (their
			// cache-write usagetype is $0.00) — Amazon Nova 2 Lite. Those must be
			// exactly zero. Guarding both directions keeps the "forgot
			// WithCacheCreation" check intact for Claude while allowing the
			// documented free-cache-write models.
			if freeCacheWriteModels[id] {
				assert.Zero(t, def.Pricing.Default.Base.CacheCreation5mPerMillion,
					"model %s is documented free-cache-write but has a 5m rate", id)
				assert.Zero(t, def.Pricing.Default.Base.CacheCreation1hPerMillion,
					"model %s is documented free-cache-write but has a 1h rate", id)
			} else {
				assert.Positive(t, def.Pricing.Default.Base.CacheCreation5mPerMillion,
					"model %s missing 5m cache write pricing", id)
				assert.Positive(t, def.Pricing.Default.Base.CacheCreation1hPerMillion,
					"model %s missing 1h cache write pricing", id)
			}
		})
	}
}

func TestModelPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	assert.Len(t, pricingMap, len(supportedModels),
		"ModelPricing should return exactly one entry per supported model")
}

// TestGeoGlobalRatio pins, per logical model, the relationship between the
// catalog's global. variant and any of its non-global siblings (bare /
// us. / eu. / au. / jp.): geo == 1.10 * global, exactly, in every priced
// column. AWS publishes the Global tier at a 10% discount to the
// Geo/In-region tier; an earlier version of this catalog inverted the
// relationship (cf. revert a7f0410), so we encode the direction here to
// fail loud on any future drift.
//
// The check walks supportedModels rather than referencing shared rate
// constants, because the catalog deliberately spells each rate out per
// entry — Anthropic's intermediate releases (e.g. Opus 4.1) have priced
// differently in the past, so there is no per-family rate variable.
func TestGeoGlobalRatio(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		if !strings.HasPrefix(id, "global.") {
			continue
		}

		bare := strings.TrimPrefix(id, "global.")

		// Pick any non-global sibling to compare against. Prefer the bare
		// entry when it exists, otherwise fall back to the us. profile
		// (every model in the catalog has at least one geo profile).
		sibling, ok := supportedModels[bare]
		if !ok {
			sibling, ok = supportedModels["us."+bare]
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
