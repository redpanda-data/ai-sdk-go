package anthropic

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAllModelsHavePricing(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			assert.Positive(t, def.Pricing.InputPerMillion,
				"model %s missing input pricing — add Pricing to its ModelDefinition", id)
			assert.Positive(t, def.Pricing.OutputPerMillion,
				"model %s missing output pricing — add Pricing to its ModelDefinition", id)
			assert.Positive(t, def.Pricing.CachedInputPerMillion,
				"model %s missing cached pricing — add CachedInputPerMillion to its ModelDefinition", id)
			require.NotNil(t, def.Pricing.Anthropic,
				"model %s missing Anthropic pricing — add Anthropic sub-struct to its ModelDefinition", id)
			assert.Positive(t, def.Pricing.Anthropic.CacheWrite5mPerMillion,
				"model %s missing 5m cache write pricing", id)
			assert.Positive(t, def.Pricing.Anthropic.CacheWrite1hPerMillion,
				"model %s missing 1h cache write pricing", id)
		})
	}
}

func TestFastModeModelsHaveFastPricing(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		if !slices.Contains(def.SupportedSpeeds, SpeedFast) {
			continue
		}

		t.Run(id, func(t *testing.T) {
			t.Parallel()

			require.NotNil(t, def.Pricing.Anthropic,
				"model %s supports fast speed but has no Anthropic pricing", id)
			assert.Positive(t, def.Pricing.Anthropic.FastInputPerMillion,
				"model %s supports fast speed but missing FastInputPerMillion", id)
			assert.Positive(t, def.Pricing.Anthropic.FastOutputPerMillion,
				"model %s supports fast speed but missing FastOutputPerMillion", id)
			assert.Positive(t, def.Pricing.Anthropic.FastCachedInputPerMillion,
				"model %s supports fast speed but missing FastCachedInputPerMillion", id)
			assert.Positive(t, def.Pricing.Anthropic.FastCacheWrite5mPerMillion,
				"model %s supports fast speed but missing FastCacheWrite5mPerMillion", id)
			assert.Positive(t, def.Pricing.Anthropic.FastCacheWrite1hPerMillion,
				"model %s supports fast speed but missing FastCacheWrite1hPerMillion", id)
		})
	}
}

func TestModelPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	assert.Len(t, pricingMap, len(supportedModels),
		"ModelPricing should return exactly one entry per supported model")
}
