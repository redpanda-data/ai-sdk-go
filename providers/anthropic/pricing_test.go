package anthropic

import (
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

func TestModelPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	assert.Len(t, pricingMap, len(supportedModels),
		"ModelPricing should return exactly one entry per supported model")
}
