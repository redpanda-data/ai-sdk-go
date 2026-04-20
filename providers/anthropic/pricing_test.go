package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
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
		})
	}
}

func TestModelPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	assert.Len(t, pricingMap, len(supportedModels),
		"ModelPricing should return exactly one entry per supported model")
}
