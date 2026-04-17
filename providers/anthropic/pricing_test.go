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
		})
	}
}

func TestDefaultPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingList := DefaultPricing()
	assert.Len(t, pricingList, len(supportedModels),
		"DefaultPricing should return exactly one entry per supported model")
}
