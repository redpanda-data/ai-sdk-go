package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAllModelsHavePricing(t *testing.T) {
	for id, def := range supportedModels {
		t.Run(id, func(t *testing.T) {
			assert.Greater(t, def.Pricing.InputPerMillion, int64(0),
				"model %s missing input pricing — add Pricing to its ModelDefinition", id)
			assert.Greater(t, def.Pricing.OutputPerMillion, int64(0),
				"model %s missing output pricing — add Pricing to its ModelDefinition", id)
		})
	}
}

func TestDefaultPricingMatchesModels(t *testing.T) {
	pricingList := DefaultPricing()
	assert.Equal(t, len(supportedModels), len(pricingList),
		"DefaultPricing should return exactly one entry per supported model")
}
