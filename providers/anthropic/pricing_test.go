package anthropic

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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

func TestFastModeModelsHaveFastPricing(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		if !slices.Contains(def.SupportedSpeeds, SpeedFast) {
			continue
		}

		t.Run(id, func(t *testing.T) {
			t.Parallel()

			fast, found := findFastOverride(def.Pricing.Overrides)
			require.True(t, found, "model %s supports fast speed but has no fast override", id)
			assert.Positive(t, fast.Base.InputPerMillion,
				"model %s supports fast speed but missing FastInputPerMillion", id)
			assert.Positive(t, fast.Base.OutputPerMillion,
				"model %s supports fast speed but missing FastOutputPerMillion", id)
			assert.Positive(t, fast.Base.CachedInputPerMillion,
				"model %s supports fast speed but missing FastCachedInputPerMillion", id)
			assert.Positive(t, fast.Base.CacheCreation5mPerMillion,
				"model %s supports fast speed but missing FastCacheWrite5mPerMillion", id)
			assert.Positive(t, fast.Base.CacheCreation1hPerMillion,
				"model %s supports fast speed but missing FastCacheWrite1hPerMillion", id)
		})
	}
}

func TestClaudeOpus5Pricing(t *testing.T) {
	t.Parallel()

	def, ok := supportedModels[ModelClaudeOpus5]
	require.True(t, ok)

	assert.Equal(t,
		pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		def.Pricing.Default.Base,
	)
	assert.Empty(t, def.Pricing.Default.Brackets)

	fast, found := findFastOverride(def.Pricing.Overrides)
	require.True(t, found)
	assert.Equal(t,
		pricing.NewRates(10.00, 50.00, 1.00).WithCacheCreation(12.50, 20.00, 0),
		fast.Base,
	)
	assert.Empty(t, fast.Brackets)
}

func TestModelPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	assert.Len(t, pricingMap, len(supportedModels),
		"ModelPricing should return exactly one entry per supported model")
}

func findFastOverride(overrides []pricing.Override) (pricing.RateCard, bool) {
	fastMatch := pricing.Selector{Speed: SpeedFast}
	for _, override := range overrides {
		if override.Match == fastMatch {
			return override.RateCard, true
		}
	}

	return pricing.RateCard{}, false
}
