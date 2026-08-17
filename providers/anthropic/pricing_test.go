package anthropic

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// TestAllModelsHavePricing pins the Anthropic-specific cache-rate shape.
// Core input/output pricing presence is enforced structurally by
// catalog.New, which Catalog() would panic on.
func TestAllModelsHavePricing(t *testing.T) {
	t.Parallel()

	for _, o := range Catalog().All() {
		t.Run(o.ID, func(t *testing.T) {
			t.Parallel()

			assert.Positive(t, o.Pricing.Default.Base.InputPerMillion,
				"model %s missing input pricing", o.ID)
			assert.Positive(t, o.Pricing.Default.Base.OutputPerMillion,
				"model %s missing output pricing", o.ID)
			assert.Positive(t, o.Pricing.Default.Base.CachedInputPerMillion,
				"model %s missing cached pricing", o.ID)
			assert.Positive(t, o.Pricing.Default.Base.CacheCreation5mPerMillion,
				"model %s missing 5m cache write pricing", o.ID)
			assert.Positive(t, o.Pricing.Default.Base.CacheCreation1hPerMillion,
				"model %s missing 1h cache write pricing", o.ID)
		})
	}
}

func TestFastModeModelsHaveFastPricing(t *testing.T) {
	t.Parallel()

	for _, o := range Catalog().All() {
		if !slices.Contains(o.Speeds, SpeedFast) {
			continue
		}

		t.Run(o.ID, func(t *testing.T) {
			t.Parallel()

			fast, found := findFastOverride(o.Pricing.Overrides)
			require.True(t, found, "model %s supports fast speed but has no fast override", o.ID)
			assert.Positive(t, fast.Base.InputPerMillion,
				"model %s supports fast speed but missing fast input rate", o.ID)
			assert.Positive(t, fast.Base.OutputPerMillion,
				"model %s supports fast speed but missing fast output rate", o.ID)
			assert.Positive(t, fast.Base.CachedInputPerMillion,
				"model %s supports fast speed but missing fast cached-input rate", o.ID)
			assert.Positive(t, fast.Base.CacheCreation5mPerMillion,
				"model %s supports fast speed but missing fast 5m cache-write rate", o.ID)
			assert.Positive(t, fast.Base.CacheCreation1hPerMillion,
				"model %s supports fast speed but missing fast 1h cache-write rate", o.ID)
		})
	}
}

func TestClaudeOpus5Pricing(t *testing.T) {
	t.Parallel()

	o, ok := Catalog().Lookup(ModelClaudeOpus5)
	require.True(t, ok)

	assert.Equal(t,
		pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		o.Pricing.Default.Base,
	)
	assert.Empty(t, o.Pricing.Default.Brackets)

	fast, found := findFastOverride(o.Pricing.Overrides)
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
	assert.Len(t, pricingMap, Catalog().Len(),
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
