package anthropic

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// TestPricingStaysFlatAcrossContextWindow guards Anthropic's documented
// rule: every model with a 1M context window includes the full window at
// standard pricing, and 200K models never had a context tier either. So no
// entry — default card or speed override — may carry a context bracket. A
// bracket here would overcharge large requests by up to 2x.
//
// Claude 4 and 4.5 once charged a >200K surcharge (input 2x, output 1.5x);
// that tier no longer exists on any catalogued model.
//
// Claude Haiku 5.5 is the documented exception ("Claude 4.6 and later models
// (except Claude Haiku 5.5)"): it is priced by prompt length, so
// TestHaiku55PricesByPromptLength covers it instead.
func TestPricingStaysFlatAcrossContextWindow(t *testing.T) {
	t.Parallel()

	for _, def := range Catalog().All() {
		id := def.ID
		if id == ModelClaudeHaiku55 {
			continue
		}

		t.Run(id, func(t *testing.T) {
			t.Parallel()

			assert.Emptyf(t, def.Pricing.Default.Brackets,
				"%s must not have context brackets: long context bills at standard rates", id)

			for _, ov := range def.Pricing.Overrides {
				assert.Emptyf(t, ov.RateCard.Brackets,
					"%s override %s must not have context brackets", id, fmt.Sprintf("%+v", ov.Match))
			}
		})
	}
}

// TestLongContextCostsTheSamePerToken drives a real Catalog and proves the
// flat rule end to end: identical usage costs the same whether the request
// context sits below or above 200K, and no bracket is reported.
func TestLongContextCostsTheSamePerToken(t *testing.T) {
	t.Parallel()

	cat, err := pricing.NewCatalog(pricing.WithSource(Catalog()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 100_000, OutputTokens: 1_000}

	below, err := cat.Calculate(ProviderName, ModelClaudeSonnet5, usage, pricing.CalcRequest{ContextTokens: 100_000})
	require.NoError(t, err)

	above, err := cat.Calculate(ProviderName, ModelClaudeSonnet5, usage, pricing.CalcRequest{ContextTokens: 900_000})
	require.NoError(t, err)

	assert.Zero(t, below.AppliedBracketMinContextTokens)
	assert.Zero(t, above.AppliedBracketMinContextTokens,
		"a 900K request must still price on the base card")

	assert.Equal(t, below.Breakdown[pricing.UsageFieldInput], above.Breakdown[pricing.UsageFieldInput],
		"input must cost the same per token across the full 1M window")
	assert.Equal(t, below.Total, above.Total,
		"identical usage must cost the same regardless of context size")
}

// TestHaiku55PricesByPromptLength pins Haiku 5.5's prompt-length tier: a
// prompt of up to 100,000 tokens bills at the base rates and anything over
// it at the higher row, for every rate including cache.
func TestHaiku55PricesByPromptLength(t *testing.T) {
	t.Parallel()

	cat, err := pricing.NewCatalog(pricing.WithSource(Catalog()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 1_000_000, OutputTokens: 1_000_000, CachedInputTokens: 1_000_000}

	for _, tt := range []struct {
		name                  string
		context               int64
		bracket               int64
		input, output, cached int64
	}{
		{"at 100,000 tokens", 100_000, 0, 10_000_000, 50_000_000, 1_000_000},
		{"over 100,000 tokens", 100_001, 100_001, 50_000_000, 250_000_000, 5_000_000},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cost, err := cat.Calculate(ProviderName, ModelClaudeHaiku55, usage, pricing.CalcRequest{ContextTokens: tt.context})
			require.NoError(t, err)
			assert.Equal(t, tt.bracket, cost.AppliedBracketMinContextTokens)
			assert.Equal(t, tt.input, cost.Breakdown[pricing.UsageFieldInput])
			assert.Equal(t, tt.output, cost.Breakdown[pricing.UsageFieldOutput])
			assert.Equal(t, tt.cached, cost.Breakdown[pricing.UsageFieldCachedInput])
		})
	}
}
