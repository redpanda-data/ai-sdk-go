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
func TestPricingStaysFlatAcrossContextWindow(t *testing.T) {
	t.Parallel()

	for _, def := range Catalog().All() {
		id := def.ID

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

	cat, err := pricing.NewCatalog(pricing.WithProvider("anthropic", Catalog().PricingByID()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 100_000, OutputTokens: 1_000}

	below, err := cat.Calculate(ModelClaudeSonnet5, usage, pricing.CalcRequest{ContextTokens: 100_000})
	require.NoError(t, err)

	above, err := cat.Calculate(ModelClaudeSonnet5, usage, pricing.CalcRequest{ContextTokens: 900_000})
	require.NoError(t, err)

	assert.Zero(t, below.AppliedBracketMinContextTokens)
	assert.Zero(t, above.AppliedBracketMinContextTokens,
		"a 900K request must still price on the base card")

	assert.Equal(t, below.Breakdown[pricing.UsageFieldInput], above.Breakdown[pricing.UsageFieldInput],
		"input must cost the same per token across the full 1M window")
	assert.Equal(t, below.Total, above.Total,
		"identical usage must cost the same regardless of context size")
}
