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

package anthropic

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// longContextThreshold is the inclusive lower bound of Anthropic's >200K
// long-context pricing tier, mirrored from the catalog entries.
const longContextThreshold = 200_001

// TestLongContextBracketsMatchSurcharge preserves main's pre-existing
// assertions for older catalog entries. Anthropic now documents Claude 4.6+
// pricing as flat across the full 1M context window; Opus 5 is correct here,
// while removal of the remaining stale brackets is tracked in #183.
//
//	input  = 2x base
//	output = 1.5x base
//	cache reads and writes = 2x base
//
// This narrow exclusion keeps the Opus 5 feature independent from #183 while
// still guarding the current main-branch behavior for every existing entry.
func TestLongContextBracketsMatchSurcharge(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		if def.Constraints.MaxInputTokens < 1_000_000 {
			continue
		}

		if id == ModelClaudeOpus5 {
			continue
		}

		t.Run(id, func(t *testing.T) {
			t.Parallel()

			type namedCard struct {
				name string
				card pricing.RateCard
			}

			cards := make([]namedCard, 0, 1+len(def.Pricing.Overrides))
			cards = append(cards, namedCard{"default", def.Pricing.Default})

			for _, ov := range def.Pricing.Overrides {
				cards = append(cards, namedCard{fmt.Sprintf("%+v", ov.Match), ov.RateCard})
			}

			for _, c := range cards {
				bracket := findBracket(c.card.Brackets, longContextThreshold)
				require.NotNilf(t, bracket,
					"%s card of 1M model %s has no >200K bracket — large requests under-report cost", c.name, id)

				assertSurcharge(t, c.name, c.card.Base, bracket.Rates)
			}
		})
	}
}

// TestOpus5PricingStaysFlatAcrossContextWindow guards Opus 5's documented
// exception to Anthropic's older >200K long-context surcharge.
func TestOpus5PricingStaysFlatAcrossContextWindow(t *testing.T) {
	t.Parallel()

	def, ok := supportedModels[ModelClaudeOpus5]
	require.True(t, ok)
	assert.Empty(t, def.Pricing.Default.Brackets)

	for _, ov := range def.Pricing.Overrides {
		assert.Empty(t, ov.RateCard.Brackets)
	}
}

// TestNonLongContextModelsStayFlat guards the inverse: 200K models must not
// grow a context bracket, which would silently overcharge above 200K.
func TestNonLongContextModelsStayFlat(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		if def.Constraints.MaxInputTokens >= 1_000_000 {
			continue
		}

		t.Run(id, func(t *testing.T) {
			t.Parallel()

			assert.Emptyf(t, def.Pricing.Default.Brackets,
				"200K model %s must not have context brackets", id)

			for _, ov := range def.Pricing.Overrides {
				assert.Emptyf(t, ov.RateCard.Brackets,
					"200K model %s override %s must not have context brackets", id, fmt.Sprintf("%+v", ov.Match))
			}
		})
	}
}

// TestLongContextPricingAppliesEndToEnd drives a real Catalog built from the
// Anthropic model pricing and proves the >200K bracket actually fires: the same
// token usage costs 2x on input when the request context crosses 200K, and the
// applied bracket threshold is reported. This closes the loop from catalog entry
// to computed cost.
func TestLongContextPricingAppliesEndToEnd(t *testing.T) {
	t.Parallel()

	cat, err := pricing.NewCatalog(pricing.WithProvider("anthropic", ModelPricing()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 100_000, OutputTokens: 1_000}

	below, err := cat.Calculate(ModelClaudeSonnet5, usage, pricing.CalcRequest{ContextTokens: 100_000})
	require.NoError(t, err)

	above, err := cat.Calculate(ModelClaudeSonnet5, usage, pricing.CalcRequest{ContextTokens: 300_000})
	require.NoError(t, err)

	assert.Equal(t, int64(0), below.AppliedBracketMinContextTokens,
		"a sub-200K request must price on the base card")
	assert.Equal(t, int64(longContextThreshold), above.AppliedBracketMinContextTokens,
		"a >200K request must price on the long-context bracket")

	assert.Equal(t,
		below.Breakdown[pricing.UsageFieldInput]*2, above.Breakdown[pricing.UsageFieldInput],
		"input above 200K must cost 2x the base rate for identical usage")
	assert.Greater(t, above.Total, below.Total,
		"the long-context request must cost more overall")
}

func findBracket(brackets []pricing.Bracket, minContext int64) *pricing.Bracket {
	for i := range brackets {
		if brackets[i].MinContextTokens == minContext {
			return &brackets[i]
		}
	}

	return nil
}

func assertSurcharge(t *testing.T, card string, base, got pricing.Rates) {
	t.Helper()

	assert.Equalf(t, base.InputPerMillion*2, got.InputPerMillion,
		"%s: input above 200K must be 2x base", card)
	assert.Equalf(t, base.OutputPerMillion*3/2, got.OutputPerMillion,
		"%s: output above 200K must be 1.5x base", card)
	assert.Equalf(t, base.CachedInputPerMillion*2, got.CachedInputPerMillion,
		"%s: cache read above 200K must be 2x base", card)
	assert.Equalf(t, base.CacheCreation5mPerMillion*2, got.CacheCreation5mPerMillion,
		"%s: 5m cache write above 200K must be 2x base", card)
	assert.Equalf(t, base.CacheCreation1hPerMillion*2, got.CacheCreation1hPerMillion,
		"%s: 1h cache write above 200K must be 2x base", card)
}
