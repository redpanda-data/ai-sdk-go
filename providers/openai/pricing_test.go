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

package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

func TestGPT56Pricing(t *testing.T) {
	t.Parallel()

	catalog, err := pricing.NewCatalog(pricing.WithProvider("openai", ModelPricing()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{
		InputTokens:                   1_000_000,
		CachedInputTokens:             1_000_000,
		CacheCreationUnknownTTLTokens: 1_000_000,
		OutputTokens:                  1_000_000,
	}
	tests := []struct {
		name        string
		model       string
		context     int64
		wantBracket int64
		wantInput   int64
		wantCached  int64
		wantWrite   int64
		wantOutput  int64
	}{
		{
			name:       "Luna at 272K uses base rates",
			model:      ModelGPT5_6Luna,
			context:    272_000,
			wantInput:  100_000_000,
			wantCached: 10_000_000,
			wantWrite:  125_000_000,
			wantOutput: 600_000_000,
		},
		{
			name:        "Luna above 272K uses long-context rates",
			model:       ModelGPT5_6Luna,
			context:     272_001,
			wantBracket: 272_001,
			wantInput:   200_000_000,
			wantCached:  20_000_000,
			wantWrite:   250_000_000,
			wantOutput:  900_000_000,
		},
		{
			name:       "Terra at 272K uses base rates",
			model:      ModelGPT5_6Terra,
			context:    272_000,
			wantInput:  250_000_000,
			wantCached: 25_000_000,
			wantWrite:  312_500_000,
			wantOutput: 1_500_000_000,
		},
		{
			name:        "Terra above 272K uses long-context rates",
			model:       ModelGPT5_6Terra,
			context:     272_001,
			wantBracket: 272_001,
			wantInput:   500_000_000,
			wantCached:  50_000_000,
			wantWrite:   625_000_000,
			wantOutput:  2_250_000_000,
		},
		{
			name:       "Sol at 272K uses base rates",
			model:      ModelGPT5_6Sol,
			context:    272_000,
			wantInput:  500_000_000,
			wantCached: 50_000_000,
			wantWrite:  625_000_000,
			wantOutput: 3_000_000_000,
		},
		{
			name:        "Sol above 272K uses long-context rates",
			model:       ModelGPT5_6Sol,
			context:     272_001,
			wantBracket: 272_001,
			wantInput:   1_000_000_000,
			wantCached:  100_000_000,
			wantWrite:   1_250_000_000,
			wantOutput:  4_500_000_000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cost, err := catalog.Calculate(tt.model, usage, pricing.CalcRequest{ContextTokens: tt.context})
			require.NoError(t, err)
			assert.Empty(t, cost.Unpriced)
			assert.Equal(t, tt.wantBracket, cost.AppliedBracketMinContextTokens)
			assert.Equal(t, tt.wantInput, cost.Breakdown[pricing.UsageFieldInput])
			assert.Equal(t, tt.wantCached, cost.Breakdown[pricing.UsageFieldCachedInput])
			assert.Equal(t, tt.wantWrite, cost.Breakdown[pricing.UsageFieldCacheCreationUnknownTTL])
			assert.Equal(t, tt.wantOutput, cost.Breakdown[pricing.UsageFieldOutput])
		})
	}
}

func TestAllModelsHavePricing(t *testing.T) {
	t.Parallel()

	// Models that predate prompt caching or don't support it.
	noCacheModels := map[string]bool{
		ModelGPT4Turbo:  true, // Legacy model, no prompt caching.
		ModelGPT35Turbo: true, // Legacy model, no prompt caching.
		ModelGPT5_2Pro:  true, // Pro tier, no caching listed.
		ModelO3Pro:      true, // Pro tier, no caching listed.
	}

	for id, def := range supportedModels {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			assert.Positive(t, def.Pricing.Default.Base.InputPerMillion,
				"model %s missing input pricing — add Pricing to its ModelDefinition", id)
			assert.Positive(t, def.Pricing.Default.Base.OutputPerMillion,
				"model %s missing output pricing — add Pricing to its ModelDefinition", id)

			if !noCacheModels[id] {
				assert.Positive(t, def.Pricing.Default.Base.CachedInputPerMillion,
					"model %s missing cached pricing — add CachedInputPerMillion or add to noCacheModels if intentional", id)
			}
		})
	}
}

func TestGPT56AliasHasSolPricing(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	aliasPricing, ok := pricingMap[ModelGPT5_6]
	require.True(t, ok)
	assert.Equal(t, supportedModels[ModelGPT5_6Sol].Pricing, aliasPricing)
}

func TestModelPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	assert.Len(t, pricingMap, len(supportedModels)+len(modelAliases),
		"ModelPricing should return one entry per supported model and alias")
}
