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

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

func TestGPT6AstraCatalog(t *testing.T) {
	t.Parallel()

	offering, ok := Catalog().Lookup("gpt-6-astra")
	require.True(t, ok)
	assert.Equal(t, catalog.ModelID("openai/gpt-6-astra"), offering.Model)
	assert.Equal(t, "GPT-6 Astra", offering.DisplayName)
	assert.Equal(t, 1_050_000, offering.Constraints.MaxInputTokens)
	assert.Equal(t, 128_000, offering.Constraints.MaxOutputTokens)
	assert.Equal(t, []llm.ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax}, offering.Reasoning.Efforts)
	assert.Equal(t, []catalog.Modality{catalog.ModalityText, catalog.ModalityImage}, offering.Modalities.Input)
	assert.Equal(t, []catalog.Modality{catalog.ModalityText}, offering.Modalities.Output)
	assert.True(t, offering.Capabilities.Tools)
	assert.True(t, offering.Capabilities.StructuredOutput)
	assert.True(t, offering.Capabilities.Streaming)

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	for _, effort := range offering.Reasoning.Efforts {
		model, err := provider.NewModel("gpt-6-astra", WithReasoningEffort(effort))
		require.NoError(t, err)
		assert.Equal(t, "gpt-6-astra", model.Name())
	}

	for _, option := range []Option{WithReasoningEffort(ReasoningEffortNone), WithReasoningEffort(ReasoningEffortMinimal), WithTemperature(0.5), WithTopP(0.9)} {
		_, err := provider.NewModel("gpt-6-astra", option)
		require.Error(t, err)
	}
}

func TestGPT6AstraPricing(t *testing.T) {
	t.Parallel()

	prices, err := pricing.NewCatalog(pricing.WithProvider("openai", Catalog().PricingByID()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 1_000_000, CachedInputTokens: 1_000_000, CacheCreationUnknownTTLTokens: 1_000_000, OutputTokens: 1_000_000}

	for _, tt := range []struct {
		name                         string
		context                      int64
		input, cached, write, output int64
	}{
		{"standard at threshold", 272_000, 1_000_000_000, 100_000_000, 1_250_000_000, 5_000_000_000},
		{"long context above threshold", 272_001, 2_000_000_000, 200_000_000, 2_500_000_000, 7_500_000_000},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cost, err := prices.Calculate("gpt-6-astra", usage, pricing.CalcRequest{ContextTokens: tt.context})
			require.NoError(t, err)
			assert.Empty(t, cost.Unpriced)
			assert.Equal(t, tt.input, cost.Breakdown[pricing.UsageFieldInput])
			assert.Equal(t, tt.cached, cost.Breakdown[pricing.UsageFieldCachedInput])
			assert.Equal(t, tt.write, cost.Breakdown[pricing.UsageFieldCacheCreationUnknownTTL])
			assert.Equal(t, tt.output, cost.Breakdown[pricing.UsageFieldOutput])
		})
	}
}
