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

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// TestAstraSurfaceCatalog covers the GPT-6 models that share Astra's
// request surface: reasoning effort low through max, never none or
// minimal, and no sampling parameters.
func TestAstraSurfaceCatalog(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		id          string
		model       catalog.ModelID
		displayName string
	}{
		{ModelGPT6Astra, "openai/gpt-6-astra", "GPT-6 Astra"},
		{ModelGPT6_1Sol, "openai/gpt-6.1-sol", "GPT-6.1 Sol"},
	} {
		t.Run(tt.id, func(t *testing.T) {
			t.Parallel()

			offering, ok := Catalog().Lookup(tt.id)
			require.True(t, ok)
			assert.Equal(t, tt.model, offering.Model)
			assert.Equal(t, tt.displayName, offering.DisplayName)
			assert.Equal(t, 922_000, offering.Constraints.MaxInputTokens)
			assert.Equal(t, 128_000, offering.Constraints.MaxOutputTokens)
			assert.Equal(t, []llm.ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax}, offering.Reasoning.Efforts)
			assert.Equal(t, []catalog.Modality{catalog.ModalityText, catalog.ModalityImage}, offering.Modalities.Input)
			assert.Equal(t, []catalog.Modality{catalog.ModalityText}, offering.Modalities.Output)
			assert.True(t, offering.Capabilities.Tools)
			assert.True(t, offering.Capabilities.ToolSearch)
			assert.True(t, offering.Capabilities.StructuredOutput)
			assert.True(t, offering.Capabilities.Streaming)

			provider, err := NewProvider("test-key")
			require.NoError(t, err)

			for _, effort := range offering.Reasoning.Efforts {
				model, err := provider.NewModel(tt.id, WithReasoningEffort(effort))
				require.NoError(t, err)
				assert.Equal(t, tt.id, model.Name())
				concrete, ok := model.(*Model)
				require.True(t, ok)

				request, err := concrete.requestMapper.ToProvider(&llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("hello")}}}})
				require.NoError(t, err)
				assert.Equal(t, tt.id, request.Model)
				assert.Equal(t, string(effort), string(request.Reasoning.Effort))
			}

			for _, option := range []Option{WithReasoningEffort(ReasoningEffortNone), WithReasoningEffort(ReasoningEffortMinimal), WithTemperature(0.5), WithTopP(0.9)} {
				_, err := provider.NewModel(tt.id, option)
				require.Error(t, err)
			}
		})
	}
}

func TestGPT6TieredPricing(t *testing.T) {
	t.Parallel()

	prices, err := pricing.NewCatalog(pricing.WithSource(Catalog()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 1_000_000, CachedInputTokens: 1_000_000, CacheCreationUnknownTTLTokens: 1_000_000, OutputTokens: 1_000_000}
	standard := pricing.Selector{}
	ultrafast := pricing.Selector{ServiceTier: llm.ServiceTierUltrafast}

	for _, tt := range []struct {
		name                         string
		model                        string
		selector                     pricing.Selector
		context                      int64
		input, cached, write, output int64
	}{
		{"Astra at threshold", ModelGPT6Astra, standard, 272_000, 1_000_000_000, 100_000_000, 1_250_000_000, 5_000_000_000},
		{"Astra above threshold", ModelGPT6Astra, standard, 272_001, 2_000_000_000, 200_000_000, 2_500_000_000, 7_500_000_000},
		{"Astra Ultrafast at threshold", ModelGPT6Astra, ultrafast, 272_000, 6_000_000_000, 600_000_000, 7_500_000_000, 30_000_000_000},
		{"Astra Ultrafast above threshold", ModelGPT6Astra, ultrafast, 272_001, 12_000_000_000, 1_200_000_000, 15_000_000_000, 45_000_000_000},
		{"6.1 Sol at threshold", ModelGPT6_1Sol, standard, 272_000, 200_000_000, 10_000_000, 250_000_000, 1_000_000_000},
		{"6.1 Sol above threshold", ModelGPT6_1Sol, standard, 272_001, 400_000_000, 20_000_000, 500_000_000, 1_500_000_000},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cost, err := prices.Calculate(ProviderName, tt.model, usage, pricing.CalcRequest{Selector: tt.selector, ContextTokens: tt.context})
			require.NoError(t, err)
			assert.Empty(t, cost.Unpriced)
			assert.Equal(t, tt.selector, cost.AppliedSelector)
			assert.Equal(t, tt.input, cost.Breakdown[pricing.UsageFieldInput])
			assert.Equal(t, tt.cached, cost.Breakdown[pricing.UsageFieldCachedInput])
			assert.Equal(t, tt.write, cost.Breakdown[pricing.UsageFieldCacheCreationUnknownTTL])
			assert.Equal(t, tt.output, cost.Breakdown[pricing.UsageFieldOutput])
		})
	}
}

// TestAstraUltrafastPricesFromResponse pins the selector dimension: OpenAI
// reports Ultrafast in the response's service_tier, so the override must key
// on ServiceTier for SelectorFromResponse to reach it.
func TestAstraUltrafastPricesFromResponse(t *testing.T) {
	t.Parallel()

	resp, err := NewResponseMapper().FromProvider(&responses.Response{
		ID:          "resp_ultrafast",
		Model:       ModelGPT6Astra,
		Status:      responses.ResponseStatusCompleted,
		ServiceTier: "ultrafast",
		Output: []responses.ResponseOutputItemUnion{{
			Type:    outputTypeMessage,
			Content: []responses.ResponseOutputMessageContentUnion{{Type: contentTypeOutputText, Text: "hi"}},
		}},
		Usage: responses.ResponseUsage{InputTokens: 1_000, OutputTokens: 1_000, TotalTokens: 2_000},
	})
	require.NoError(t, err)
	assert.Equal(t, llm.ServiceTierUltrafast, resp.ServiceTier)

	prices, err := pricing.NewCatalog(pricing.WithSource(Catalog()))
	require.NoError(t, err)

	cost, err := prices.Calculate(ProviderName, resp.InvokedModelID, resp.Usage, pricing.CalcRequest{Selector: pricing.SelectorFromResponse(resp)})
	require.NoError(t, err)
	assert.Equal(t, pricing.Selector{ServiceTier: llm.ServiceTierUltrafast}, cost.AppliedSelector)
	assert.Equal(t, int64(6_000_000), cost.Breakdown[pricing.UsageFieldInput])
	assert.Equal(t, int64(30_000_000), cost.Breakdown[pricing.UsageFieldOutput])
}
