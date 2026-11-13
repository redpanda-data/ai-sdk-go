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

package fireworks_test

import (
	"encoding/json"
	"math"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

var contractPricing = sync.OnceValues(func() (*pricing.Catalog, error) {
	return pricing.NewCatalog(pricing.WithSource(fireworks.Catalog()))
})

func contractPricingUsage(api catalog.WireProtocol, r contractResult) *llm.TokenUsage {
	input := r.Input
	if api != catalog.WireProtocolAnthropicMessages {
		input -= r.Cached
	}

	return &llm.TokenUsage{
		InputTokens: int(input), CachedInputTokens: int(r.Cached),
		OutputTokens: int(r.Output - r.Reasoned), ReasoningTokens: int(r.Reasoned),
	}
}

func contractPricingRequest(c contractCase) pricing.CalcRequest {
	var request pricing.CalcRequest
	if c.Feature == contractPriority {
		request.Selector.ServiceTier = llm.ServiceTierPriority
	}

	return request
}

func contractCost(c contractCase, input, cached, output int64) float64 {
	pc, err := contractPricing()
	if err != nil {
		return math.Inf(1)
	}

	return contractCostWithCatalog(pc, c, input, cached, output)
}

func contractCostWithCatalog(pc *pricing.Catalog, c contractCase, input, cached, output int64) float64 {
	u := contractPricingUsage(c.API, contractResult{Input: input, Cached: cached, Output: output})
	request := contractPricingRequest(c)

	cost, err := pc.Calculate(fireworks.ProviderName, c.Model, u, request)
	if err != nil || len(cost.Unpriced) > 0 || len(cost.Fallbacks) > 0 || cost.AppliedSelector != request.Selector {
		return math.Inf(1)
	}

	return float64(cost.Total) / 1e8
}

func contractReservation(c contractCase, input, output int64) float64 {
	pc, err := contractPricing()
	if err != nil {
		return math.Inf(1)
	}

	return contractReservationWithCatalog(pc, c, input, output)
}

func contractReservationWithCatalog(pc *pricing.Catalog, c contractCase, input, output int64) float64 {
	cachedInput := input
	if c.API == catalog.WireProtocolAnthropicMessages {
		cachedInput = 0
	}

	return max(contractCostWithCatalog(pc, c, input, 0, output), contractCostWithCatalog(pc, c, cachedInput, input, output))
}

func contractSyntheticPricing(t *testing.T, info pricing.Info) *pricing.Catalog {
	t.Helper()

	pc, err := pricing.NewCatalog(pricing.WithProvider(fireworks.ProviderName, map[string]pricing.Info{"synthetic": info}))
	require.NoError(t, err)

	return pc
}

func TestFireworksContractSyntheticPricing(t *testing.T) {
	t.Parallel()

	pc := contractSyntheticPricing(t, pricing.FlatInfo(2, 5, .5).WithOverride(
		pricing.Selector{ServiceTier: llm.ServiceTierPriority}, pricing.RateCard{Base: pricing.NewRates(4, 10, 1)}))

	for _, test := range []struct {
		api   catalog.WireProtocol
		usage string
	}{
		{catalog.WireProtocolAnthropicMessages, `{"input_tokens":80,"cache_read_input_tokens":20,"output_tokens":30,"output_tokens_details":{"reasoning_tokens":10}}`},
		{catalog.WireProtocolOpenAIChatCompletions, `{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":20},"completion_tokens":30,"completion_tokens_details":{"reasoning_tokens":10}}`},
		{catalog.WireProtocolOpenAIResponses, `{"input_tokens":100,"input_tokens_details":{"cached_tokens":20},"output_tokens":30,"output_tokens_details":{"reasoning_tokens":10}}`},
	} {
		t.Run(string(test.api), func(t *testing.T) {
			t.Parallel()

			var native map[string]any
			require.NoError(t, json.Unmarshal([]byte(test.usage), &native))
			r := contractParseObject(test.api, map[string]any{"model": "synthetic", "usage": native})
			u := contractPricingUsage(test.api, r)
			require.Equal(t, &llm.TokenUsage{InputTokens: 80, CachedInputTokens: 20, OutputTokens: 20, ReasoningTokens: 10}, u)

			for _, tier := range []struct {
				feature string
				factor  int64
			}{
				{contractComplete, 1},
				{contractPriority, 2},
			} {
				t.Run(tier.feature, func(t *testing.T) {
					t.Parallel()

					c := contractCase{Model: r.Model, API: test.api, Feature: tier.feature}
					cost, err := pc.Calculate(fireworks.ProviderName, r.Model, u, contractPricingRequest(c))
					require.NoError(t, err)
					require.Empty(t, cost.Unpriced)
					require.Empty(t, cost.Fallbacks)
					require.Equal(t, int64(32_000)*tier.factor, cost.Total, "80*$2 + 20*$0.50 + (20+10)*$5 per million; Priority doubles each rate")
					require.Equal(t, int64(16_000)*tier.factor, cost.Breakdown[pricing.UsageFieldInput])
					require.Equal(t, int64(1_000)*tier.factor, cost.Breakdown[pricing.UsageFieldCachedInput])
					require.Equal(t, int64(10_000)*tier.factor, cost.Breakdown[pricing.UsageFieldOutput])
					require.Equal(t, int64(5_000)*tier.factor, cost.Breakdown[pricing.UsageFieldReasoning])
					require.InDelta(t, .00032*float64(tier.factor), contractCostWithCatalog(pc, c, r.Input, r.Cached, r.Output), 1e-12)
					require.InDelta(t, .00035*float64(tier.factor), contractReservationWithCatalog(pc, c, 100, 30), 1e-12)
				})
			}
		})
	}
}

func TestFireworksContractSyntheticCatalogRepricing(t *testing.T) {
	t.Parallel()

	c := contractCase{Model: "synthetic", API: catalog.WireProtocolOpenAIResponses}
	e := contractExchange{
		ResponseHeaders: map[string]string{"Content-Type": "application/json"},
		Response:        `{"model":"synthetic","usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":20},"output_tokens":30,"output_tokens_details":{"reasoning_tokens":10}}}`,
	}

	for _, test := range []struct {
		name string
		info pricing.Info
		cost float64
		hold float64
	}{
		{"original", pricing.FlatInfo(2, 5, .5), .00032, .00035},
		{"new_catalog_prices", pricing.FlatInfo(4, 10, 1), .00064, .0007},
		{"cache_more_expensive_than_fresh_input", pricing.FlatInfo(2, 5, 3), .00037, .00045},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			pc := contractSyntheticPricing(t, test.info)
			r := contractParse(t, c, e)
			require.InDelta(t, test.cost, contractCostWithCatalog(pc, c, r.Input, r.Cached, r.Output), 1e-12)
			require.InDelta(t, test.hold, contractReservationWithCatalog(pc, c, 100, 30), 1e-12)
		})
	}
}

func TestFireworksContractSyntheticUnpricedUsage(t *testing.T) {
	t.Parallel()

	for _, api := range []catalog.WireProtocol{catalog.WireProtocolAnthropicMessages, catalog.WireProtocolOpenAIChatCompletions, catalog.WireProtocolOpenAIResponses} {
		t.Run(string(api), func(t *testing.T) {
			t.Parallel()

			c := contractCase{Model: "synthetic", API: api}
			missingCache := contractSyntheticPricing(t, pricing.FlatInfo(2, 5, 0))
			require.True(t, math.IsInf(contractCostWithCatalog(missingCache, c, 100, 20, 30), 1))
			require.True(t, math.IsInf(contractReservationWithCatalog(missingCache, c, 100, 30), 1))
			require.InDelta(t, .00035, contractCostWithCatalog(missingCache, c, 100, 0, 30), 1e-12)

			pc := contractSyntheticPricing(t, pricing.FlatInfo(2, 5, pricing.RateFree))
			c.Feature = contractPriority
			require.True(t, math.IsInf(contractCostWithCatalog(pc, c, 100, 20, 30), 1), "missing requested tier must not silently fall back to cheaper rates")
			c.Feature = contractComplete
			require.InDelta(t, .00035, contractReservationWithCatalog(pc, c, 100, 30), 1e-12)
			c.Model = "unknown-reported-identity"
			require.True(t, math.IsInf(contractCostWithCatalog(pc, c, 100, 20, 30), 1))
			require.True(t, math.IsInf(contractReservationWithCatalog(pc, c, 100, 30), 1))
		})
	}
}
