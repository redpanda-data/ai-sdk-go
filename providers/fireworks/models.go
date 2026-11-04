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

package fireworks

import (
	"sync"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// ProviderName is the provider-scoped discovery and pricing key.
const ProviderName = "fireworks"

const (
	ModelGLM53                = "accounts/fireworks/models/glm-5p3"
	ModelGLM53Flash           = "accounts/fireworks/models/glm-5p3-flash"
	ModelMiniMaxM3            = "accounts/fireworks/models/minimax-m3"
	ModelDeepSeekV41Flash     = "accounts/fireworks/models/deepseek-v4p1-flash"
	ModelDeepSeekV41Flash1017 = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
	ModelKimiK3               = "accounts/fireworks/models/kimi-k3"

	// Deprecated: Fireworks serverless retires this offering on 2026-11-07.
	// No replacement is declared by the provider in this catalog.
	ModelGPTOSS120B = "accounts/fireworks/models/gpt-oss-120b"
)

var modelCatalog = sync.OnceValue(func() *catalog.Catalog {
	return catalog.MustNew(ProviderName, entries(), catalog.WithRequiredWireAPIs())
})

// Catalog returns Fireworks' immutable, explicitly scoped serverless catalog.
func Catalog() *catalog.Catalog {
	return modelCatalog()
}

func entries() []catalog.Entry {
	// Checked 2026-11-04: https://docs.fireworks.ai/serverless/pricing
	// Prices are input/output/cache USD per million, not per-token rates.
	// Output ceiling: https://docs.z.ai/guides/llm/glm-5.3 (read 2026-11-04; capped-at-vendor).
	glm := serverlessEntry(ModelGLM53, catalog.ModelGLM53, 1048576, 131072, false,
		serverlessPrice(1.40, 4.40, .26, 1.75, 5.50, .325))
	glm.Reasoning.Efforts = []llm.ReasoningEffort{"low", "high", "max"}
	glm.WireAPISupport = apiSupport(glm.Capabilities, glm.Reasoning)

	// Output ceiling: https://docs.z.ai/guides/llm/glm-5.3-flash (read 2026-11-04; capped-at-vendor).
	flash := serverlessEntry(ModelGLM53Flash, catalog.ModelGLM53Flash, 1048576, 131072, true,
		serverlessPrice(.15, .50, .03, .1875, .625, .0375))
	flash.Reasoning.Efforts = []llm.ReasoningEffort{"low", "high", "max"}
	flash.WireAPISupport = apiSupport(flash.Capabilities, flash.Reasoning)

	// Host FAQ explicitly permits output up to the remaining served window:
	// https://fireworks.ai/models/fireworks/minimax-m3 (read 2026-11-04).
	mini := serverlessEntry(ModelMiniMaxM3, catalog.ModelMiniMaxM3, 512000, 512000, false,
		serverlessPrice(.30, 1.20, .06, .45, 1.80, .09))
	mini.Reasoning.Adaptive = true
	mini.WireAPISupport = apiSupport(mini.Capabilities, mini.Reasoning)

	// Creator output ceiling: https://developers.openai.com/api/docs/models/gpt-oss-120b (read 2026-11-04).
	oss := serverlessEntry(ModelGPTOSS120B, catalog.ModelGPTOSS120B, 131072, 131072, false,
		serverlessPrice(.15, .60, .015, .18, .72, .018))
	oss.Reasoning.Efforts = []llm.ReasoningEffort{"low", "medium", "high"}
	oss.WireAPISupport = apiSupport(oss.Capabilities, oss.Reasoning)
	// https://docs.fireworks.ai/updates/changelog (2026-10-19).
	oss.Life.Retires = catalog.MustDate("2026-11-07")

	// Creator output ceiling: https://api-docs.deepseek.com/api/list-models (read 2026-11-04).
	deepseek := serverlessEntry(ModelDeepSeekV41Flash, catalog.ModelDeepSeekV41Flash, 1048576, 393216, true,
		serverlessPrice(.30, 1.20, .006, .375, 1.50, .0075))
	deepseek.Reasoning.Efforts = []llm.ReasoningEffort{"none", "low", "high", "max"}
	deepseek.WireAPISupport = apiSupport(deepseek.Capabilities, deepseek.Reasoning)

	deepseek1017 := serverlessEntry(ModelDeepSeekV41Flash1017, catalog.ModelDeepSeekV41Flash1017, 1048576, 393216, true,
		serverlessPrice(.30, 1.20, .006, .375, 1.50, .0075))
	deepseek1017.Reasoning = deepseek.Reasoning
	deepseek1017.WireAPISupport = apiSupport(deepseek1017.Capabilities, deepseek1017.Reasoning)

	// Host FAQ explicitly permits output up to the remaining window:
	// https://fireworks.ai/models/fireworks/kimi-k3 (read 2026-11-04).
	kimi := serverlessEntry(ModelKimiK3, catalog.ModelKimiK3, 1048576, 1048576, true,
		serverlessPrice(3, 15, .30, 3.75, 18.75, .375))
	kimi.Reasoning.Efforts = []llm.ReasoningEffort{"low", "high", "max"}
	kimi.WireAPISupport = apiSupport(kimi.Capabilities, kimi.Reasoning)

	return []catalog.Entry{glm, flash, mini, oss, deepseek, deepseek1017, kimi}
}

func serverlessPrice(input, output, cached, priorityInput, priorityOutput, priorityCached float64) pricing.Info {
	// https://docs.fireworks.ai/tools-sdks/anthropic-compatibility#token-usage
	// Cache creation is reported as zero, not an Anthropic write multiplier.
	rates := pricing.NewRates(input, output, cached).
		WithCacheCreation(pricing.RateFree, pricing.RateFree, pricing.RateFree)
	priority := pricing.NewRates(priorityInput, priorityOutput, priorityCached).
		WithCacheCreation(pricing.RateFree, pricing.RateFree, pricing.RateFree)

	return pricing.FlatInfoFromRates(rates).WithOverride(
		pricing.Selector{ServiceTier: llm.ServiceTierPriority}, pricing.RateCard{Base: priority})
}

func serverlessEntry(id string, model catalog.ModelID, maxInput, maxOutput int, vision bool, price pricing.Info) catalog.Entry {
	caps := llm.ModelCapabilities{
		Streaming: true, Tools: true, JSONMode: true, StructuredOutput: true,
		Vision: vision, MultiTurn: true, SystemPrompts: true, Reasoning: true,
	}
	modalities := catalog.Modalities{Input: []catalog.Modality{catalog.ModalityText}, Output: []catalog.Modality{catalog.ModalityText}}

	if vision {
		modalities.Input = append(modalities.Input, catalog.ModalityImage)
	}

	return catalog.Entry{
		ID: id, Model: model, ResolutionPolicy: catalog.ResolutionPolicyExact, Capabilities: caps, Modalities: modalities,
		// Input metadata: GET /inference/v1/models (read 2026-11-04).
		// Output ceilings are independently sourced above, never inferred from input.
		// SupportedParams intentionally admits no SDK controls: there is no SDK model transport.
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0, 2}, MaxInputTokens: maxInput, MaxOutputTokens: maxOutput,
		},
		WireAPIs: []catalog.WireAPI{
			{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect},
			{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect},
			{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
		},
		WireAPISupport: apiSupport(caps, catalog.ReasoningSupport{}),
		Pricing:        price,
		Attributes: map[string]string{
			"pricing_checked":      "2026-11-04",
			"priority_tier_status": "blocked_missing_effective_tier",
		},
	}
}

func apiSupport(caps llm.ModelCapabilities, reasoning catalog.ReasoningSupport) []catalog.WireAPISupport {
	messages := caps
	messages.JSONMode = false
	responsesReasoning := reasoning
	// Responses does not document an adaptive request control.
	responsesReasoning.Adaptive = false

	return []catalog.WireAPISupport{
		{API: catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect}, Capabilities: messages, Reasoning: reasoning},
		{API: catalog.WireAPI{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect}, Capabilities: caps, Reasoning: reasoning},
		{API: catalog.WireAPI{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect}, Capabilities: caps, Reasoning: responsesReasoning},
	}
}
