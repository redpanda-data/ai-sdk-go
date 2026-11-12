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

package meta

import (
	"sync"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

// ModelMuseSpark13 is the Standard-tier model. Prompts and completions are
// not used for training. The Contributor tier is intentionally not registered.
const ModelMuseSpark13 = "muse-spark-1.3"

// ModelMuseWonder10 is Muse Wonder, in preview; Model API serves it in
// Standard mode only.
const ModelMuseWonder10 = "muse-wonder-1.0"

// ModelMuseEmber10 is Muse Ember; Model API serves it in Standard mode only.
const ModelMuseEmber10 = "muse-ember-1.0"

// maxOutputTokens is a conservative SDK limit, not a published vendor maximum,
// for the models whose pages publish none. NewModel bounds explicitly supplied
// budgets by each model's MaxOutputTokens; omitted budgets use the server default.
const maxOutputTokens = 32_768

// ProviderName is the catalog key Provider.Name() returns; see pricing.ProviderKey.
const ProviderName = "meta"

var catalogOnce = sync.OnceValue(func() *catalog.Catalog {
	return catalog.MustNew(ProviderName, entries())
})

// Catalog returns the immutable Meta Model API catalog.
func Catalog() *catalog.Catalog { return catalogOnce() }

func entries() []catalog.Entry {
	return []catalog.Entry{{
		ID:    ModelMuseSpark13,
		Model: catalog.ModelMuseSpark13,
		WireAPIs: []catalog.WireAPI{
			{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect},
			{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
		},
		// https://dev.meta.ai/docs/models
		Capabilities: llm.ModelCapabilities{
			Streaming: true, Tools: true, JSONMode: true, StructuredOutput: true,
			Vision: true, Audio: true, MultiTurn: true, SystemPrompts: true, Reasoning: true,
		},
		Modalities: catalog.Modalities{
			// Audio is accepted, but Meta warns that 1.3 audio quality is degraded.
			Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityVideo, catalog.ModalityAudio, catalog.ModalityDocument},
			Output: []catalog.Modality{catalog.ModalityText},
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0, 2},
			MaxInputTokens:   1_048_576,
			// Meta publishes the shared context window, not a separate output cap.
			// This is the SDK-enforced bound, not a claim about the vendor limit.
			MaxOutputTokens: maxOutputTokens,
			SupportedParams: []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
		},
		// https://dev.meta.ai/docs/reasoning — max is Standard-tier only.
		Reasoning: catalog.ReasoningSupport{Efforts: []llm.ReasoningEffort{
			openai.ReasoningEffortMinimal, openai.ReasoningEffortLow, openai.ReasoningEffortMedium,
			openai.ReasoningEffortHigh, openai.ReasoningEffortXHigh, openai.ReasoningEffortMax,
		}},
		Attributes: map[string]string{"output_token_limit_source": "sdk_conservative_limit"},
		Life:       catalog.Lifecycle{Available: catalog.MustDate("2026-09-02")},
		// https://dev.meta.ai/docs/pricing-rate-limits — no long-context premium.
		Pricing: pricing.FlatInfo(1.25, 4.25, 0.15),
	}, {
		ID:    ModelMuseWonder10,
		Model: catalog.ModelMuseWonder10,
		WireAPIs: []catalog.WireAPI{
			{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect},
			{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
		},
		// https://dev.meta.ai/docs/models (read 2026-11-05): text, image, video,
		// audio and PDF in, text out; tool calling and structured output.
		Capabilities: llm.ModelCapabilities{
			Streaming: true, Tools: true, JSONMode: true, StructuredOutput: true,
			Vision: true, Audio: true, MultiTurn: true, SystemPrompts: true, Reasoning: true,
		},
		Modalities: catalog.Modalities{
			Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityVideo, catalog.ModalityAudio, catalog.ModalityDocument},
			Output: []catalog.Modality{catalog.ModalityText},
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0, 2},
			// 1,048,576-token context window per https://dev.meta.ai/docs/models (read 2026-11-05).
			MaxInputTokens: 1_048_576,
			// Meta publishes the shared context window, not a separate output cap.
			// This is the SDK-enforced bound, not a claim about the vendor limit.
			MaxOutputTokens: maxOutputTokens,
			SupportedParams: []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
		},
		// https://dev.meta.ai/docs/reasoning (read 2026-11-05): minimal to max.
		Reasoning: catalog.ReasoningSupport{Efforts: []llm.ReasoningEffort{
			openai.ReasoningEffortMinimal, openai.ReasoningEffortLow, openai.ReasoningEffortMedium,
			openai.ReasoningEffortHigh, openai.ReasoningEffortXHigh, openai.ReasoningEffortMax,
		}},
		Attributes: map[string]string{"output_token_limit_source": "sdk_conservative_limit"},
		// Preview, on Model API since 2026-10-12 per https://dev.meta.ai/docs/models
		// and https://research.meta.ai/blog/muse-wonder (read 2026-11-05).
		Life: catalog.Lifecycle{Stage: catalog.StagePreview, Available: catalog.MustDate("2026-10-12")},
		// https://dev.meta.ai/docs/pricing-rate-limits (read 2026-11-05) — Standard
		// mode; no long-context premium.
		Pricing: pricing.FlatInfo(4.00, 20.00, 0.20),
	}, {
		ID:    ModelMuseEmber10,
		Model: catalog.ModelMuseEmber10,
		WireAPIs: []catalog.WireAPI{
			{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect},
			{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
		},
		// https://dev.meta.ai/docs/muse-ember (read 2026-11-12): text, image,
		// video, audio and PDF in, text out; tool calling and structured output.
		// It "accepts the same request body as Muse Spark".
		Capabilities: llm.ModelCapabilities{
			Streaming: true, Tools: true, JSONMode: true, StructuredOutput: true,
			Vision: true, Audio: true, MultiTurn: true, SystemPrompts: true, Reasoning: true,
		},
		Modalities: catalog.Modalities{
			Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityVideo, catalog.ModalityAudio, catalog.ModalityDocument},
			Output: []catalog.Modality{catalog.ModalityText},
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0, 2},
			// 1,048,576-token context window and 524,288-token max output per
			// https://dev.meta.ai/docs/muse-ember (read 2026-11-12); no separate
			// input cap is published.
			MaxInputTokens:  1_048_576,
			MaxOutputTokens: 524_288,
			SupportedParams: []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
		},
		// https://dev.meta.ai/docs/reasoning (read 2026-11-12): minimal to max.
		Reasoning: catalog.ReasoningSupport{Efforts: []llm.ReasoningEffort{
			openai.ReasoningEffortMinimal, openai.ReasoningEffortLow, openai.ReasoningEffortMedium,
			openai.ReasoningEffortHigh, openai.ReasoningEffortXHigh, openai.ReasoningEffortMax,
		}},
		// On Model API since 2026-11-09 per https://research.meta.ai/blog/muse-ember
		// (read 2026-11-12).
		Life: catalog.Lifecycle{Available: catalog.MustDate("2026-11-09")},
		// https://dev.meta.ai/docs/pricing-rate-limits (read 2026-11-12) — Standard
		// mode; a request with more than 200K input tokens, cached included, is
		// billed in full at the long-context rates.
		Pricing: pricing.TieredInfo(
			pricing.NewRates(0.10, 0.50, 0.004),
			pricing.Bracket{
				MinContextTokens: 200_001,
				Rates:            pricing.NewRates(0.30, 1.50, 0.012),
			},
		),
	}}
}
