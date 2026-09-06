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

// maxOutputTokens is a conservative SDK limit, not a published vendor maximum.
// Keep the draft explicit about this bound until the vendor cap is confirmed.
const maxOutputTokens = 32_768

var catalogOnce = sync.OnceValue(func() *catalog.Catalog {
	return catalog.MustNew("meta", entries())
})

// Catalog returns the immutable Meta Model API catalog.
func Catalog() *catalog.Catalog { return catalogOnce() }

func entries() []catalog.Entry {
	return []catalog.Entry{{
		ID:    ModelMuseSpark13,
		Model: catalog.ModelMuseSpark13,
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
			MaxOutputTokens:   maxOutputTokens,
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
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
	}}
}
