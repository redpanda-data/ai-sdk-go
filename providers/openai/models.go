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
	"sync"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

var catalogOnce = sync.OnceValue(func() *catalog.Catalog {
	return catalog.MustNew("openai", entries())
})

// Catalog returns the validated OpenAI model catalog: every offering with
// its capabilities, constraints, modalities, reasoning controls, pricing,
// and lifecycle. The catalog is immutable and shared; all reads return
// deep copies.
func Catalog() *catalog.Catalog {
	return catalogOnce()
}

// Modality sets shared by the entries below.
var (
	textOnly = catalog.Modalities{
		Input:  []catalog.Modality{catalog.ModalityText},
		Output: []catalog.Modality{catalog.ModalityText},
	}
	textImage = catalog.Modalities{
		Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage},
		Output: []catalog.Modality{catalog.ModalityText},
	}
	textImageAudio = catalog.Modalities{
		Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityAudio},
		Output: []catalog.Modality{catalog.ModalityText},
	}
)

// Capability sets shared by the entries below.
var (
	// flagshipCaps is the full multimodal reasoning surface of the GPT-5
	// flagship and mini lines.
	flagshipCaps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true,
		StructuredOutput: true,
		Vision:           true,
		Audio:            true,
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        true,
	}

	// visionChatCaps covers multimodal-in chat models without reasoning:
	// the GPT-4.1 line, GPT-4o Mini, GPT-4 Turbo.
	visionChatCaps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true,
		StructuredOutput: true,
		Vision:           true,
		MultiTurn:        true,
		SystemPrompts:    true,
	}

	// textChatCaps covers text-only chat models without vision, audio, or
	// reasoning: GPT-3.5 Turbo.
	textChatCaps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true,
		StructuredOutput: true,
		MultiTurn:        true,
		SystemPrompts:    true,
	}

	// nanoCaps covers the GPT-5 nano tiers: image input and reasoning, but
	// no audio.
	nanoCaps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true,
		StructuredOutput: true,
		Vision:           true,
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        true,
	}

	// oSeriesCaps covers the o-series reasoning models: vision ("thinking
	// with images") but no JSON mode or structured output.
	oSeriesCaps = llm.ModelCapabilities{
		Streaming:     true,
		Tools:         true,
		Vision:        true,
		MultiTurn:     true,
		SystemPrompts: true,
		Reasoning:     true,
	}
)

// entries returns the authored OpenAI catalog.
//
// Lifecycle sourcing: OpenAI's deprecations page
// (https://developers.openai.com/api/docs/deprecations). OpenAI announces
// exact shutdown dates, which go to Retires; models with no announcement
// carry no dates. Available is the first-party launch date. ReplacedBy is
// set only when OpenAI's recommended replacement is itself a catalogued
// offering; recommendations pointing at models this SDK does not carry
// are omitted rather than approximated.
//
// The catalog is append-only: retired models keep their entries (with
// Retires in the past) so historical usage stays priceable and the
// failure stays explainable.
func entries() []catalog.Entry {
	return []catalog.Entry{
		// GPT-5 Series (2025 Flagship)
		{
			ID:           ModelGPT5,
			Model:        catalog.ModelGPT5,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    272000, // documented max input; the 400K window reserves 128K for output
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-08-07"),
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
				ReplacedBy: ModelGPT5_6Sol,
			},
			Pricing: pricing.FlatInfo(1.25, 10.00, 0.125),
		},
		{
			ID:           ModelGPT5Mini,
			Model:        catalog.ModelGPT5Mini,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    272000, // documented max input; the 400K window reserves 128K for output
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-08-07"),
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
				ReplacedBy: ModelGPT5_6Terra,
			},
			// $0.25 / $2.00 / $0.025 per M (input / output / cached input).
			Pricing: pricing.FlatInfo(0.25, 2.00, 0.025),
		},
		{
			ID:    ModelGPT5Nano,
			Model: catalog.ModelGPT5Nano,
			// Nano trades audio for speed but keeps image input and reasoning.
			Capabilities: nanoCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    272000, // documented max input; the 400K window reserves 128K for output
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "reasoning_effort"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-08-07"),
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
				ReplacedBy: ModelGPT5_6Luna,
			},
			Pricing: pricing.FlatInfo(0.05, 0.40, 0.005),
		},
		{
			ID:           ModelGPT5_1,
			Model:        catalog.ModelGPT5_1,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    272000, // documented max input; the 400K window reserves 128K for output
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2025-11-13"),
			},
			// $1.25 / $10.00 / $0.125 per M (input / output / cached input).
			Pricing: pricing.FlatInfo(1.25, 10.00, 0.125),
		},
		{
			ID:           ModelGPT5_2,
			Model:        catalog.ModelGPT5_2,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    400000, // 400K context window
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2025-12-11"),
			},
			Pricing: pricing.FlatInfo(1.75, 14.00, 0.175),
		},
		{
			// Retired 2026-08-10; entry retained (append-only catalog).
			ID:           ModelGPT5_2Instant,
			Model:        catalog.ModelGPT5_2Instant,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    400000, // 400K context window
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortMedium}, // Instant variant only supports medium
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-12-11"),
				Deprecated: catalog.MustDate("2026-05-08"),
				Retires:    catalog.MustDate("2026-08-10"),
				ReplacedBy: ModelGPT5_6Sol,
			},
			Pricing: pricing.FlatInfo(1.75, 14.00, 0.175),
		},
		{
			ID:           ModelGPT5_2Pro,
			Model:        catalog.ModelGPT5_2Pro,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    400000, // 400K context window
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh}, // Pro variant starts at medium
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2025-12-11"),
			},
			Pricing: pricing.FlatInfo(21.00, 168.00, 0),
		},
		{
			// Retired 2026-08-10; entry retained (append-only catalog).
			ID:           ModelGPT5_3ChatLatest,
			Model:        catalog.ModelGPT5_3Instant,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    400000, // 400K context window
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortMedium}, // Chat-latest only supports medium
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2026-03-03"),
				Deprecated: catalog.MustDate("2026-05-08"),
				Retires:    catalog.MustDate("2026-08-10"),
				ReplacedBy: ModelGPT5_6Sol,
			},
			Pricing: pricing.FlatInfo(1.75, 14.00, 0.175),
		},

		// GPT-5.6 Series
		gpt56Entry(ModelGPT5_6Luna, catalog.ModelGPT5_6Luna, nil,
			// Per M tokens: $0.20 input, $1.20 output, $0.02 cached input, $0.25 cache write.
			// Above 272K: $0.40 input, $1.80 output, $0.04 cached input, $0.50 cache write.
			pricing.TieredInfo(
				pricing.NewRates(0.20, 1.20, 0.02).WithCacheCreation(0, 0, 0.25),
				pricing.Bracket{
					MinContextTokens: 272_001,
					Rates:            pricing.NewRates(0.40, 1.80, 0.04).WithCacheCreation(0, 0, 0.50),
				},
			)),
		gpt56Entry(ModelGPT5_6Terra, catalog.ModelGPT5_6Terra, nil,
			// Per M tokens: $2.00 input, $12.00 output, $0.20 cached input, $2.50 cache write.
			// Above 272K: $4.00 input, $18.00 output, $0.40 cached input, $5.00 cache write.
			pricing.TieredInfo(
				pricing.NewRates(2.00, 12.00, 0.20).WithCacheCreation(0, 0, 2.50),
				pricing.Bracket{
					MinContextTokens: 272_001,
					Rates:            pricing.NewRates(4.00, 18.00, 0.40).WithCacheCreation(0, 0, 5.00),
				},
			)),
		// "gpt-5.6" is OpenAI's official alias for Sol.
		gpt56Entry(ModelGPT5_6Sol, catalog.ModelGPT5_6Sol, []string{ModelGPT5_6},
			// Per M tokens: $4.00 input, $20.00 output, $0.40 cached input, $5.00 cache write.
			// Above 272K: $8.00 input, $30.00 output, $0.80 cached input, $10.00 cache write.
			pricing.TieredInfo(
				pricing.NewRates(4.00, 20.00, 0.40).WithCacheCreation(0, 0, 5.00),
				pricing.Bracket{
					MinContextTokens: 272_001,
					Rates:            pricing.NewRates(8.00, 30.00, 0.80).WithCacheCreation(0, 0, 10.00),
				},
			)),

		// GPT-5.5 (May 2026 Flagship)
		{
			ID:           ModelGPT5_5,
			Model:        catalog.ModelGPT5_5,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    1_050_000, // 1.05M context window
				MaxOutputTokens:   128000,    // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-04-23"),
			},
			// Per M tokens: $5.00 input, $30.00 output, $0.50 cached input.
			// Above 272K: $10.00 input, $45.00 output, $1.00 cached input.
			Pricing: pricing.TieredInfo(
				pricing.NewRates(5.00, 30.00, 0.50),
				pricing.Bracket{
					MinContextTokens: 272_001,
					Rates:            pricing.NewRates(10.00, 45.00, 1.00),
				},
			),
		},

		// GPT-5.4 Series (March 2026 Flagship)
		{
			ID:           ModelGPT5_4,
			Model:        catalog.ModelGPT5_4,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    1_050_000, // 1.05M context window
				MaxOutputTokens:   128000,    // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-03-05"),
			},
			// Per M tokens: $2.50 input, $15.00 output, $0.25 cached input.
			// Above 272K: 2x input, 1.5x output, 2x cached (model page).
			Pricing: pricing.TieredInfo(
				pricing.NewRates(2.50, 15.00, 0.25),
				pricing.Bracket{
					MinContextTokens: 272_001,
					Rates:            pricing.NewRates(5.00, 22.50, 0.50),
				},
			),
		},
		{
			ID:           ModelGPT5_4Mini,
			Model:        catalog.ModelGPT5_4Mini,
			Capabilities: flagshipCaps,
			Modalities:   textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    400000, // 400K context window
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-03-17"),
			},
			Pricing: pricing.FlatInfo(0.75, 4.50, 0.075),
		},
		{
			ID:    ModelGPT5_4Nano,
			Model: catalog.ModelGPT5_4Nano,
			// Nano trades audio for speed but keeps image input and reasoning.
			Capabilities: nanoCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    400000, // 400K context window
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "reasoning_effort"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-03-17"),
			},
			Pricing: pricing.FlatInfo(0.20, 1.25, 0.02),
		},

		// GPT-4.1 Series (Enhanced Performance)
		{
			ID:           ModelGPT41,
			Model:        catalog.ModelGPT41,
			Capabilities: visionChatCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    1047576, // ~1M context window
				MaxOutputTokens:   32768,   // 32K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2025-04-14"),
			},
			Pricing: pricing.FlatInfo(2.00, 8.00, 0.50),
		},
		{
			ID:           ModelGPT41Mini,
			Model:        catalog.ModelGPT41Mini,
			Capabilities: visionChatCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    1047576, // ~1M context window
				MaxOutputTokens:   32768,   // 32K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2025-04-14"),
			},
			Pricing: pricing.FlatInfo(0.40, 1.60, 0.10),
		},

		// O-Series Reasoning Models
		{
			ID:           ModelO3,
			Model:        catalog.ModelO3,
			Capabilities: oSeriesCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0}, // Reasoning models prefer lower randomness
				MaxInputTokens:   200000,               // 200K context window
				MaxOutputTokens:  100000,               // 100K output tokens
				SupportedParams:  []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-04-16"),
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
				ReplacedBy: ModelGPT5_6Sol,
			},
			Pricing: pricing.FlatInfo(2.00, 8.00, 0.50),
		},
		{
			ID:           ModelO4Mini,
			Model:        catalog.ModelO4Mini,
			Capabilities: oSeriesCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   200000, // 200K context window
				MaxOutputTokens:  100000, // 100K output tokens
				SupportedParams:  []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-04-16"),
				Deprecated: catalog.MustDate("2026-04-22"),
				Retires:    catalog.MustDate("2026-10-23"),
				ReplacedBy: ModelGPT5_6Terra,
			},
			Pricing: pricing.FlatInfo(1.10, 4.40, 0.275),
		},

		// GPT-4o Series (Multimodal)
		{
			ID:    ModelGPT4O,
			Model: catalog.ModelGPT4o,
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
			},
			Modalities: textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    128000, // 128K context window
				MaxOutputTokens:   16384,  // 16K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "logprobs", "seed", "frequency_penalty", "presence_penalty"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
				ConditionalRules: []llm.ConditionalRule{
					{
						Condition: "stream_enabled",
						Disables:  []string{"logprobs"},
						Message:   "logprobs not supported with streaming",
					},
				},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2024-05-13"),
			},
			Pricing: pricing.FlatInfo(2.50, 10.00, 1.25),
		},
		{
			ID:           ModelGPT4OMini,
			Model:        catalog.ModelGPT4oMini,
			Capabilities: visionChatCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    128000, // 128K context window
				MaxOutputTokens:   16384,  // 16K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2024-07-18"),
			},
			Pricing: pricing.FlatInfo(0.15, 0.60, 0.075),
		},

		// Legacy but still supported (2025)
		{
			ID:           ModelGPT4Turbo,
			Model:        catalog.ModelGPT4Turbo,
			Capabilities: visionChatCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    128000, // 128K context window
				MaxOutputTokens:   4096,   // 4K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2023-11-06"),
				Deprecated: catalog.MustDate("2026-04-22"),
				Retires:    catalog.MustDate("2026-10-23"),
				ReplacedBy: ModelGPT5_6Sol,
			},
			Pricing: pricing.FlatInfo(10.00, 30.00, 0),
		},
		{
			ID:           ModelGPT35Turbo,
			Model:        catalog.ModelGPT35Turbo,
			Capabilities: textChatCaps,
			Modalities:   textOnly,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    16385, // 16K context window
				MaxOutputTokens:   4096,  // 4K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2023-03-01"),
				Deprecated: catalog.MustDate("2026-04-22"),
				Retires:    catalog.MustDate("2026-10-23"),
				ReplacedBy: ModelGPT5_6Terra,
			},
			Pricing: pricing.FlatInfo(0.50, 1.50, 0),
		},

		// O1 Pro - Advanced reasoning model
		{
			ID:           ModelO1Pro,
			Model:        catalog.ModelO1Pro,
			Capabilities: oSeriesCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0}, // Reasoning models prefer lower randomness
				MaxInputTokens:   200000,               // 200K context window
				MaxOutputTokens:  100000,               // 100K output tokens
				SupportedParams:  []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-03-19"),
				Deprecated: catalog.MustDate("2026-04-22"),
				Retires:    catalog.MustDate("2026-10-23"),
				ReplacedBy: ModelGPT5_6Sol,
			},
			Pricing: pricing.FlatInfo(150.00, 600.00, 75.00),
		},

		// O3 Pro - Professional-grade reasoning
		{
			ID:           ModelO3Pro,
			Model:        catalog.ModelO3Pro,
			Capabilities: oSeriesCaps,
			Modalities:   textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0}, // Reasoning models prefer lower randomness
				MaxInputTokens:   200000,               // 200K context window
				MaxOutputTokens:  100000,               // 100K output tokens
				SupportedParams:  []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-06-10"),
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
				ReplacedBy: ModelGPT5_6Sol,
			},
			Pricing: pricing.FlatInfo(20.00, 80.00, 0),
		},
	}
}

// gpt56Entry builds one GPT-5.6 family entry. Sol, Terra, and Luna share
// every capability, constraint, and lifecycle attribute — they differ only
// in identity and rates.
func gpt56Entry(id string, model catalog.ModelID, aliases []string, rates pricing.Info) catalog.Entry {
	return catalog.Entry{
		ID:      id,
		Model:   model,
		Aliases: aliases,
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         true,
			StructuredOutput: true,
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true,
		},
		Modalities: textImage,
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1_050_000,
			MaxOutputTokens:   128_000,
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		Reasoning: catalog.ReasoningSupport{
			Efforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
		},
		Life: catalog.Lifecycle{
			Available: catalog.MustDate("2026-07-09"),
		},
		Pricing: rates,
	}
}
