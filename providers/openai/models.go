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
			ID:    ModelGPT5,
			Model: catalog.ModelGPT5,
			Label: "OpenAI GPT-5",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true,
			},
			Modalities: textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    272000, // 272K context window
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
				ReplacedBy: ModelGPT5_5,
			},
			Pricing: pricing.FlatInfo(0.625, 5.00, 0.125),
		},
		{
			ID:    ModelGPT5Mini,
			Model: catalog.ModelGPT5Mini,
			Label: "OpenAI GPT-5 Mini",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true,
			},
			Modalities: textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    272000, // 272K context window
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
				ReplacedBy: ModelGPT5_4Mini,
			},
			// $0.25 / $2.00 / $0.025 per M (input / output / cached input).
			Pricing: pricing.FlatInfo(0.25, 2.00, 0.025),
		},
		{
			ID:    ModelGPT5Nano,
			Model: catalog.ModelGPT5Nano,
			Label: "OpenAI GPT-5 Nano",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           false, // Nano typically has reduced capabilities
				Audio:            false,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        false, // Nano focuses on speed over reasoning
			},
			Modalities: textOnly,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    272000, // 272K context window
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-08-07"),
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
				ReplacedBy: ModelGPT5_4Nano,
			},
			Pricing: pricing.FlatInfo(0.05, 0.40, 0.005),
		},
		{
			ID:    ModelGPT5_1,
			Model: catalog.ModelGPT5_1,
			Label: "OpenAI GPT-5.1",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true, // Configurable: defaults to none, supports low/medium/high
			},
			Modalities: textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    272000, // 272K context window
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
			ID:    ModelGPT5_2,
			Model: catalog.ModelGPT5_2,
			Label: "OpenAI GPT-5.2 Thinking",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true,
			},
			Modalities: textImageAudio,
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
			Pricing: pricing.FlatInfo(0.875, 7.00, 0.175),
		},
		{
			// Retired 2026-08-10; entry retained (append-only catalog).
			ID:    ModelGPT5_2Instant,
			Model: catalog.ModelGPT5_2Instant,
			Label: "OpenAI GPT-5.2 Instant",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true,
			},
			Modalities: textImageAudio,
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
				Retires:    catalog.MustDate("2026-08-10"),
				ReplacedBy: ModelGPT5_6Sol,
			},
			Pricing: pricing.FlatInfo(0.875, 7.00, 0.175),
		},
		{
			ID:    ModelGPT5_2Pro,
			Model: catalog.ModelGPT5_2Pro,
			Label: "OpenAI GPT-5.2 Pro",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true,
			},
			Modalities: textImageAudio,
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
			Pricing: pricing.FlatInfo(10.50, 84.00, 0),
		},
		{
			// Retired 2026-08-10; entry retained (append-only catalog).
			ID:    ModelGPT5_3ChatLatest,
			Model: catalog.ModelGPT5_3Instant,
			Label: "OpenAI GPT-5.3 Chat Latest",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true,
			},
			Modalities: textImageAudio,
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
				Retires:    catalog.MustDate("2026-08-10"),
				ReplacedBy: ModelGPT5_6Sol,
			},
			Pricing: pricing.FlatInfo(1.75, 14.00, 0.175),
		},

		// GPT-5.6 Series
		gpt56Entry(ModelGPT5_6Luna, catalog.ModelGPT5_6Luna, "OpenAI GPT-5.6 Luna", nil,
			// Per M tokens: $0.20 input, $1.20 output, $0.02 cached input, $0.25 cache write.
			// Above 272K: $0.40 input, $1.80 output, $0.04 cached input, $0.50 cache write.
			pricing.TieredInfo(
				pricing.NewRates(0.20, 1.20, 0.02).WithCacheCreation(0, 0, 0.25),
				pricing.Bracket{
					MinContextTokens: 272_001,
					Rates:            pricing.NewRates(0.40, 1.80, 0.04).WithCacheCreation(0, 0, 0.50),
				},
			)),
		gpt56Entry(ModelGPT5_6Terra, catalog.ModelGPT5_6Terra, "OpenAI GPT-5.6 Terra", nil,
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
		gpt56Entry(ModelGPT5_6Sol, catalog.ModelGPT5_6Sol, "OpenAI GPT-5.6 Sol", []string{ModelGPT5_6},
			// Per M tokens: $5.00 input, $30.00 output, $0.50 cached input, $6.25 cache write.
			// Above 272K: $10.00 input, $45.00 output, $1.00 cached input, $12.50 cache write.
			pricing.TieredInfo(
				pricing.NewRates(5.00, 30.00, 0.50).WithCacheCreation(0, 0, 6.25),
				pricing.Bracket{
					MinContextTokens: 272_001,
					Rates:            pricing.NewRates(10.00, 45.00, 1.00).WithCacheCreation(0, 0, 12.50),
				},
			)),

		// GPT-5.5 (May 2026 Flagship)
		{
			ID:    ModelGPT5_5,
			Model: catalog.ModelGPT5_5,
			Label: "OpenAI GPT-5.5",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true,
			},
			Modalities: textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    1048576, // 1M context window
				MaxOutputTokens:   128000,  // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-04-23"),
			},
			Pricing: pricing.FlatInfo(5.00, 30.00, 0.50),
		},

		// GPT-5.4 Series (March 2026 Flagship)
		{
			ID:    ModelGPT5_4,
			Model: catalog.ModelGPT5_4,
			Label: "OpenAI GPT-5.4",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true,
			},
			Modalities: textImageAudio,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    272000, // 272K context window
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-03-05"),
			},
			Pricing: pricing.FlatInfo(2.50, 15.00, 0.25),
		},
		{
			ID:    ModelGPT5_4Mini,
			Model: catalog.ModelGPT5_4Mini,
			Label: "OpenAI GPT-5.4 Mini",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				Audio:            true,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        true,
			},
			Modalities: textImageAudio,
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
			Label: "OpenAI GPT-5.4 Nano",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           false, // Nano has reduced capabilities
				Audio:            false,
				MultiTurn:        true,
				SystemPrompts:    true,
				Reasoning:        false, // Nano focuses on speed over reasoning
			},
			Modalities: textOnly,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    400000, // 400K context window
				MaxOutputTokens:   128000, // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-03-17"),
			},
			Pricing: pricing.FlatInfo(0.20, 1.25, 0.02),
		},

		// GPT-4.1 Series (Enhanced Performance)
		{
			ID:    ModelGPT41,
			Model: catalog.ModelGPT41,
			Label: "OpenAI GPT-4.1",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				MultiTurn:        true,
				SystemPrompts:    true,
			},
			Modalities: textImage,
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
			ID:    ModelGPT41Mini,
			Model: catalog.ModelGPT41Mini,
			Label: "OpenAI GPT-4.1 Mini",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				MultiTurn:        true,
				SystemPrompts:    true,
			},
			Modalities: textImage,
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
			ID:    ModelO3,
			Model: catalog.ModelO3,
			Label: "OpenAI o3 (Reasoning)",
			Capabilities: llm.ModelCapabilities{
				Streaming:     true,
				Tools:         true,
				Vision:        true, // Supports "thinking with images"
				MultiTurn:     true,
				SystemPrompts: true,
				Reasoning:     true,
			},
			Modalities: textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0}, // Reasoning models prefer lower randomness
				MaxInputTokens:    200000,               // 200K context window
				MaxOutputTokens:   100000,               // 100K output tokens
				SupportedParams:   []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-04-16"),
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
				// OpenAI's recommended replacement (gpt-5.5) applies to the
				// snapshot family; derived succession covers the series view.
			},
			Pricing: pricing.FlatInfo(2.00, 8.00, 0.50),
		},
		{
			ID:    ModelO4Mini,
			Model: catalog.ModelO4Mini,
			Label: "OpenAI o4-mini (Fast Reasoning)",
			Capabilities: llm.ModelCapabilities{
				Streaming:     true,
				Tools:         true,
				Vision:        true,
				MultiTurn:     true,
				SystemPrompts: true,
				Reasoning:     true,
			},
			Modalities: textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0},
				MaxInputTokens:    200000, // 200K context window
				MaxOutputTokens:   100000, // 100K output tokens
				SupportedParams:   []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-04-16"),
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
			},
			Pricing: pricing.FlatInfo(1.10, 4.40, 0.275),
		},

		// GPT-4o Series (Multimodal)
		{
			ID:    ModelGPT4O,
			Model: catalog.ModelGPT4o,
			Label: "OpenAI GPT-4o",
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
			ID:    ModelGPT4OMini,
			Model: catalog.ModelGPT4oMini,
			Label: "OpenAI GPT-4o Mini",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				MultiTurn:        true,
				SystemPrompts:    true,
			},
			Modalities: textImage,
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
			ID:    ModelGPT4Turbo,
			Model: catalog.ModelGPT4Turbo,
			Label: "OpenAI GPT-4 Turbo",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				Vision:           true,
				MultiTurn:        true,
				SystemPrompts:    true,
			},
			Modalities: textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    128000, // 128K context window
				MaxOutputTokens:   4096,   // 4K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2023-11-06"),
			},
			Pricing: pricing.FlatInfo(5.00, 15.00, 0),
		},
		{
			ID:    ModelGPT35Turbo,
			Model: catalog.ModelGPT35Turbo,
			Label: "OpenAI GPT-3.5 Turbo",
			Capabilities: llm.ModelCapabilities{
				Streaming:        true,
				Tools:            true,
				JSONMode:         true,
				StructuredOutput: true,
				MultiTurn:        true,
				SystemPrompts:    true,
				// No vision or audio
			},
			Modalities: textOnly,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 2.0},
				MaxInputTokens:    16385, // 16K context window
				MaxOutputTokens:   4096,  // 4K output tokens
				SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
				MutuallyExclusive: [][]string{{"temperature", "top_p"}},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2023-03-01"),
				Retires:    catalog.MustDate("2026-10-23"),
				ReplacedBy: ModelGPT5_6Terra,
			},
			Pricing: pricing.FlatInfo(0.50, 1.50, 0),
		},

		// O1 Pro - Advanced reasoning model
		{
			ID:    ModelO1Pro,
			Model: catalog.ModelO1Pro,
			Label: "OpenAI o1-pro (Advanced Reasoning)",
			Capabilities: llm.ModelCapabilities{
				Streaming:     true,
				Tools:         true,
				Vision:        true,
				MultiTurn:     true,
				SystemPrompts: true,
				Reasoning:     true,
			},
			Modalities: textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0}, // Reasoning models prefer lower randomness
				MaxInputTokens:    200000,               // 200K context window
				MaxOutputTokens:   100000,               // 100K output tokens
				SupportedParams:   []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2025-03-19"),
			},
			Pricing: pricing.FlatInfo(150.00, 600.00, 75.00),
		},

		// O3 Pro - Professional-grade reasoning
		{
			ID:    ModelO3Pro,
			Model: catalog.ModelO3Pro,
			Label: "OpenAI o3-pro (Professional Reasoning)",
			Capabilities: llm.ModelCapabilities{
				Streaming:     true,
				Tools:         true,
				Vision:        true,
				MultiTurn:     true,
				SystemPrompts: true,
				Reasoning:     true,
			},
			Modalities: textImage,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0}, // Reasoning models prefer lower randomness
				MaxInputTokens:    200000,               // 200K context window
				MaxOutputTokens:   100000,               // 100K output tokens
				SupportedParams:   []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-06-10"),
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
			},
			Pricing: pricing.FlatInfo(20.00, 80.00, 0),
		},
	}
}

// gpt56Entry builds one GPT-5.6 family entry. Sol, Terra, and Luna share
// every capability, constraint, and lifecycle attribute — they differ only
// in identity and rates.
func gpt56Entry(id string, model catalog.ModelID, label string, aliases []string, rates pricing.Info) catalog.Entry {
	return catalog.Entry{
		ID:      id,
		Model:   model,
		Label:   label,
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
