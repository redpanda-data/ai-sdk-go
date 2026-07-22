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
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// ModelDefinition defines an OpenAI model with its capabilities and constraints.
type ModelDefinition struct {
	Name                      string
	Label                     string
	Capabilities              llm.ModelCapabilities
	Constraints               llm.ModelConstraints
	SupportedReasoningEfforts []ReasoningEffort // Ascending order: safest/lowest first
	Pricing                   pricing.Info      // Cost per million tokens (microcents)
}

func regionalFlatInfo(defaultRates, regionalRates pricing.Rates) pricing.Info {
	return pricing.FlatInfoFromRates(defaultRates).
		WithOverride(pricing.Selector{Region: "us"}, pricing.RateCard{Base: regionalRates}).
		WithOverride(pricing.Selector{Region: "eu"}, pricing.RateCard{Base: regionalRates})
}

var modelAliases = map[string]string{
	ModelGPT5_6: ModelGPT5_6Sol,
}

// resolveModelFamily returns the model family key for a given model string.
// If the model string has a known family as a prefix, that family is returned
// (longest match wins). Otherwise the original string is returned unchanged.
//
// Unlike Anthropic and Bedrock, the OpenAI SDK has no built-in alias
// resolution, so timestamped snapshot IDs like "o3-2025-04-16" are not
// recognized. This function bridges that gap:
//
//	"o3-2025-04-16"  -> "o3"
//	"gpt-4o-2024-11-20" -> "gpt-4o"
//	"gpt-4o"            -> "gpt-4o" (unchanged, exact match)
func resolveModelFamily(model string) string {
	if family, ok := modelAliases[model]; ok {
		return family
	}

	best := ""

	for family := range supportedModels {
		if strings.HasPrefix(model, family) && len(family) > len(best) {
			best = family
		}
	}

	if best != "" {
		for alias := range modelAliases {
			if strings.HasPrefix(model, alias+"-") && !strings.HasPrefix(best, alias+"-") {
				return model
			}
		}

		return best
	}

	return model
}

// supportedModels defines all current OpenAI models with their constraints.
// Based on current OpenAI API documentation and model specifications.
// When adding a new model, both capabilities and constraints must be defined here.
var supportedModels = map[string]ModelDefinition{
	// GPT-5 Series (2025 Flagship)
	ModelGPT5: {
		Name:  ModelGPT5,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    272000, // 272K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		Pricing:                   pricing.FlatInfo(0.625, 5.00, 0.125),
	},
	ModelGPT5Mini: {
		Name:  ModelGPT5Mini,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    272000, // 272K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		// $0.25 / $2.00 / $0.025 per M (input / output / cached input).
		Pricing: pricing.FlatInfo(0.25, 2.00, 0.025),
	},
	ModelGPT5Nano: {
		Name:  ModelGPT5Nano,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    272000, // 272K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		Pricing: pricing.FlatInfo(0.05, 0.40, 0.005),
	},
	ModelGPT5_1: {
		Name:  ModelGPT5_1,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    272000, // 272K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		// $1.25 / $10.00 / $0.125 per M (input / output / cached input).
		Pricing: pricing.FlatInfo(1.25, 10.00, 0.125),
	},
	ModelGPT5_2: {
		Name:  ModelGPT5_2,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    400000, // 400K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
		Pricing:                   pricing.FlatInfo(0.875, 7.00, 0.175),
	},
	ModelGPT5_2Instant: {
		Name:  ModelGPT5_2Instant,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    400000, // 400K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortMedium}, // Instant variant only supports medium
		Pricing:                   pricing.FlatInfo(0.875, 7.00, 0.175),
	},
	ModelGPT5_2Pro: {
		Name:  ModelGPT5_2Pro,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    400000, // 400K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh}, // Pro variant starts at medium
		Pricing:                   pricing.FlatInfo(10.50, 84.00, 0),
	},

	// GPT-5.3 Series
	ModelGPT5_3ChatLatest: {
		Name:  ModelGPT5_3ChatLatest,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    400000, // 400K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortMedium}, // Chat-latest only supports medium
		Pricing:                   pricing.FlatInfo(1.75, 14.00, 0.175),
	},
	ModelGPT5_3Codex: {
		Name:  ModelGPT5_3Codex,
		Label: "OpenAI GPT-5.3 Codex",
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
		Constraints: llm.ModelConstraints{
			MaxInputTokens:    400_000,
			MaxOutputTokens:   128_000,
			SupportedParams:   []string{"max_tokens", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
		// $1.75 / $14.00 / $0.175 per M (input / output / cached input).
		Pricing: pricing.FlatInfo(1.75, 14.00, 0.175),
	},

	// GPT-5.6 Series
	ModelGPT5_6Luna: {
		Name:  ModelGPT5_6Luna,
		Label: "OpenAI GPT-5.6 Luna",
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1_050_000,
			MaxOutputTokens:   128_000,
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
		// Per M tokens: $1.00 input, $6.00 output, $0.10 cached input, $1.25 cache write.
		// Above 272K: $2.00 input, $9.00 output, $0.20 cached input, $2.50 cache write.
		Pricing: pricing.TieredInfo(
			pricing.NewRates(1.00, 6.00, 0.10).WithCacheCreation(0, 0, 1.25),
			pricing.Bracket{
				MinContextTokens: 272_001,
				Rates:            pricing.NewRates(2.00, 9.00, 0.20).WithCacheCreation(0, 0, 2.50),
			},
		),
	},
	ModelGPT5_6Terra: {
		Name:  ModelGPT5_6Terra,
		Label: "OpenAI GPT-5.6 Terra",
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1_050_000,
			MaxOutputTokens:   128_000,
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
		// Per M tokens: $2.50 input, $15.00 output, $0.25 cached input, $3.125 cache write.
		// Above 272K: $5.00 input, $22.50 output, $0.50 cached input, $6.25 cache write.
		Pricing: pricing.TieredInfo(
			pricing.NewRates(2.50, 15.00, 0.25).WithCacheCreation(0, 0, 3.125),
			pricing.Bracket{
				MinContextTokens: 272_001,
				Rates:            pricing.NewRates(5.00, 22.50, 0.50).WithCacheCreation(0, 0, 6.25),
			},
		),
	},
	ModelGPT5_6Sol: {
		Name:  ModelGPT5_6Sol,
		Label: "OpenAI GPT-5.6 Sol",
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1_050_000,
			MaxOutputTokens:   128_000,
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
		// Per M tokens: $5.00 input, $30.00 output, $0.50 cached input, $6.25 cache write.
		// Above 272K: $10.00 input, $45.00 output, $1.00 cached input, $12.50 cache write.
		Pricing: pricing.TieredInfo(
			pricing.NewRates(5.00, 30.00, 0.50).WithCacheCreation(0, 0, 6.25),
			pricing.Bracket{
				MinContextTokens: 272_001,
				Rates:            pricing.NewRates(10.00, 45.00, 1.00).WithCacheCreation(0, 0, 12.50),
			},
		),
	},

	// GPT-5.5 (May 2026 Flagship)
	ModelGPT5_5: {
		Name:  ModelGPT5_5,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1048576, // 1M context window
			MaxOutputTokens:   128000,  // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
		// Regional processing has a 10% premium.
		Pricing: regionalFlatInfo(
			pricing.NewRates(5.00, 30.00, 0.50),
			pricing.NewRates(5.50, 33.00, 0.55),
		),
	},
	ModelGPT5_5Pro: {
		Name:  ModelGPT5_5Pro,
		Label: "OpenAI GPT-5.5 Pro",
		Capabilities: llm.ModelCapabilities{
			Tools:            true,
			JSONMode:         true,
			StructuredOutput: true,
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true,
		},
		Constraints: llm.ModelConstraints{
			MaxInputTokens:    1_050_000,
			MaxOutputTokens:   128_000,
			SupportedParams:   []string{"max_tokens", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
		// Regional processing has a 10% premium.
		Pricing: regionalFlatInfo(
			pricing.NewRates(30.00, 180.00, 0),
			pricing.NewRates(33.00, 198.00, 0),
		),
	},

	// GPT-5.4 Series (March 2026 Flagship)
	ModelGPT5_4: {
		Name:  ModelGPT5_4,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    272000, // 272K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
		// Regional processing has a 10% premium.
		Pricing: regionalFlatInfo(
			pricing.NewRates(2.50, 15.00, 0.25),
			pricing.NewRates(2.75, 16.50, 0.275),
		),
	},
	ModelGPT5_4Mini: {
		Name:  ModelGPT5_4Mini,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    400000, // 400K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh},
		// Regional processing has a 10% premium.
		Pricing: regionalFlatInfo(
			pricing.NewRates(0.75, 4.50, 0.075),
			pricing.NewRates(0.825, 4.95, 0.0825),
		),
	},
	ModelGPT5_4Nano: {
		Name:  ModelGPT5_4Nano,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    400000, // 400K context window
			MaxOutputTokens:   128000, // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		// Regional processing has a 10% premium.
		Pricing: regionalFlatInfo(
			pricing.NewRates(0.20, 1.25, 0.02),
			pricing.NewRates(0.22, 1.375, 0.022),
		),
	},

	// GPT-4.1 Series (Enhanced Performance)
	ModelGPT41: {
		Name:  ModelGPT41,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1047576, // ~1M context window
			MaxOutputTokens:   32768,   // 32K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		Pricing: pricing.FlatInfo(2.00, 8.00, 0.50),
	},
	ModelGPT41Mini: {
		Name:  ModelGPT41Mini,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1047576, // ~1M context window
			MaxOutputTokens:   32768,   // 32K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		Pricing: pricing.FlatInfo(0.40, 1.60, 0.10),
	},

	// O-Series Reasoning Models
	ModelO3: {
		Name:  ModelO3,
		Label: "OpenAI o3 (Reasoning)",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true, // Supports "thinking with images"
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0}, // Reasoning models prefer lower randomness
			MaxInputTokens:    200000,               // 200K context window
			MaxOutputTokens:   100000,               // 100K output tokens
			SupportedParams:   []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		Pricing:                   pricing.FlatInfo(2.00, 8.00, 0.50),
	},
	ModelO4Mini: {
		Name:  ModelO4Mini,
		Label: "OpenAI o4-mini (Fast Reasoning)",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0},
			MaxInputTokens:    200000, // 200K context window
			MaxOutputTokens:   100000, // 100K output tokens
			SupportedParams:   []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		Pricing:                   pricing.FlatInfo(1.10, 4.40, 0.275),
	},

	// GPT-4o Series (Multimodal)
	ModelGPT4O: {
		Name:  ModelGPT4O,
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
		Pricing: pricing.FlatInfo(2.50, 10.00, 1.25),
	},
	ModelGPT4OMini: {
		Name:  ModelGPT4OMini,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    128000, // 128K context window
			MaxOutputTokens:   16384,  // 16K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		Pricing: pricing.FlatInfo(0.15, 0.60, 0.075),
	},

	// Legacy but still supported (2025)
	ModelGPT4Turbo: {
		Name:  ModelGPT4Turbo,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    128000, // 128K context window
			MaxOutputTokens:   4096,   // 4K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		Pricing: pricing.FlatInfo(5.00, 15.00, 0),
	},
	ModelGPT35Turbo: {
		Name:  ModelGPT35Turbo,
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
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    16385, // 16K context window
			MaxOutputTokens:   4096,  // 4K output tokens
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
		Pricing: pricing.FlatInfo(0.50, 1.50, 0),
	},

	// O1 Pro - Advanced reasoning model
	ModelO1Pro: {
		Name:  ModelO1Pro,
		Label: "OpenAI o1-pro (Advanced Reasoning)",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0}, // Reasoning models prefer lower randomness
			MaxInputTokens:    200000,               // 200K context window
			MaxOutputTokens:   100000,               // 100K output tokens
			SupportedParams:   []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		Pricing:                   pricing.FlatInfo(150.00, 600.00, 75.00),
	},

	// O3 Pro - Professional-grade reasoning
	ModelO3Pro: {
		Name:  ModelO3Pro,
		Label: "OpenAI o3-pro (Professional Reasoning)",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0}, // Reasoning models prefer lower randomness
			MaxInputTokens:    200000,               // 200K context window
			MaxOutputTokens:   100000,               // 100K output tokens
			SupportedParams:   []string{"temperature", "max_tokens", "reasoning_effort", "reasoning_summary"},
			MutuallyExclusive: [][]string{},
		},
		SupportedReasoningEfforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		Pricing:                   pricing.FlatInfo(20.00, 80.00, 0),
	},
}
