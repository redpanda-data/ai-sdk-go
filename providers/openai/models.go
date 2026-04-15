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
)

// ModelDefinition defines an OpenAI model with its capabilities and constraints.
type ModelDefinition struct {
	Name                      string
	Label                     string
	Capabilities              llm.ModelCapabilities
	Constraints               llm.ModelConstraints
	SupportedReasoningEfforts []ReasoningEffort // Ascending order: safest/lowest first
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
	best := ""
	for family := range supportedModels {
		if strings.HasPrefix(model, family) && len(family) > len(best) {
			best = family
		}
	}
	if best != "" {
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
	},
}
