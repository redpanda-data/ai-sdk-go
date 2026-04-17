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

package anthropic

import (
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Model ID constants for Anthropic Claude models.
// These are model family identifiers (non-timestamped). The Anthropic API
// accepts them directly and resolves to the latest snapshot.
const (
	ModelClaudeSonnet46 = "claude-sonnet-4-6"
	ModelClaudeSonnet45 = "claude-sonnet-4-5"
	ModelClaudeHaiku45  = "claude-haiku-4-5"
	ModelClaudeOpus47   = "claude-opus-4-7"
	ModelClaudeOpus46   = "claude-opus-4-6"
	ModelClaudeOpus45   = "claude-opus-4-5"
	ModelClaudeOpus41   = "claude-opus-4-1"
)

// Effort controls the output effort level for supported models.
type Effort string

const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
	EffortXHigh  Effort = "xhigh"
	EffortMax    Effort = "max"
)

// Speed controls the inference speed mode for supported models.
type Speed string

const (
	SpeedStandard Speed = "standard"
	SpeedFast     Speed = "fast"
)

// ModelDefinition defines a Claude model with its capabilities and constraints.
type ModelDefinition struct {
	Name             string
	Label            string
	Capabilities     llm.ModelCapabilities
	Constraints      llm.ModelConstraints
	SupportedEfforts []Effort // Which effort values this model accepts
	SupportedSpeeds  []Speed  // Which speed values this model accepts
	AdaptiveThinking bool     // Whether model uses adaptive thinking by default
	Pricing          pricing.Info
}

// resolveModelFamily returns the model family key for a given model string.
// If the model string has a known family as a prefix, that family is returned.
// Otherwise the original string is returned unchanged.
// e.g., "claude-sonnet-4-5-20250929" -> "claude-sonnet-4-5"
//
//	"claude-sonnet-4-5"          -> "claude-sonnet-4-5" (unchanged)
func resolveModelFamily(model string) string {
	for family := range supportedModels {
		if strings.HasPrefix(model, family) {
			return family
		}
	}

	return model
}

// supportedModels defines all Claude models with their capabilities and constraints.
// Based on Anthropic API documentation and model specifications.
var supportedModels = map[string]ModelDefinition{
	ModelClaudeOpus47: {
		Name:  ModelClaudeOpus47,
		Label: "Claude Opus 4.7",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         false, // Anthropic doesn't have native JSON mode
			StructuredOutput: false, // Use tool calling for structured output instead
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Adaptive thinking only; use effort to bias toward more/less thinking
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 1.0},
			MaxInputTokens:   1000000, // 1M context window
			MaxOutputTokens:  128000,  // 128K output tokens
			// Opus 4.7 rejects thinking.type.enabled — thinking budget is not user-controllable.
			// Use adaptive thinking + effort to bias reasoning depth.
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "effort", "speed"},
			MutuallyExclusive: [][]string{},
		},
		SupportedEfforts: []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax},
		SupportedSpeeds:  []Speed{SpeedStandard, SpeedFast},
		AdaptiveThinking: true,
		Pricing: pricing.Info{
			InputPerMillion:       500_000_000,
			OutputPerMillion:      2_500_000_000,
			CachedInputPerMillion: 50_000_000,
		},
	},
	ModelClaudeSonnet46: {
		Name:  ModelClaudeSonnet46,
		Label: "Claude Sonnet 4.6",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         false, // Anthropic doesn't have native JSON mode
			StructuredOutput: false, // Use tool calling for structured output instead
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Extended thinking + adaptive thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0},
			MaxInputTokens:    200000, // 200K context window
			MaxOutputTokens:   64000,  // 64K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "effort", "thinking_budget"},
			MutuallyExclusive: [][]string{},
		},
		SupportedEfforts: []Effort{EffortLow, EffortMedium, EffortHigh},
		AdaptiveThinking: true,
		Pricing: pricing.Info{
			InputPerMillion:       300_000_000,
			OutputPerMillion:      1_500_000_000,
			CachedInputPerMillion: 30_000_000,
		},
	},
	ModelClaudeSonnet45: {
		Name:  ModelClaudeSonnet45,
		Label: "Claude Sonnet 4.5",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         false, // Anthropic doesn't have native JSON mode
			StructuredOutput: false, // Use tool calling for structured output instead
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Extended thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0},
			MaxInputTokens:    200000, // 200K context window
			MaxOutputTokens:   64000,  // 64K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion:       300_000_000,
			OutputPerMillion:      1_500_000_000,
			CachedInputPerMillion: 30_000_000,
		},
	},
	ModelClaudeHaiku45: {
		Name:  ModelClaudeHaiku45,
		Label: "Claude Haiku 4.5",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         false, // Anthropic doesn't have native JSON mode
			StructuredOutput: false, // Use tool calling for structured output instead
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Extended thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0},
			MaxInputTokens:    200000, // 200K context window
			MaxOutputTokens:   64000,  // 64K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion:       100_000_000,
			OutputPerMillion:      500_000_000,
			CachedInputPerMillion: 10_000_000,
		},
	},
	ModelClaudeOpus46: {
		Name:  ModelClaudeOpus46,
		Label: "Claude Opus 4.6",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         false, // Anthropic doesn't have native JSON mode
			StructuredOutput: false, // Use tool calling for structured output instead
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Extended thinking + adaptive thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0},
			MaxInputTokens:    1000000, // 1M context window (beta)
			MaxOutputTokens:   128000,  // 128K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "effort", "thinking_budget", "speed"},
			MutuallyExclusive: [][]string{},
		},
		SupportedEfforts: []Effort{EffortLow, EffortMedium, EffortHigh, EffortMax},
		SupportedSpeeds:  []Speed{SpeedStandard, SpeedFast},
		AdaptiveThinking: true,
		Pricing: pricing.Info{
			InputPerMillion:       500_000_000,
			OutputPerMillion:      2_500_000_000,
			CachedInputPerMillion: 50_000_000,
		},
	},
	ModelClaudeOpus41: {
		Name:  ModelClaudeOpus41,
		Label: "Claude Opus 4.1",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         false, // Anthropic doesn't have native JSON mode
			StructuredOutput: false, // Use tool calling for structured output instead
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Extended thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0},
			MaxInputTokens:    200000, // 200K context window
			MaxOutputTokens:   32000,  // 32K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion:       1_500_000_000,
			OutputPerMillion:      7_500_000_000,
			CachedInputPerMillion: 150_000_000,
		},
	},
	ModelClaudeOpus45: {
		Name:  ModelClaudeOpus45,
		Label: "Claude Opus 4.5",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         false, // Anthropic doesn't have native JSON mode
			StructuredOutput: false, // Use tool calling for structured output instead
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Extended thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 1.0},
			MaxInputTokens:    200000, // 200K context window
			MaxOutputTokens:   64000,  // 64K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "effort"},
			MutuallyExclusive: [][]string{},
		},
		SupportedEfforts: []Effort{EffortLow, EffortMedium, EffortHigh},
		Pricing: pricing.Info{
			InputPerMillion:       500_000_000,
			OutputPerMillion:      2_500_000_000,
			CachedInputPerMillion: 50_000_000,
		},
	},
}
