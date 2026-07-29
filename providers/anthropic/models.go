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
	ModelClaudeFable5   = "claude-fable-5"
	ModelClaudeOpus5    = "claude-opus-5"
	ModelClaudeSonnet5  = "claude-sonnet-5"
	ModelClaudeSonnet46 = "claude-sonnet-4-6"
	ModelClaudeSonnet45 = "claude-sonnet-4-5"
	ModelClaudeHaiku45  = "claude-haiku-4-5"
	ModelClaudeOpus48   = "claude-opus-4-8"
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

// Speed controls the inference speed mode for supported models. It is
// an alias of llm.Speed so catalog authors can pass Speed constants
// into pricing.Selector without casting.
type Speed = llm.Speed

const (
	SpeedStandard = llm.SpeedStandard
	SpeedFast     = llm.SpeedFast
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
// If the model string has a known family as a prefix, the longest match is
// returned (to handle families that share a common prefix, e.g.
// "claude-opus-4" vs "claude-opus-4-5"). Otherwise the original string is
// returned unchanged.
//
//	"claude-sonnet-4-5-20250929" -> "claude-sonnet-4-5"
//	"claude-sonnet-4-5"          -> "claude-sonnet-4-5" (unchanged)
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

// supportedModels defines all Claude models with their capabilities and constraints.
// Based on Anthropic API documentation and model specifications.
// claudeWireAPIs is the wire set every cataloged Claude model answers on
// api.anthropic.com: the native Messages API plus the OpenAI-compatible
// Chat Completions endpoint (Anthropic documents the compat endpoint for
// current models; verified by live calls against claude-fable-5,
// claude-sonnet-5, claude-haiku-4-5, and claude-opus-4-8, July 2026).
// Uniform today; move onto per-model entries if a model ever diverges.
var claudeWireAPIs = []llm.WireAPI{llm.WireAPIAnthropicMessages, llm.WireAPIOpenAIChatCompletions}

var supportedModels = map[string]ModelDefinition{
	ModelClaudeFable5: {
		Name:  ModelClaudeFable5,
		Label: "Claude Fable 5",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
			MaxInputTokens:  1000000, // 1M context window
			MaxOutputTokens: 128000,  // 128K output tokens
			// Fable 5 rejects thinking.type.enabled — thinking budget is not user-controllable.
			// Use adaptive thinking + effort to bias reasoning depth. No fast mode, so no "speed".
			SupportedParams:   []string{"max_tokens", "effort"},
			MutuallyExclusive: [][]string{},
		},
		SupportedEfforts: []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax},
		AdaptiveThinking: true,
		Pricing: pricing.TieredInfo(
			// Cache rates derived from Anthropic's prompt-caching multipliers
			// (5m-write = 1.25x base input, 1h-write = 2x, cache-read = 0.10x).
			pricing.NewRates(10.00, 50.00, 1.00).WithCacheCreation(12.50, 20.00, 0),
			pricing.Bracket{
				// Anthropic >200K long-context surcharge: input 2x, output 1.5x, cache 2x.
				MinContextTokens: 200_001,
				Rates:            pricing.NewRates(20.00, 75.00, 2.00).WithCacheCreation(25.00, 40.00, 0),
			},
		),
	},
	ModelClaudeOpus5: {
		Name:  ModelClaudeOpus5,
		Label: "Claude Opus 5",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         false, // Anthropic doesn't have native JSON mode
			StructuredOutput: false, // Use tool calling for structured output instead
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Thinking defaults on; use effort to bias reasoning depth
		},
		Constraints: llm.ModelConstraints{
			MaxInputTokens:  1000000, // 1M context window
			MaxOutputTokens: 128000,  // 128K output tokens
			// Opus 5 rejects thinking.type.enabled — thinking budget is not user-controllable.
			// Non-default sampling parameters are also rejected. Use adaptive
			// thinking + effort to bias reasoning depth.
			SupportedParams:   []string{"max_tokens", "effort", "speed"},
			MutuallyExclusive: [][]string{},
		},
		SupportedEfforts: []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax},
		SupportedSpeeds:  []Speed{SpeedStandard, SpeedFast},
		AdaptiveThinking: true,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		).WithOverride(
			pricing.Selector{Speed: SpeedFast},
			pricing.RateCard{
				Base: pricing.NewRates(10.00, 50.00, 1.00).
					WithCacheCreation(12.50, 20.00, 0),
			},
		),
	},
	ModelClaudeOpus48: {
		Name:  ModelClaudeOpus48,
		Label: "Claude Opus 4.8",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
			// Opus 4.8 rejects thinking.type.enabled — thinking budget is not user-controllable.
			// Use adaptive thinking + effort to bias reasoning depth.
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "effort", "speed"},
			MutuallyExclusive: [][]string{},
		},
		SupportedEfforts: []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax},
		SupportedSpeeds:  []Speed{SpeedStandard, SpeedFast},
		AdaptiveThinking: true,
		Pricing: pricing.TieredInfo(
			pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
			pricing.Bracket{
				// Anthropic >200K long-context surcharge: input 2x, output 1.5x, cache 2x.
				MinContextTokens: 200_001,
				Rates:            pricing.NewRates(10.00, 37.50, 1.00).WithCacheCreation(12.50, 20.00, 0),
			},
		).WithOverride(
			pricing.Selector{Speed: SpeedFast},
			pricing.RateCard{
				// Opus 4.8 fast mode is 3x cheaper than Opus 4.6/4.7's fast mode.
				// Cache rates derived from Anthropic's prompt-caching multipliers
				// (5m-write = 1.25x base input, 1h-write = 2x, cache-read = 0.10x).
				Base: pricing.NewRates(10.00, 50.00, 1.00).
					WithCacheCreation(12.50, 20.00, 0),
				Brackets: []pricing.Bracket{{
					// >200K long-context surcharge on fast-mode rates.
					MinContextTokens: 200_001,
					Rates:            pricing.NewRates(20.00, 75.00, 2.00).WithCacheCreation(25.00, 40.00, 0),
				}},
			},
		),
	},
	ModelClaudeOpus47: {
		Name:  ModelClaudeOpus47,
		Label: "Claude Opus 4.7",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
		AdaptiveThinking: true,
		Pricing: pricing.TieredInfo(
			pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
			pricing.Bracket{
				// Anthropic >200K long-context surcharge: input 2x, output 1.5x, cache 2x.
				MinContextTokens: 200_001,
				Rates:            pricing.NewRates(10.00, 37.50, 1.00).WithCacheCreation(12.50, 20.00, 0),
			},
		),
	},
	ModelClaudeSonnet5: {
		Name:  ModelClaudeSonnet5,
		Label: "Claude Sonnet 5",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
			// Sonnet 5 shares Opus 4.7's request surface: manual thinking budget
			// is removed (adaptive thinking + effort instead), no fast mode.
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "effort"},
			MutuallyExclusive: [][]string{},
		},
		// First Sonnet-tier model with xhigh; supports the full effort range.
		SupportedEfforts: []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax},
		AdaptiveThinking: true,
		Pricing: pricing.TieredInfo(
			// List price $3/$15 per MTok (the introductory $2/$10 through
			// 2026-08-31 is deliberately not tracked here). Cache rates from
			// Anthropic's prompt-caching multipliers (5m-write 1.25x, 1h-write 2x,
			// cache-read 0.10x of base input).
			pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
			pricing.Bracket{
				// Anthropic >200K long-context surcharge: input 2x, output 1.5x, cache 2x.
				// Matches Anthropic's published Sonnet 1M pricing ($6/$22.50 above 200K).
				MinContextTokens: 200_001,
				Rates:            pricing.NewRates(6.00, 22.50, 0.60).WithCacheCreation(7.50, 12.00, 0),
			},
		),
	},
	ModelClaudeSonnet46: {
		Name:  ModelClaudeSonnet46,
		Label: "Claude Sonnet 4.6",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
		),
	},
	ModelClaudeSonnet45: {
		Name:  ModelClaudeSonnet45,
		Label: "Claude Sonnet 4.5",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
		),
	},
	ModelClaudeHaiku45: {
		Name:  ModelClaudeHaiku45,
		Label: "Claude Haiku 4.5",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(1.00, 5.00, 0.10).WithCacheCreation(1.25, 2.00, 0),
		),
	},
	ModelClaudeOpus46: {
		Name:  ModelClaudeOpus46,
		Label: "Claude Opus 4.6",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
		Pricing: pricing.TieredInfo(
			pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
			pricing.Bracket{
				// Anthropic >200K long-context surcharge: input 2x, output 1.5x, cache 2x.
				MinContextTokens: 200_001,
				Rates:            pricing.NewRates(10.00, 37.50, 1.00).WithCacheCreation(12.50, 20.00, 0),
			},
		).WithOverride(
			pricing.Selector{Speed: SpeedFast},
			pricing.RateCard{
				Base: pricing.NewRates(30.00, 150.00, 3.00).
					WithCacheCreation(37.50, 60.00, 0),
				Brackets: []pricing.Bracket{{
					// >200K long-context surcharge on fast-mode rates.
					MinContextTokens: 200_001,
					Rates:            pricing.NewRates(60.00, 225.00, 6.00).WithCacheCreation(75.00, 120.00, 0),
				}},
			},
		),
	},
	ModelClaudeOpus41: {
		Name:  ModelClaudeOpus41,
		Label: "Claude Opus 4.1",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(15.00, 75.00, 1.50).
				WithCacheCreation(18.75, 30.00, 0),
		),
	},
	ModelClaudeOpus45: {
		Name:  ModelClaudeOpus45,
		Label: "Claude Opus 4.5",
		Capabilities: llm.ModelCapabilities{
			WireAPIs:         claudeWireAPIs,
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
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.00, 25.00, 0.50).
				WithCacheCreation(6.25, 10.00, 0),
		),
	},
}

// WireAPIsForModel reports which wire contracts an Anthropic-hosted model
// answers (llm.ModelCapabilities.WireAPIs). Snapshot IDs resolve through the
// usual family resolution. ok is false for models absent from the catalog;
// callers decide their own fallback.
func WireAPIsForModel(modelID string) (apis []llm.WireAPI, ok bool) {
	def, found := supportedModels[resolveModelFamily(modelID)]
	if !found {
		return nil, false
	}
	return def.Capabilities.WireAPIs, true
}
