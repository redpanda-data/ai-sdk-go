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
	"sync"

	"github.com/redpanda-data/ai-sdk-go/catalog"
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

	// ModelClaudeOpus41 is Claude Opus 4.1.
	//
	// Deprecated: retired by Anthropic on 2026-08-05; requests fail. Use
	// [ModelClaudeOpus48]. The catalog entry remains so historical usage
	// stays priceable.
	ModelClaudeOpus41 = "claude-opus-4-1"
)

// ReasoningEffort controls how much work a model spends on reasoning
// (sent as the "effort" field of Anthropic's output_config). It is an
// alias of [llm.ReasoningEffort] so effort values are portable across
// provider packages; the constants below declare the values Claude models
// accept. Which subset a specific model supports is validated against the
// model catalog in NewModel.
type ReasoningEffort = llm.ReasoningEffort

const (
	// ReasoningEffortLow biases the model toward fast, shallow reasoning.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium is the balanced middle setting.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh biases the model toward deep reasoning.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh spends very high effort (frontier models, Opus 4.7+).
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
	// ReasoningEffortMax removes all effort ceilings (frontier models, Opus 4.6+).
	ReasoningEffortMax ReasoningEffort = "max"
)

// Speed controls the inference speed mode for supported models. It is
// an alias of llm.Speed so catalog authors can pass Speed constants
// into pricing.Selector without casting.
type Speed = llm.Speed

// Speed values Claude models accept.
const (
	SpeedStandard = llm.SpeedStandard
	SpeedFast     = llm.SpeedFast
)

var catalogOnce = sync.OnceValue(func() *catalog.Catalog {
	return catalog.MustNew("anthropic", entries())
})

// Catalog returns the validated Anthropic model catalog: every offering
// with its capabilities, constraints, modalities, reasoning controls,
// pricing, and lifecycle. The catalog is immutable and shared; all reads
// return deep copies.
func Catalog() *catalog.Catalog {
	return catalogOnce()
}

// claudeCaps is the capability set shared by every Claude model:
// Anthropic has no native JSON mode or structured output (use tool
// calling instead), and every catalogued Claude is multimodal-in.
var claudeCaps = llm.ModelCapabilities{
	Streaming:     true,
	Tools:         true,
	Vision:        true,
	MultiTurn:     true,
	SystemPrompts: true,
	Reasoning:     true,
}

// claudeModalities is shared by every catalogued Claude model: text,
// image, and PDF inputs; text output.
var claudeModalities = catalog.Modalities{
	Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityDocument},
	Output: []catalog.Modality{catalog.ModalityText},
}

// entries returns the authored Anthropic catalog.
//
// Lifecycle sourcing: Anthropic's model-deprecations page
// (https://platform.claude.com/docs/en/about-claude/model-deprecations).
// Anthropic publishes tentative retirement dates as "not sooner than"
// floors — those are RetirementNotBefore, never Retires. Available is the
// first-party launch date (Anthropic is the launch platform, so it equals
// the model's release date).
//
// The catalog is append-only: retired models keep their entries (with
// Retires in the past) so historical usage stays priceable and the
// failure stays explainable.
func entries() []catalog.Entry {
	return []catalog.Entry{
		{
			ID:           ModelClaudeFable5,
			Model:        catalog.ModelClaudeFable5,
			Label:        "Claude Fable 5",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000, // 1M context window
				MaxOutputTokens: 128000,  // 128K output tokens
				// Fable 5 rejects thinking.type.enabled — thinking budget is not
				// user-controllable — and rejects non-default sampling parameters.
				// Use adaptive thinking + effort to bias reasoning depth.
				SupportedParams:   []string{"max_tokens", "reasoning_effort"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
				Adaptive: true,
			},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2026-06-07"),
				RetirementNotBefore: catalog.MustDate("2027-06-09"),
			},
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
		{
			ID:           ModelClaudeOpus5,
			Model:        catalog.ModelClaudeOpus5,
			Label:        "Claude Opus 5",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000, // 1M context window
				MaxOutputTokens: 128000,  // 128K output tokens
				// Opus 5 rejects thinking.type.enabled — thinking budget is not
				// user-controllable. Non-default sampling parameters are also
				// rejected. Use adaptive thinking + effort to bias reasoning depth.
				SupportedParams:   []string{"max_tokens", "reasoning_effort", "speed"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
				Adaptive: true,
			},
			Speeds: []Speed{SpeedStandard, SpeedFast},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2026-07-24"),
				RetirementNotBefore: catalog.MustDate("2027-07-24"),
			},
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
		{
			ID:           ModelClaudeOpus48,
			Model:        catalog.ModelClaudeOpus48,
			Label:        "Claude Opus 4.8",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   1000000, // 1M context window
				MaxOutputTokens:  128000,  // 128K output tokens
				// Opus 4.8 rejects thinking.type.enabled — thinking budget is not
				// user-controllable. Use adaptive thinking + effort to bias
				// reasoning depth.
				SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "speed"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
				Adaptive: true,
			},
			Speeds: []Speed{SpeedStandard, SpeedFast},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2026-05-28"),
				RetirementNotBefore: catalog.MustDate("2027-05-28"),
			},
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
		{
			ID:           ModelClaudeOpus47,
			Model:        catalog.ModelClaudeOpus47,
			Label:        "Claude Opus 4.7",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   1000000, // 1M context window
				MaxOutputTokens:  128000,  // 128K output tokens
				// Opus 4.7 rejects thinking.type.enabled — thinking budget is not
				// user-controllable. Use adaptive thinking + effort to bias
				// reasoning depth.
				SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "speed"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
				Adaptive: true,
			},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2026-04-14"),
				RetirementNotBefore: catalog.MustDate("2027-04-16"),
			},
			Pricing: pricing.TieredInfo(
				pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
				pricing.Bracket{
					// Anthropic >200K long-context surcharge: input 2x, output 1.5x, cache 2x.
					MinContextTokens: 200_001,
					Rates:            pricing.NewRates(10.00, 37.50, 1.00).WithCacheCreation(12.50, 20.00, 0),
				},
			),
		},
		{
			ID:           ModelClaudeSonnet5,
			Model:        catalog.ModelClaudeSonnet5,
			Label:        "Claude Sonnet 5",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   1000000, // 1M context window
				MaxOutputTokens:  128000,  // 128K output tokens
				// Sonnet 5 shares Opus 4.7's request surface: manual thinking budget
				// is removed (adaptive thinking + effort instead), no fast mode.
				SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				// First Sonnet-tier model with xhigh; supports the full effort range.
				Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
				Adaptive: true,
			},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2026-06-29"),
				RetirementNotBefore: catalog.MustDate("2027-06-30"),
			},
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
		{
			ID:           ModelClaudeSonnet46,
			Model:        catalog.ModelClaudeSonnet46,
			Label:        "Claude Sonnet 4.6",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0},
				MaxInputTokens:    200000, // 200K context window
				MaxOutputTokens:   64000,  // 64K output tokens
				SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "thinking_budget"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
				Adaptive: true,
				Budget:   true,
			},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2026-02-17"),
				RetirementNotBefore: catalog.MustDate("2027-02-17"),
			},
			Pricing: pricing.FlatInfoFromRates(
				pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
			),
		},
		{
			ID:           ModelClaudeSonnet45,
			Model:        catalog.ModelClaudeSonnet45,
			Label:        "Claude Sonnet 4.5",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0},
				MaxInputTokens:    200000, // 200K context window
				MaxOutputTokens:   64000,  // 64K output tokens
				SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens"},
				MutuallyExclusive: [][]string{},
			},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2025-09-29"),
				RetirementNotBefore: catalog.MustDate("2026-09-29"),
			},
			Pricing: pricing.FlatInfoFromRates(
				pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
			),
		},
		{
			ID:           ModelClaudeHaiku45,
			Model:        catalog.ModelClaudeHaiku45,
			Label:        "Claude Haiku 4.5",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0},
				MaxInputTokens:    200000, // 200K context window
				MaxOutputTokens:   64000,  // 64K output tokens
				SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens"},
				MutuallyExclusive: [][]string{},
			},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2025-10-15"),
				RetirementNotBefore: catalog.MustDate("2026-10-15"),
			},
			Pricing: pricing.FlatInfoFromRates(
				pricing.NewRates(1.00, 5.00, 0.10).WithCacheCreation(1.25, 2.00, 0),
			),
		},
		{
			ID:           ModelClaudeOpus46,
			Model:        catalog.ModelClaudeOpus46,
			Label:        "Claude Opus 4.6",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0},
				MaxInputTokens:    1000000, // 1M context window (beta)
				MaxOutputTokens:   128000,  // 128K output tokens
				SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "thinking_budget", "speed"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortMax},
				Adaptive: true,
				Budget:   true,
			},
			Speeds: []Speed{SpeedStandard, SpeedFast},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2026-02-04"),
				RetirementNotBefore: catalog.MustDate("2027-02-05"),
			},
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
		{
			ID:           ModelClaudeOpus45,
			Model:        catalog.ModelClaudeOpus45,
			Label:        "Claude Opus 4.5",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0},
				MaxInputTokens:    200000, // 200K context window
				MaxOutputTokens:   64000,  // 64K output tokens
				SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort"},
				MutuallyExclusive: [][]string{},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts: []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
			},
			Life: catalog.Lifecycle{
				Available:           catalog.MustDate("2025-11-24"),
				RetirementNotBefore: catalog.MustDate("2026-11-24"),
			},
			Pricing: pricing.FlatInfoFromRates(
				pricing.NewRates(5.00, 25.00, 0.50).
					WithCacheCreation(6.25, 10.00, 0),
			),
		},
		{
			// Retired 2026-08-05. The entry stays: the catalog is
			// append-only so historical usage remains priceable and the
			// failure mode ("retired", not "unknown model") stays
			// explainable.
			ID:           ModelClaudeOpus41,
			Model:        catalog.ModelClaudeOpus41,
			Label:        "Claude Opus 4.1",
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange:  [2]float64{0.0, 1.0},
				MaxInputTokens:    200000, // 200K context window
				MaxOutputTokens:   32000,  // 32K output tokens
				SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens"},
				MutuallyExclusive: [][]string{},
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-08-05"),
				Deprecated: catalog.MustDate("2026-06-05"),
				Retires:    catalog.MustDate("2026-08-05"),
				ReplacedBy: ModelClaudeOpus48,
			},
			Pricing: pricing.FlatInfoFromRates(
				pricing.NewRates(15.00, 75.00, 1.50).
					WithCacheCreation(18.75, 30.00, 0),
			),
		},
	}
}
