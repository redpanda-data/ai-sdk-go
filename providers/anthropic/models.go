package anthropic

import (
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Model ID constants for Anthropic Claude models.
// These are model family identifiers (non-timestamped). The Anthropic API
// accepts them directly and resolves to the latest snapshot.
const (
	ModelClaudeSonnet46 = "claude-sonnet-4-6"
	ModelClaudeSonnet45 = "claude-sonnet-4-5"
	ModelClaudeHaiku45  = "claude-haiku-4-5"
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
	},
}
