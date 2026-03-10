package bedrock

import (
	"sort"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// modelFamilies is sorted by descending length for deterministic longest-prefix matching.
var modelFamilies = buildModelFamilies()

func buildModelFamilies() []string {
	families := make([]string, 0, len(supportedModels))
	for family := range supportedModels {
		families = append(families, family)
	}

	sort.Slice(families, func(i, j int) bool {
		return len(families[i]) > len(families[j])
	})

	return families
}

// Model family constants for Claude models on Bedrock.
// These are the canonical short names used as map keys.
const (
	ModelClaudeSonnet46 = "claude-sonnet-4-6"
	ModelClaudeSonnet45 = "claude-sonnet-4-5"
	ModelClaudeHaiku45  = "claude-haiku-4-5"
	ModelClaudeOpus46   = "claude-opus-4-6"
	ModelClaudeOpus45   = "claude-opus-4-5"
	ModelClaudeOpus41   = "claude-opus-4-1"
)

// ModelDefinition defines a model with its capabilities and constraints.
type ModelDefinition struct {
	Name         string
	Label        string
	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints
}

// resolveModelFamily extracts the family key from any Bedrock model ID format.
//
// Supports multiple formats:
//
//	"claude-sonnet-4-6"                           → "claude-sonnet-4-6" (direct)
//	"eu.anthropic.claude-sonnet-4-6"              → "claude-sonnet-4-6" (cross-region inference profile)
//	"global.anthropic.claude-opus-4-6-v1"         → "claude-opus-4-6"  (global inference profile)
//	"eu.anthropic.claude-sonnet-4-5-20250929-v1:0"→ "claude-sonnet-4-5" (versioned)
//	"anthropic.claude-sonnet-4-6-v2:0"            → "claude-sonnet-4-6" (provider prefix)
//
// Algorithm:
//  1. Strip region + provider prefix (everything through "anthropic.")
//  2. Match remaining string against known family keys using prefix matching
//  3. If no match, return original string (caller rejects unknown models)
func resolveModelFamily(model string) string {
	for _, family := range modelFamilies {
		if strings.HasPrefix(model, family) {
			return family
		}

		if strings.HasPrefix(model, "anthropic."+family) ||
			strings.Contains(model, ".anthropic."+family) {
			return family
		}
	}

	return model
}

// supportedModels defines Claude models available on Bedrock via the Converse API.
// Standard features only — no Anthropic-specific thinking/effort/speed.
var supportedModels = map[string]ModelDefinition{
	ModelClaudeSonnet46: {
		Name:  ModelClaudeSonnet46,
		Label: "Claude Sonnet 4.6",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 1.0},
			MaxInputTokens:   200000,
			MaxOutputTokens:  64000,
			SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
		},
	},
	ModelClaudeSonnet45: {
		Name:  ModelClaudeSonnet45,
		Label: "Claude Sonnet 4.5",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 1.0},
			MaxInputTokens:   200000,
			MaxOutputTokens:  64000,
			SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
		},
	},
	ModelClaudeHaiku45: {
		Name:  ModelClaudeHaiku45,
		Label: "Claude Haiku 4.5",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 1.0},
			MaxInputTokens:   200000,
			MaxOutputTokens:  64000,
			SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
		},
	},
	ModelClaudeOpus46: {
		Name:  ModelClaudeOpus46,
		Label: "Claude Opus 4.6",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 1.0},
			MaxInputTokens:   1000000,
			MaxOutputTokens:  128000,
			SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
		},
	},
	ModelClaudeOpus45: {
		Name:  ModelClaudeOpus45,
		Label: "Claude Opus 4.5",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 1.0},
			MaxInputTokens:   200000,
			MaxOutputTokens:  64000,
			SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
		},
	},
	ModelClaudeOpus41: {
		Name:  ModelClaudeOpus41,
		Label: "Claude Opus 4.1",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 1.0},
			MaxInputTokens:   200000,
			MaxOutputTokens:  32000,
			SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
		},
	},
}
