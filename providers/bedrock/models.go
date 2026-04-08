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

// Model ID constants for Claude models on Bedrock.
// These are real Bedrock model IDs that can be passed directly to NewModel.
const (
	ModelClaudeSonnet46 = "anthropic.claude-sonnet-4-6"
	ModelClaudeSonnet45 = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	ModelClaudeHaiku45  = "anthropic.claude-haiku-4-5-20251001-v1:0"
	ModelClaudeOpus46   = "anthropic.claude-opus-4-6-v1"
	ModelClaudeOpus45   = "anthropic.claude-opus-4-5-20251101-v1:0"
)

// familyClaudeSonnet46 and siblings are short family keys used internally
// for longest-prefix matching in resolveModelFamily.
const (
	familyClaudeSonnet46 = "claude-sonnet-4-6"
	familyClaudeSonnet45 = "claude-sonnet-4-5"
	familyClaudeHaiku45  = "claude-haiku-4-5"
	familyClaudeOpus46   = "claude-opus-4-6"
	familyClaudeOpus45   = "claude-opus-4-5"
)

// ModelDefinition defines a model with its capabilities and constraints.
type ModelDefinition struct {
	Name         string // Real Bedrock model ID (e.g. "anthropic.claude-sonnet-4-5-20250929-v1:0")
	Label        string
	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints
}

// inferenceProfileRegion maps an AWS region to the Bedrock inference profile
// geographic prefix (e.g. "us-east-1" → "us", "eu-west-1" → "eu").
func inferenceProfileRegion(region string) string {
	if idx := strings.IndexByte(region, '-'); idx > 0 {
		return region[:idx]
	}

	return "us"
}

// hasRegionPrefix reports whether a model ID already contains a region
// inference-profile prefix (e.g. "us.anthropic.claude-sonnet-4-6").
// It checks for the "{region}.{provider}." pattern by counting dot-separated
// segments: bare model IDs like "anthropic.claude-sonnet-4-6" have one dot,
// while prefixed IDs have two or more.
func hasRegionPrefix(modelID string) bool {
	_, after, ok := strings.Cut(modelID, ".")
	if !ok {
		return false
	}

	return strings.ContainsRune(after, '.')
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
	familyClaudeSonnet46: {
		Name:  ModelClaudeSonnet46,
		Label: "Claude Sonnet 4.6",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        false,
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
	familyClaudeSonnet45: {
		Name:  ModelClaudeSonnet45,
		Label: "Claude Sonnet 4.5",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        false,
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
	familyClaudeHaiku45: {
		Name:  ModelClaudeHaiku45,
		Label: "Claude Haiku 4.5",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        false,
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
	familyClaudeOpus46: {
		Name:  ModelClaudeOpus46,
		Label: "Claude Opus 4.6",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        false,
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
	familyClaudeOpus45: {
		Name:  ModelClaudeOpus45,
		Label: "Claude Opus 4.5",
		Capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			Vision:        false,
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
}
