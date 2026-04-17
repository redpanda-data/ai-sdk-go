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
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Model ID constants for Claude models on Bedrock.
// These are real Bedrock model IDs that can be passed directly to NewModel.
const (
	ModelClaudeSonnet46 = "anthropic.claude-sonnet-4-6"
	ModelClaudeSonnet45 = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	ModelClaudeHaiku45  = "anthropic.claude-haiku-4-5-20251001-v1:0"
	ModelClaudeOpus47   = "anthropic.claude-opus-4-7"
	ModelClaudeOpus46   = "anthropic.claude-opus-4-6-v1"
	ModelClaudeOpus45   = "anthropic.claude-opus-4-5-20251101-v1:0"
)

// ModelDefinition defines a model with its capabilities and constraints.
type ModelDefinition struct {
	Name         string // Real Bedrock model ID (e.g. "anthropic.claude-sonnet-4-5-20250929-v1:0")
	Label        string
	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints
	Pricing      pricing.Info
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

// lookupModel finds a ModelDefinition by model ID.
// It tries a direct map lookup first, then strips the region prefix
// (e.g. "us." from "us.anthropic.claude-sonnet-4-6") and retries.
func lookupModel(modelName string) (ModelDefinition, bool) {
	if def, ok := supportedModels[modelName]; ok {
		return def, true
	}

	// Strip region prefix: "us.anthropic.claude-…" → "anthropic.claude-…"
	if _, after, ok := strings.Cut(modelName, "."); ok {
		if def, ok := supportedModels[after]; ok {
			return def, true
		}
	}

	return ModelDefinition{}, false
}

// supportedModels defines Claude models available on Bedrock via the Converse API.
// Standard features only — no Anthropic-specific thinking/effort/speed.
var supportedModels = map[string]ModelDefinition{
	ModelClaudeOpus47: {
		Name:  ModelClaudeOpus47,
		Label: "Claude Opus 4.7",
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
		Pricing: pricing.Info{
			InputPerMillion:       500_000_000,
			OutputPerMillion:      2_500_000_000,
			CachedInputPerMillion: 50_000_000,
		},
	},
	ModelClaudeOpus45: {
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
		Pricing: pricing.Info{
			InputPerMillion:       500_000_000,
			OutputPerMillion:      2_500_000_000,
			CachedInputPerMillion: 50_000_000,
		},
	},
}
