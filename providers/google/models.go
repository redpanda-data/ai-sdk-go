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

package google

import (
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Model ID constants for Google Gemini models.
const (
	ModelGemini35Flash       = "gemini-3.5-flash"
	ModelGemini31ProPreview  = "gemini-3.1-pro-preview"
	ModelGemini3ProPreview   = "gemini-3-pro-preview"
	ModelGemini3FlashPreview = "gemini-3-flash-preview"
	ModelGemini25Pro         = "gemini-2.5-pro"
	ModelGemini25Flash       = "gemini-2.5-flash"
	ModelGemini25FlashLite   = "gemini-2.5-flash-lite"
)

// ModelDefinition defines a Gemini model with its capabilities and constraints.
type ModelDefinition struct {
	Name         string
	Label        string
	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints
	Pricing      pricing.Info
}

// resolveModelFamily collapses provider-returned versioned model identifiers to
// the stable model key used by the SDK catalog.
//
// Gemini responses can report values such as "models/gemini-2.5-flash-001",
// while the SDK catalog is keyed by family IDs like "gemini-2.5-flash". Pricing
// and capability lookup need the stable family key, so this resolver picks the
// longest supportedModels prefix after removing the optional "models/" prefix.
func resolveModelFamily(model string) string {
	model = strings.TrimPrefix(model, "models/")

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

// supportedModels defines all Gemini models with their capabilities and constraints.
// Based on https://ai.google.dev/gemini-api/docs/models
var supportedModels = map[string]ModelDefinition{
	ModelGemini35Flash: {
		Name:  ModelGemini35Flash,
		Label: "Gemini 3.5 Flash",
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
			TemperatureRange: [2]float64{0.0, 2.0},
			MaxInputTokens:   1048576,
			MaxOutputTokens:  65535,
		},
		Pricing: pricing.FlatInfo(1.50, 9.00, 0.15),
	},
	ModelGemini31ProPreview: {
		Name:  ModelGemini31ProPreview,
		Label: "Gemini 3.1 Pro Preview",
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
			TemperatureRange: [2]float64{0.0, 2.0},
			MaxInputTokens:   1048576, // 1M input tokens
			MaxOutputTokens:  65535,   // 64K output tokens
		},
		Pricing: pricing.TieredInfo(
			pricing.NewRates(2.00, 12.00, 0.20),
			pricing.Bracket{
				MinContextTokens: 200_001,
				Rates:            pricing.NewRates(4.00, 18.00, 0.40),
			},
		),
	},
	ModelGemini3ProPreview: {
		Name:  ModelGemini3ProPreview,
		Label: "Gemini 3 Pro Preview",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         true,
			StructuredOutput: true,
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Gemini 3 has thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 2.0},
			MaxInputTokens:   1048576, // 1M input tokens
			MaxOutputTokens:  65535,   // 65K output tokens
		},
		Pricing: pricing.TieredInfo(
			pricing.NewRates(2.00, 12.00, 0.20),
			pricing.Bracket{
				MinContextTokens: 200_001,
				Rates:            pricing.NewRates(4.00, 18.00, 0.40),
			},
		),
	},
	ModelGemini3FlashPreview: {
		Name:  ModelGemini3FlashPreview,
		Label: "Gemini 3 Flash Preview",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         true,
			StructuredOutput: true,
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Gemini 3 has thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 2.0},
			MaxInputTokens:   1048576, // 1M input tokens
			MaxOutputTokens:  65535,   // 65K output tokens
		},
		Pricing: pricing.FlatInfo(0.50, 3.00, 0.05),
	},
	ModelGemini25Pro: {
		Name:  ModelGemini25Pro,
		Label: "Gemini 2.5 Pro",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         true, // Gemini supports JSON mode via response_mime_type
			StructuredOutput: true, // Gemini 2.5 has structured outputs support
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Gemini 2.5 has thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 2.0},
			MaxInputTokens:   1048576, // 1M input tokens
			MaxOutputTokens:  65535,   // 65K output tokens
		},
		Pricing: pricing.TieredInfo(
			pricing.NewRates(1.25, 10.00, 0.125),
			pricing.Bracket{
				MinContextTokens: 200_001,
				Rates:            pricing.NewRates(2.50, 15.00, 0.25),
			},
		),
	},
	ModelGemini25Flash: {
		Name:  ModelGemini25Flash,
		Label: "Gemini 2.5 Flash",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         true,
			StructuredOutput: true,
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 2.0},
			MaxInputTokens:   1048576, // 1M input tokens
			MaxOutputTokens:  65535,   // 65K output tokens
		},
		Pricing: pricing.FlatInfo(0.30, 2.50, 0.03),
	},
	ModelGemini25FlashLite: {
		Name:  ModelGemini25FlashLite,
		Label: "Gemini 2.5 Flash Lite",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         true,
			StructuredOutput: true,
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true, // Thinking support
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange: [2]float64{0.0, 2.0},
			MaxInputTokens:   1048576, // 1M input tokens
			MaxOutputTokens:  65535,   // 65K output tokens
		},
		Pricing: pricing.FlatInfo(0.10, 0.40, 0.01),
	},
}
