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
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Model ID constants for Google Gemini models.
const (
	ModelGemini31ProPreview  = "gemini-3.1-pro-preview"
	ModelGemini3ProPreview   = "gemini-3-pro-preview"
	ModelGemini3FlashPreview = "gemini-3-flash-preview"
	ModelGemini25Pro         = "gemini-2.5-pro"
	ModelGemini25Flash       = "gemini-2.5-flash"
	ModelGemini25FlashLite   = "gemini-2.5-flash-lite"
	ModelGemini20Flash       = "gemini-2.0-flash"
)

// ModelDefinition defines a Gemini model with its capabilities and constraints.
type ModelDefinition struct {
	Name         string
	Label        string
	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints
	Pricing      pricing.Info
}

// supportedModels defines all Gemini models with their capabilities and constraints.
// Based on https://ai.google.dev/gemini-api/docs/models
var supportedModels = map[string]ModelDefinition{
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
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1048576, // 1M input tokens
			MaxOutputTokens:   65535,   // 64K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion: 200_000_000, OutputPerMillion: 1_200_000_000, CachedInputPerMillion: 20_000_000,
			Tiers: []pricing.Tier{
				{MaxInputTokens: 200_000, InputPerMillion: 200_000_000, OutputPerMillion: 1_200_000_000, CachedInputPerMillion: 20_000_000},
				{MaxInputTokens: 0, InputPerMillion: 400_000_000, OutputPerMillion: 1_800_000_000, CachedInputPerMillion: 40_000_000},
			},
		},
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
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1048576, // 1M input tokens
			MaxOutputTokens:   65535,   // 65K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion: 200_000_000, OutputPerMillion: 1_200_000_000, CachedInputPerMillion: 20_000_000,
			Tiers: []pricing.Tier{
				{MaxInputTokens: 200_000, InputPerMillion: 200_000_000, OutputPerMillion: 1_200_000_000, CachedInputPerMillion: 20_000_000},
				{MaxInputTokens: 0, InputPerMillion: 400_000_000, OutputPerMillion: 1_800_000_000, CachedInputPerMillion: 40_000_000},
			},
		},
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
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1048576, // 1M input tokens
			MaxOutputTokens:   65535,   // 65K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion: 50_000_000, OutputPerMillion: 300_000_000, CachedInputPerMillion: 5_000_000,
		},
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
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1048576, // 1M input tokens
			MaxOutputTokens:   65535,   // 65K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion: 125_000_000, OutputPerMillion: 1_000_000_000, CachedInputPerMillion: 12_500_000,
			Tiers: []pricing.Tier{
				{MaxInputTokens: 200_000, InputPerMillion: 125_000_000, OutputPerMillion: 1_000_000_000, CachedInputPerMillion: 12_500_000},
				{MaxInputTokens: 0, InputPerMillion: 250_000_000, OutputPerMillion: 1_500_000_000, CachedInputPerMillion: 25_000_000},
			},
		},
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
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1048576, // 1M input tokens
			MaxOutputTokens:   65535,   // 65K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion: 30_000_000, OutputPerMillion: 250_000_000, CachedInputPerMillion: 3_000_000,
		},
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
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1048576, // 1M input tokens
			MaxOutputTokens:   65535,   // 65K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion: 10_000_000, OutputPerMillion: 40_000_000, CachedInputPerMillion: 1_000_000,
		},
	},
	ModelGemini20Flash: {
		Name:  ModelGemini20Flash,
		Label: "Gemini 2.0 Flash",
		Capabilities: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			JSONMode:         true,
			StructuredOutput: false, // Previous generation
			Vision:           true,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        false, // Standard model without explicit thinking
		},
		Constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1048576, // 1M input tokens
			MaxOutputTokens:   8192,    // 8K output tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
		},
		Pricing: pricing.Info{
			InputPerMillion: 10_000_000, OutputPerMillion: 40_000_000, CachedInputPerMillion: 2_500_000,
		},
	},
}
