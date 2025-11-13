package gemini

import "github.com/redpanda-data/ai-sdk-go/llm"

// Model ID constants for Google Gemini models.
const (
	ModelGemini25Pro       = "gemini-2.5-pro"
	ModelGemini25Flash     = "gemini-2.5-flash"
	ModelGemini25FlashLite = "gemini-2.5-flash-lite"
	ModelGemini20Flash     = "gemini-2.0-flash"
)

// ModelDefinition defines a Gemini model with its capabilities and constraints.
type ModelDefinition struct {
	Name         string
	Label        string
	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints
}

// modelAliases maps common model name aliases to their canonical versions.
var modelAliases = map[string]string{
	// 2.5 Pro aliases
	"gemini-2.5-pro-latest": ModelGemini25Pro,
	"gemini-pro":            ModelGemini25Pro,

	// 2.5 Flash aliases
	"gemini-2.5-flash-latest": ModelGemini25Flash,
	"gemini-flash":            ModelGemini25Flash,

	// 2.5 Flash Lite aliases
	"gemini-2.5-flash-lite-latest": ModelGemini25FlashLite,
	"gemini-flash-lite":            ModelGemini25FlashLite,
}

// supportedModels defines all Gemini models with their capabilities and constraints.
// Based on https://ai.google.dev/gemini-api/docs/models
var supportedModels = map[string]ModelDefinition{
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
			MaxTokensLimit:    1048576, // 1M input tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
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
			MaxTokensLimit:    1048576, // 1M input tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
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
			MaxTokensLimit:    1048576, // 1M input tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
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
			MaxTokensLimit:    1048576, // 1M input tokens
			SupportedParams:   []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"},
			MutuallyExclusive: [][]string{},
		},
	},
}
