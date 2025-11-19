package openaicompat

import "github.com/redpanda-data/ai-sdk-go/llm"

// getDefaultCapabilities returns permissive capabilities for dynamic models.
// Assumes Chat Completion API supports all standard features.
func getDefaultCapabilities() llm.ModelCapabilities {
	return llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true, // Most OpenAI-compatible APIs support json_object
		StructuredOutput: true, // Most support json_schema, but not all (e.g., DeepSeek)
		Vision:           true,
		Audio:            false, // Not commonly supported in Chat API
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        false, // Model-specific, not general Chat API feature
	}
}

// getDefaultConstraints returns permissive constraints for dynamic models.
// Uses broad ranges to accommodate various OpenAI-compatible services.
func getDefaultConstraints() llm.ModelConstraints {
	return llm.ModelConstraints{
		TemperatureRange:  [2]float64{0.0, 2.0},
		MaxInputTokens:    100000, // Generous limit for most models
		MaxOutputTokens:   16384,  // Conservative default for most models
		SupportedParams:   []string{"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "logprobs", "stop"},
		MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		ConditionalRules:  []llm.ConditionalRule{},
	}
}
