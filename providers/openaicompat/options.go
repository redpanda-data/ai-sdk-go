package openaicompat

import (
	"fmt"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Option configures an OpenAI model instance using functional options.
type Option func(*Config) error

// Config holds the configuration for an OpenAI model instance.
type Config struct {
	ModelName   string
	Constraints llm.ModelConstraints

	// Capability flags
	SupportsReasoning  bool
	CustomCapabilities *llm.ModelCapabilities // Override default capabilities if set

	// OpenAI-specific parameters
	Temperature      *float64
	TopP             *float64
	MaxTokens        *int
	FrequencyPenalty *float64
	PresencePenalty  *float64
	Seed             *int
	LogProbs         *bool
	Stop             []string

	// Track which options have been set for conflict detection with model constraints
	setOptions map[string]bool
}

// WithCapabilities overrides the default model capabilities.
// Use this to specify exact capabilities for models with known limitations.
func WithCapabilities(caps llm.ModelCapabilities) Option {
	return func(cfg *Config) error {
		cfg.CustomCapabilities = &caps
		return nil
	}
}

// WithReasoning marks this model as supporting extended reasoning capabilities.
// Reasoning models have different characteristics like higher latency and specialized
// processing for complex reasoning tasks.
func WithReasoning() Option {
	return func(cfg *Config) error {
		cfg.SupportsReasoning = true
		return nil
	}
}

// WithTemperature sets the temperature parameter (0.0-2.0).
// Controls randomness in the model's responses.
func WithTemperature(temp float64) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("temperature")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		err = cfg.Constraints.ValidateTemperature(temp)
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		// Check for conflicts before setting
		cfg.setOptions["temperature"] = true

		err = cfg.Constraints.ValidateMutualExclusion(cfg.setOptions)
		if err != nil {
			delete(cfg.setOptions, "temperature") // Rollback
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		cfg.Temperature = &temp

		return nil
	}
}

// WithTopP sets the top_p parameter (0.0-1.0).
// An alternative to temperature for controlling randomness using nucleus sampling.
func WithTopP(topP float64) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("top_p")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if topP < 0 || topP > 1 {
			return fmt.Errorf("%s: top_p must be 0.0-1.0, got %f", cfg.ModelName, topP)
		}

		cfg.setOptions["top_p"] = true

		err = cfg.Constraints.ValidateMutualExclusion(cfg.setOptions)
		if err != nil {
			delete(cfg.setOptions, "top_p")
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		cfg.TopP = &topP

		return nil
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(tokens int) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("max_tokens")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if tokens < 1 {
			return fmt.Errorf("%s: max_tokens must be positive, got %d", cfg.ModelName, tokens)
		}

		if tokens > cfg.Constraints.MaxTokensLimit {
			return fmt.Errorf("%s: max_tokens %d exceeds limit %d", cfg.ModelName, tokens, cfg.Constraints.MaxTokensLimit)
		}

		cfg.MaxTokens = &tokens
		cfg.setOptions["max_tokens"] = true

		return nil
	}
}

// WithFrequencyPenalty sets the frequency penalty (-2.0 to 2.0).
// Reduces repetition of tokens based on their frequency in the text so far.
func WithFrequencyPenalty(penalty float64) Option {
	return func(cfg *Config) error {
		if !slices.Contains(cfg.Constraints.SupportedParams, "frequency_penalty") {
			return fmt.Errorf("%s: frequency_penalty not supported", cfg.ModelName)
		}

		if penalty < -2.0 || penalty > 2.0 {
			return fmt.Errorf("%s: frequency_penalty must be -2.0 to 2.0, got %f", cfg.ModelName, penalty)
		}

		cfg.FrequencyPenalty = &penalty
		cfg.setOptions["frequency_penalty"] = true

		return nil
	}
}

// WithPresencePenalty sets the presence penalty (-2.0 to 2.0).
// Reduces repetition of tokens based on whether they appear in the text so far.
func WithPresencePenalty(penalty float64) Option {
	return func(cfg *Config) error {
		if !slices.Contains(cfg.Constraints.SupportedParams, "presence_penalty") {
			return fmt.Errorf("%s: presence_penalty not supported", cfg.ModelName)
		}

		if penalty < -2.0 || penalty > 2.0 {
			return fmt.Errorf("%s: presence_penalty must be -2.0 to 2.0, got %f", cfg.ModelName, penalty)
		}

		cfg.PresencePenalty = &penalty
		cfg.setOptions["presence_penalty"] = true

		return nil
	}
}

// WithSeed sets the seed for deterministic outputs.
// Only supported by models that include "seed" in their SupportedParams.
func WithSeed(seed int) Option {
	return func(cfg *Config) error {
		if !slices.Contains(cfg.Constraints.SupportedParams, "seed") {
			return fmt.Errorf("%s: seed not supported", cfg.ModelName)
		}

		cfg.Seed = &seed
		cfg.setOptions["seed"] = true

		return nil
	}
}

// WithLogProbs enables log probabilities in the response.
// May conflict with streaming depending on the model.
func WithLogProbs(enabled bool) Option {
	return func(cfg *Config) error {
		if !slices.Contains(cfg.Constraints.SupportedParams, "logprobs") {
			return fmt.Errorf("%s: logprobs not supported", cfg.ModelName)
		}

		cfg.setOptions["logprobs"] = true

		err := cfg.Constraints.ValidateConditionalRules(cfg.setOptions)
		if err != nil {
			delete(cfg.setOptions, "logprobs")
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		cfg.LogProbs = &enabled

		return nil
	}
}

// WithStop sets custom stop sequences for generation.
func WithStop(sequences ...string) Option {
	return func(cfg *Config) error {
		if len(sequences) > 4 {
			return fmt.Errorf("%s: maximum 4 stop sequences allowed, got %d", cfg.ModelName, len(sequences))
		}

		if slices.Contains(sequences, "") {
			return fmt.Errorf("%s: stop sequences cannot be empty", cfg.ModelName)
		}

		cfg.Stop = sequences
		cfg.setOptions["stop"] = true

		return nil
	}
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if c.ModelName == "" {
		return fmt.Errorf("%w: model name is required", llm.ErrInvalidConfig)
	}

	// Validate that all set options are actually supported
	for option := range c.setOptions {
		err := c.Constraints.ValidateParameterSupport(option)
		if err != nil {
			return fmt.Errorf("%w: %w", llm.ErrInvalidConfig, err)
		}
	}

	// Validate mutual exclusion rules
	err := c.Constraints.ValidateMutualExclusion(c.setOptions)
	if err != nil {
		return fmt.Errorf("%w: %w", llm.ErrInvalidConfig, err)
	}

	// Validate conditional rules
	err = c.Constraints.ValidateConditionalRules(c.setOptions)
	if err != nil {
		return fmt.Errorf("%w: %w", llm.ErrInvalidConfig, err)
	}

	return nil
}
