package anthropic

import (
	"fmt"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Option configures an Anthropic model instance using functional options.
type Option func(*Config) error

// Config holds the configuration for an Anthropic model instance.
type Config struct {
	ModelName   string
	Constraints llm.ModelConstraints

	// Anthropic-specific parameters
	Temperature *float64
	TopP        *float64
	TopK        *int
	MaxTokens   *int
	Stop        []string

	// Extended thinking configuration
	EnableThinking bool // Enable extended thinking for reasoning models

	// Track which options have been set for conflict detection with model constraints
	setOptions map[string]bool
}

// WithTemperature sets the temperature parameter (0.0-1.0).
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
// Nucleus sampling parameter for controlling randomness.
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

// WithTopK sets the top_k parameter.
// Only sample from the top K options for each subsequent token.
func WithTopK(topK int) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("top_k")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if topK < 1 {
			return fmt.Errorf("%s: top_k must be positive, got %d", cfg.ModelName, topK)
		}

		cfg.TopK = &topK
		cfg.setOptions["top_k"] = true

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

// WithThinking enables extended thinking for reasoning.
// Only applicable to models with Reasoning capability.
func WithThinking(enabled bool) Option {
	return func(cfg *Config) error {
		cfg.EnableThinking = enabled
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
