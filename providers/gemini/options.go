package gemini

import (
	"errors"
	"fmt"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Option configures a Gemini model instance using functional options.
type Option func(*Config) error

// Config holds the configuration for a Gemini model instance.
type Config struct {
	ModelName   string
	Constraints llm.ModelConstraints

	// Gemini-specific parameters
	Temperature      *float64
	TopP             *float64
	TopK             *int32
	MaxTokens        *int32
	Stop             []string
	PresencePenalty  *float32
	FrequencyPenalty *float32
	ResponseMimeType *string // For JSON mode: "application/json"
	ResponseSchema   *string // JSON schema for structured output

	// Extended thinking configuration
	EnableThinking bool // Enable thinking for reasoning models

	// Custom model name override (inherits base model capabilities)
	CustomModelName string

	// Track which options have been set for conflict detection
	setOptions map[string]bool
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

		cfg.setOptions["temperature"] = true

		err = cfg.Constraints.ValidateMutualExclusion(cfg.setOptions)
		if err != nil {
			delete(cfg.setOptions, "temperature")
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
func WithTopK(topK int32) Option {
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
func WithMaxTokens(tokens int32) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("max_tokens")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if tokens < 1 {
			return fmt.Errorf("%s: max_tokens must be positive, got %d", cfg.ModelName, tokens)
		}

		// Safely check limit without overflow
		if cfg.Constraints.MaxTokensLimit > 0 && int(tokens) > cfg.Constraints.MaxTokensLimit {
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
		if len(sequences) > 5 {
			return fmt.Errorf("%s: maximum 5 stop sequences allowed, got %d", cfg.ModelName, len(sequences))
		}

		if slices.Contains(sequences, "") {
			return fmt.Errorf("%s: stop sequences cannot be empty", cfg.ModelName)
		}

		cfg.Stop = sequences
		cfg.setOptions["stop"] = true

		return nil
	}
}

// WithPresencePenalty sets the presence penalty parameter.
// Positive values penalize new tokens based on whether they appear in the text so far.
func WithPresencePenalty(penalty float32) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("presence_penalty")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if penalty < -2.0 || penalty > 2.0 {
			return fmt.Errorf("%s: presence_penalty must be -2.0 to 2.0, got %f", cfg.ModelName, penalty)
		}

		cfg.PresencePenalty = &penalty
		cfg.setOptions["presence_penalty"] = true

		return nil
	}
}

// WithFrequencyPenalty sets the frequency penalty parameter.
// Positive values penalize new tokens based on their frequency in the text so far.
func WithFrequencyPenalty(penalty float32) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("frequency_penalty")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if penalty < -2.0 || penalty > 2.0 {
			return fmt.Errorf("%s: frequency_penalty must be -2.0 to 2.0, got %f", cfg.ModelName, penalty)
		}

		cfg.FrequencyPenalty = &penalty
		cfg.setOptions["frequency_penalty"] = true

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

// WithJSONMode enables JSON mode output.
// Sets the response MIME type to application/json.
func WithJSONMode() Option {
	return func(cfg *Config) error {
		mimeType := "application/json"
		cfg.ResponseMimeType = &mimeType

		return nil
	}
}

// WithResponseSchema sets a JSON schema for structured output.
// Automatically enables JSON mode.
func WithResponseSchema(schema string) Option {
	return func(cfg *Config) error {
		if schema == "" {
			return errors.New("response schema cannot be empty")
		}

		mimeType := "application/json"
		cfg.ResponseMimeType = &mimeType
		cfg.ResponseSchema = &schema

		return nil
	}
}

// WithCustomModelName allows overriding the model name sent to the API while
// inheriting the base model's capabilities and constraints. This is useful for:
//   - Using newly released models before SDK updates
//   - Testing beta/experimental model variants
//   - Using timestamped versions not yet in the SDK
//
// The custom name will be sent to Gemini's API, but validation and constraints
// are inherited from the base model specified in NewModel().
func WithCustomModelName(customName string) Option {
	return func(cfg *Config) error {
		if customName == "" {
			return errors.New("custom model name cannot be empty")
		}

		cfg.CustomModelName = customName

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
