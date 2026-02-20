package anthropic

import (
	"errors"
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
	MaxTokens   int // Required by Anthropic API, defaults to 4096
	Stop        []string

	// Extended thinking configuration
	EnableThinking   bool   // Enable extended thinking for reasoning models
	ThinkingBudget   *int64 // Explicit thinking budget in tokens (min 1024)
	AdaptiveThinking bool   // Whether model supports adaptive thinking (set from ModelDefinition)

	// Effort and speed configuration
	Effort *Effort // Output effort level
	Speed  *Speed  // Inference speed mode

	// Prompt caching configuration
	EnableCaching bool // Enable prompt caching by setting cache_control markers

	// Custom model name override (inherits base model capabilities)
	CustomModelName string

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

		if tokens > cfg.Constraints.MaxInputTokens {
			return fmt.Errorf("%s: max_tokens %d exceeds limit %d", cfg.ModelName, tokens, cfg.Constraints.MaxInputTokens)
		}

		cfg.MaxTokens = tokens
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

// WithThinkingBudget sets an explicit thinking budget in tokens.
// The minimum budget is 1024 tokens (API requirement).
// Only applicable to models with "thinking_budget" in SupportedParams.
func WithThinkingBudget(tokens int64) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("thinking_budget")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if tokens < 1024 {
			return fmt.Errorf("%s: thinking_budget must be at least 1024, got %d", cfg.ModelName, tokens)
		}

		cfg.ThinkingBudget = &tokens
		cfg.setOptions["thinking_budget"] = true

		return nil
	}
}

// WithEffort sets the output effort level for the model.
// Only applicable to models with "effort" in SupportedParams.
// The specific effort value is validated against the model's SupportedEfforts in NewModel().
func WithEffort(effort Effort) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("effort")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		cfg.Effort = &effort
		cfg.setOptions["effort"] = true

		return nil
	}
}

// WithSpeed sets the inference speed mode for the model.
// Only applicable to models with "speed" in SupportedParams.
// The specific speed value is validated against the model's SupportedSpeeds in NewModel().
func WithSpeed(speed Speed) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("speed")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		cfg.Speed = &speed
		cfg.setOptions["speed"] = true

		return nil
	}
}

// WithCustomModelName allows overriding the model name sent to the API while
// inheriting the base model's capabilities and constraints. This is useful for:
//   - Using newly released models before SDK updates (e.g., "claude-opus-4-2" using claude-opus-4-1 constraints)
//   - Testing beta/experimental model variants
//   - Using timestamped versions not yet in the SDK
//
// The custom name will be sent to Anthropic's API, but validation and constraints
// are inherited from the base model specified in NewModel().
//
// Example:
//
//	provider.NewModel("claude-opus-4-1", WithCustomModelName("claude-opus-4-2-beta"))
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
