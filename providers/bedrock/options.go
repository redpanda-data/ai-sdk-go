package bedrock

import (
	"fmt"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Option configures a Bedrock model instance using functional options.
type Option func(*Config) error

// Config holds the configuration for a Bedrock model instance.
type Config struct {
	ModelName   string
	Constraints llm.ModelConstraints

	Temperature *float64
	TopP        *float64
	MaxTokens   *int32
	Stop        []string

	// EnableCaching enables prompt caching on Bedrock.
	EnableCaching bool

	// Track which options have been set for conflict detection.
	setOptions map[string]bool
}

// WithTemperature sets the temperature parameter (0.0-1.0).
func WithTemperature(temp float64) Option {
	return func(cfg *Config) error {
		if err := cfg.Constraints.ValidateParameterSupport("temperature"); err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if err := cfg.Constraints.ValidateTemperature(temp); err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		cfg.setOptions["temperature"] = true

		if err := cfg.Constraints.ValidateMutualExclusion(cfg.setOptions); err != nil {
			delete(cfg.setOptions, "temperature")
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		cfg.Temperature = &temp

		return nil
	}
}

// WithTopP sets the top_p parameter (0.0-1.0).
func WithTopP(topP float64) Option {
	return func(cfg *Config) error {
		if err := cfg.Constraints.ValidateParameterSupport("top_p"); err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if topP < 0 || topP > 1 {
			return fmt.Errorf("%s: top_p must be 0.0-1.0, got %f", cfg.ModelName, topP)
		}

		cfg.setOptions["top_p"] = true

		if err := cfg.Constraints.ValidateMutualExclusion(cfg.setOptions); err != nil {
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
		if err := cfg.Constraints.ValidateParameterSupport("max_tokens"); err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		if tokens < 1 {
			return fmt.Errorf("%s: max_tokens must be positive, got %d", cfg.ModelName, tokens)
		}

		if tokens > cfg.Constraints.MaxOutputTokens {
			return fmt.Errorf("%s: max_tokens %d exceeds limit %d", cfg.ModelName, tokens, cfg.Constraints.MaxOutputTokens)
		}

		v := int32(tokens) //nolint:gosec // bounds checked: 1 <= tokens <= MaxOutputTokens (<=128000)
		cfg.MaxTokens = &v
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

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if c.ModelName == "" {
		return fmt.Errorf("%w: model name is required", llm.ErrInvalidConfig)
	}

	for option := range c.setOptions {
		if err := c.Constraints.ValidateParameterSupport(option); err != nil {
			return fmt.Errorf("%w: %w", llm.ErrInvalidConfig, err)
		}
	}

	if err := c.Constraints.ValidateMutualExclusion(c.setOptions); err != nil {
		return fmt.Errorf("%w: %w", llm.ErrInvalidConfig, err)
	}

	if err := c.Constraints.ValidateConditionalRules(c.setOptions); err != nil {
		return fmt.Errorf("%w: %w", llm.ErrInvalidConfig, err)
	}

	return nil
}
