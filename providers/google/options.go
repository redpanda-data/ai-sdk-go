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
	"errors"
	"fmt"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Option configures a Google Gemini model instance using functional options.
type Option func(*Config) error

// Config holds the configuration for a Google Gemini model instance.
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

	// Extended thinking configuration
	EnableThinking bool   // Enable thinking for reasoning models
	ThinkingBudget *int32 // Optional thinking budget in tokens

	// Custom model name override (inherits base model capabilities)
	CustomModelName string
}

// WithTemperature sets the temperature parameter (0.0-2.0).
// Controls randomness in the model's responses.
func WithTemperature(temp float64) Option {
	return func(cfg *Config) error {
		if err := cfg.Constraints.ValidateTemperature(temp); err != nil {
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
		if topP < 0 || topP > 1 {
			return fmt.Errorf("%s: top_p must be 0.0-1.0, got %f", cfg.ModelName, topP)
		}

		cfg.TopP = &topP

		return nil
	}
}

// WithTopK sets the top_k parameter.
// Only sample from the top K options for each subsequent token.
func WithTopK(topK int32) Option {
	return func(cfg *Config) error {
		if topK < 1 {
			return fmt.Errorf("%s: top_k must be positive, got %d", cfg.ModelName, topK)
		}

		cfg.TopK = &topK

		return nil
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(tokens int32) Option {
	return func(cfg *Config) error {
		if tokens < 1 {
			return fmt.Errorf("%s: max_tokens must be positive, got %d", cfg.ModelName, tokens)
		}

		// Safely check limit without overflow
		if cfg.Constraints.MaxInputTokens > 0 && int(tokens) > cfg.Constraints.MaxInputTokens {
			return fmt.Errorf("%s: max_tokens %d exceeds limit %d", cfg.ModelName, tokens, cfg.Constraints.MaxInputTokens)
		}

		cfg.MaxTokens = &tokens

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

		return nil
	}
}

// WithPresencePenalty sets the presence penalty parameter.
// Positive values penalize new tokens based on whether they appear in the text so far.
func WithPresencePenalty(penalty float32) Option {
	return func(cfg *Config) error {
		if penalty < -2.0 || penalty > 2.0 {
			return fmt.Errorf("%s: presence_penalty must be -2.0 to 2.0, got %f", cfg.ModelName, penalty)
		}

		cfg.PresencePenalty = &penalty

		return nil
	}
}

// WithFrequencyPenalty sets the frequency penalty parameter.
// Positive values penalize new tokens based on their frequency in the text so far.
func WithFrequencyPenalty(penalty float32) Option {
	return func(cfg *Config) error {
		if penalty < -2.0 || penalty > 2.0 {
			return fmt.Errorf("%s: frequency_penalty must be -2.0 to 2.0, got %f", cfg.ModelName, penalty)
		}

		cfg.FrequencyPenalty = &penalty

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

// WithThinkingBudget sets the thinking budget in tokens.
// Only applicable when thinking is enabled.
func WithThinkingBudget(tokens int32) Option {
	return func(cfg *Config) error {
		cfg.ThinkingBudget = &tokens
		return nil
	}
}

// WithCustomModelName allows overriding the model name sent to the API while
// inheriting the base model's capabilities and constraints. This is useful for:
//   - Using newly released models before SDK updates
//   - Testing beta/experimental model variants
//   - Using timestamped versions not yet in the SDK
//
// The custom name will be sent to Google's API, but validation and constraints
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

	return nil
}
