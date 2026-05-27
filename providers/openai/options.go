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

package openai

import (
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Option configures an OpenAI model instance using functional options.
type Option func(*Config) error

// Config holds the configuration for an OpenAI model instance.
//
// The OpenAI Responses API (used by this provider) does NOT support the
// Chat Completions knobs Seed, PresencePenalty, FrequencyPenalty or
// Stop. They are intentionally absent from Config. For Chat Completions
// behaviour use providers/openaicompat.
type Config struct {
	ModelName   string
	Constraints llm.ModelConstraints

	// Generation parameters supported by Responses API
	Temperature *float64
	TopP        *float64
	MaxTokens   *int

	// Log probability output controls (Responses API: opted in via
	// the `Include` request field + TopLogprobs request field).
	LogProbs    *bool
	TopLogProbs *int

	// Reasoning model parameters (GPT-5, O-series)
	ReasoningEffort  *ReasoningEffort  // Defaults to medium if not specified
	ReasoningSummary *ReasoningSummary // Optional, no default
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
// An alternative to temperature for controlling randomness using nucleus sampling.
func WithTopP(topP float64) Option {
	return func(cfg *Config) error {
		if topP < 0 || topP > 1 {
			return fmt.Errorf("%s: top_p must be 0.0-1.0, got %f", cfg.ModelName, topP)
		}

		cfg.TopP = &topP

		return nil
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
// Validated against the model's MaxOutputTokens constraint.
func WithMaxTokens(tokens int) Option {
	return func(cfg *Config) error {
		if tokens < 1 {
			return fmt.Errorf("%s: max_tokens must be positive, got %d", cfg.ModelName, tokens)
		}

		if cfg.Constraints.MaxOutputTokens > 0 && tokens > cfg.Constraints.MaxOutputTokens {
			return fmt.Errorf("%s: max_tokens %d exceeds model output limit %d", cfg.ModelName, tokens, cfg.Constraints.MaxOutputTokens)
		}

		cfg.MaxTokens = &tokens

		return nil
	}
}

// WithLogProbs enables log probabilities in the Responses API output.
// When set, the request mapper adds "message.output_text.logprobs" to
// the Include list.
func WithLogProbs(enabled bool) Option {
	return func(cfg *Config) error {
		cfg.LogProbs = &enabled
		return nil
	}
}

// WithTopLogProbs sets the number of top log-probabilities to return per
// generated token. Implies LogProbs (the request mapper sets both).
// Valid range is [0, 20] per the OpenAI Responses API.
func WithTopLogProbs(n int) Option {
	return func(cfg *Config) error {
		if n < 0 || n > 20 {
			return fmt.Errorf("%s: top_logprobs must be 0-20, got %d", cfg.ModelName, n)
		}

		cfg.TopLogProbs = &n

		return nil
	}
}

// WithReasoningEffort sets the computational effort for reasoning models.
// Valid values: "none" (GPT-5.1+ only), "minimal", "low", "medium" (default), "high", "xhigh" (GPT-5.2+).
// Only supported by reasoning models (GPT-5, O-series).
func WithReasoningEffort(effort ReasoningEffort) Option {
	return func(cfg *Config) error {
		switch effort {
		case ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh:
			cfg.ReasoningEffort = &effort
			return nil
		default:
			return fmt.Errorf("%s: invalid reasoning effort %q", cfg.ModelName, effort)
		}
	}
}

// WithReasoningSummary sets how reasoning traces are summarized.
// Valid values: "auto", "concise", "detailed".
// Only supported by reasoning models (GPT-5, O-series).
func WithReasoningSummary(summary ReasoningSummary) Option {
	return func(cfg *Config) error {
		switch summary {
		case ReasoningSummaryAuto, ReasoningSummaryConcise, ReasoningSummaryDetailed:
			cfg.ReasoningSummary = &summary
			return nil
		default:
			return fmt.Errorf("%s: invalid reasoning summary %q", cfg.ModelName, summary)
		}
	}
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if c.ModelName == "" {
		return fmt.Errorf("%w: model name is required", llm.ErrInvalidConfig)
	}

	return nil
}
