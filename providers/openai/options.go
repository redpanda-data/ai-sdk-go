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
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Option configures an OpenAI model instance using functional options.
type Option func(*Config) error

// Config holds the configuration for an OpenAI model instance.
type Config struct {
	ModelName   string
	Constraints llm.ModelConstraints

	// OpenAI-specific parameters
	Temperature      *float64
	TopP             *float64
	MaxTokens        *int
	FrequencyPenalty *float64
	PresencePenalty  *float64
	Seed             *int
	LogProbs         *bool
	Stop             []string

	// Reasoning model parameters (GPT-5, O-series)
	ReasoningEffort  *ReasoningEffort  // Defaults to medium if not specified
	ReasoningSummary *ReasoningSummary // Optional, no default

	// ServiceTier is the processing tier sent as service_tier. Unset, OpenAI
	// uses the project's configured tier.
	ServiceTier *llm.ServiceTier

	// Track which options have been set for conflict detection with model constraints
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

		if tokens > cfg.Constraints.MaxInputTokens {
			return fmt.Errorf("%s: max_tokens %d exceeds limit %d", cfg.ModelName, tokens, cfg.Constraints.MaxInputTokens)
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

// WithReasoningEffort sets the computational effort for reasoning models.
// Valid values: "none" (GPT-5.1+ only), "minimal", "low", "medium" (default), "high", "xhigh" (GPT-5.2+), "max" (GPT-5.6+).
// Only supported by reasoning models (GPT-5, O-series).
func WithReasoningEffort(effort ReasoningEffort) Option {
	return func(cfg *Config) error {
		err := cfg.Constraints.ValidateParameterSupport("reasoning_effort")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		switch effort {
		case ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax:
			cfg.ReasoningEffort = &effort
			cfg.setOptions["reasoning_effort"] = true

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
		err := cfg.Constraints.ValidateParameterSupport("reasoning_summary")
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		switch summary {
		case ReasoningSummaryAuto, ReasoningSummaryConcise, ReasoningSummaryDetailed:
			cfg.ReasoningSummary = &summary
			cfg.setOptions["reasoning_summary"] = true

			return nil
		default:
			return fmt.Errorf("%s: invalid reasoning summary %q", cfg.ModelName, summary)
		}
	}
}

// unrequestableServiceTiers are canonical tiers that are not Responses API
// service_tier values: Batch is the separate Batch API, and reserved and
// provisioned throughput are Bedrock capacity modes.
var unrequestableServiceTiers = []llm.ServiceTier{
	llm.ServiceTierBatch,
	llm.ServiceTierReserved,
	llm.ServiceTierProvisionedThroughput,
}

// WithServiceTier selects the processing tier OpenAI serves the request on,
// sent as the Responses API service_tier: llm.ServiceTierFlex for Flex
// processing, llm.ServiceTierPriority for Fast mode (formerly Priority
// processing; OpenAI accepts "priority" and "fast" alike),
// llm.ServiceTierScale for Scale Tier, or llm.ServiceTierDefault to force
// Standard processing. Without this option OpenAI applies the project's
// configured tier ("auto").
//
// The tier is validated and normalized by ResponsesServiceTier, so "fast"
// selects Fast mode. "auto" is rejected: it is OpenAI's default, which
// leaving the option unset already keeps, and normalizing it to Default
// would silently force Standard. A tier the SDK does not name yet is sent as
// given, and OpenAI rejects a tier the model does not offer; Batch, reserved,
// and provisioned-throughput tiers cannot be requested here and fail.
//
// The response reports the tier that actually served the request in
// llm.Response.ServiceTier, which pricing keys on. It can differ from the
// requested tier: Fast mode reports "default" when OpenAI's ramp-rate limit
// downgrades a request to Standard.
func WithServiceTier(tier llm.ServiceTier) Option {
	return func(cfg *Config) error {
		normalized, err := ResponsesServiceTier(tier)
		if err != nil {
			return fmt.Errorf("%s: %w", cfg.ModelName, err)
		}

		cfg.ServiceTier = &normalized

		return nil
	}
}

// ResponsesServiceTier normalizes tier with llm.NormalizeServiceTier and
// reports whether it can be sent as a Responses API service_tier. It rejects
// the empty tier; "auto", OpenAI's default, which an unset service tier
// already keeps and which normalization would turn into a forced Standard;
// and the Batch, reserved, and provisioned-throughput tiers, which are not
// request values. Providers that forward a tier to the Responses API share
// it so their configs hold the same normalized value.
func ResponsesServiceTier(tier llm.ServiceTier) (llm.ServiceTier, error) {
	if strings.EqualFold(strings.TrimSpace(string(tier)), "auto") {
		return "", errors.New(`service tier "auto" is OpenAI's default: leave the service tier unset to keep the project's configured tier`)
	}

	normalized := llm.NormalizeServiceTier(string(tier))
	if normalized == "" {
		return "", errors.New("service tier cannot be empty")
	}

	if slices.Contains(unrequestableServiceTiers, normalized) {
		return "", fmt.Errorf("service tier %q cannot be requested on the Responses API", normalized)
	}

	return normalized, nil
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
