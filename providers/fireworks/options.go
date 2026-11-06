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

package fireworks

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Fireworks reasoning efforts. Availability depends on the selected model and API.
const (
	ReasoningEffortNone   llm.ReasoningEffort = "none"
	ReasoningEffortLow    llm.ReasoningEffort = "low"
	ReasoningEffortMedium llm.ReasoningEffort = "medium"
	ReasoningEffortHigh   llm.ReasoningEffort = "high"
	ReasoningEffortMax    llm.ReasoningEffort = "max"
)

// Option configures a Fireworks model using the selected API's native controls.
type Option func(*modelConfig) error

type modelConfig struct {
	maxTokens   int
	temperature *float64
	effort      *llm.ReasoningEffort
	thinking    *bool
}

// WithMaxTokens bounds generated output, including reasoning. The default is
// 4096 tokens, capped at the offering's output limit.
func WithMaxTokens(tokens int) Option {
	return func(cfg *modelConfig) error {
		cfg.maxTokens = tokens
		return nil
	}
}

// WithTemperature sets sampling temperature within the offering's range.
func WithTemperature(temperature float64) Option {
	return func(cfg *modelConfig) error {
		cfg.temperature = &temperature
		return nil
	}
}

// WithReasoningEffort selects an exact effort advertised for the selected API.
// DeepSeek's "none" disables thinking on Messages and Chat Completions.
func WithReasoningEffort(effort llm.ReasoningEffort) Option {
	return func(cfg *modelConfig) error {
		cfg.effort = &effort
		return nil
	}
}

// WithThinking selects adaptive thinking when true, or disables it when false.
// It is only available on bindings declaring adaptive reasoning support.
// MiniMax Messages with disabled thinking cannot use schema output.
func WithThinking(enabled bool) Option {
	return func(cfg *modelConfig) error {
		cfg.thinking = &enabled
		return nil
	}
}

func (cfg modelConfig) validate(offering catalog.Offering) error {
	if cfg.maxTokens < 1 || cfg.maxTokens > offering.Constraints.MaxOutputTokens {
		return fmt.Errorf("max_tokens must be between 1 and %d", offering.Constraints.MaxOutputTokens)
	}

	if cfg.temperature != nil {
		if math.IsNaN(*cfg.temperature) || math.IsInf(*cfg.temperature, 0) {
			return errors.New("temperature must be finite")
		}

		if err := offering.Constraints.ValidateTemperature(*cfg.temperature); err != nil {
			return err
		}
	}

	if cfg.effort != nil && !slices.Contains(offering.Reasoning.Efforts, *cfg.effort) {
		return fmt.Errorf("reasoning effort %q is not supported (supported: %v)", *cfg.effort, offering.Reasoning.Efforts)
	}

	if cfg.thinking != nil && !offering.Reasoning.Adaptive {
		return errors.New("adaptive thinking is not supported; use an advertised reasoning effort instead")
	}

	return nil
}
