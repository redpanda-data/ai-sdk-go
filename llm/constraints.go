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

package llm

import (
	"fmt"
	"slices"
)

// ModelConstraints defines validation rules and limitations for a specific model.
// This shared type is used by all providers to ensure consistent validation behavior.
type ModelConstraints struct {
	// TemperatureRange defines the valid range for temperature parameter [min, max]
	TemperatureRange [2]float64

	// MaxInputTokens is the maximum context window size (input tokens)
	MaxInputTokens int

	// MaxOutputTokens is the maximum number of tokens the model can generate in a single response
	MaxOutputTokens int

	// SupportedParams lists all parameters this model accepts
	SupportedParams []string

	// MutuallyExclusive defines groups of parameters that cannot be used together.
	// Each slice contains parameter names that conflict with each other.
	// Example: [["temperature", "top_p"]] means temperature and top_p cannot both be set.
	MutuallyExclusive [][]string

	// ConditionalRules define complex validation rules based on configuration state.
	// These handle cases like "if feature X is enabled, parameters Y and Z are disabled"
	ConditionalRules []ConditionalRule
}

// ConditionalRule represents a complex validation rule that depends on configuration state.
type ConditionalRule struct {
	// Condition describes when this rule applies (e.g., "stream_enabled", "thinking_enabled")
	Condition string

	// Disables lists parameters that become unavailable when Condition is true
	Disables []string

	// Message provides a human-readable explanation when this rule is violated
	Message string
}

// ValidateParameterSupport checks if a parameter is supported by the given constraints.
func (c *ModelConstraints) ValidateParameterSupport(param string) error {
	if slices.Contains(c.SupportedParams, param) {
		return nil
	}

	return fmt.Errorf("parameter %s is not supported", param)
}

// ValidateTemperature checks if a temperature value is valid for these constraints.
func (c *ModelConstraints) ValidateTemperature(temp float64) error {
	minTemp, maxTemp := c.TemperatureRange[0], c.TemperatureRange[1]
	if temp < minTemp || temp > maxTemp {
		return fmt.Errorf("temperature %f out of range [%f, %f]", temp, minTemp, maxTemp)
	}

	return nil
}

// ValidateMutualExclusion checks if the given set of parameters violates mutual exclusion rules.
func (c *ModelConstraints) ValidateMutualExclusion(setParams map[string]bool) error {
	for _, group := range c.MutuallyExclusive {
		var conflicting []string

		for _, param := range group {
			if setParams[param] {
				conflicting = append(conflicting, param)
			}
		}

		if len(conflicting) > 1 {
			return fmt.Errorf("cannot use %v together", conflicting)
		}
	}

	return nil
}

// ValidateConditionalRules checks if any conditional rules are violated.
func (c *ModelConstraints) ValidateConditionalRules(setParams map[string]bool) error {
	for _, rule := range c.ConditionalRules {
		if setParams[rule.Condition] {
			for _, disabled := range rule.Disables {
				if setParams[disabled] {
					return fmt.Errorf("%s: %s conflicts with %s", rule.Message, disabled, rule.Condition)
				}
			}
		}
	}

	return nil
}
