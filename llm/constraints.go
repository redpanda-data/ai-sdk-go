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

	// MaxTokensLimit is the maximum context window size (input tokens)
	MaxTokensLimit int

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
