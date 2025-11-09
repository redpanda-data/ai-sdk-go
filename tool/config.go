package tool

import (
	"errors"
	"fmt"
	"time"
)

// Config holds configuration for tool registration and execution.
type Config struct {
	// Timeout for tool execution (0 = no timeout)
	Timeout time.Duration

	// Maximum response size in estimated tokens (0 = use registry default)
	MaxResponseTokens int

	// Message to return when response exceeds token limit (empty = use registry default)
	ResponseTooLargeMessage string

	// Custom metadata for the tool
	Metadata map[string]any
}

// Option configures tool registration behavior with validation.
type Option func(*Config) error

// NewConfig creates a default configuration.
func defaultToolConfig() *Config {
	return &Config{
		Timeout:                 30 * time.Second, // Reasonable default for most operations
		MaxResponseTokens:       25000,            // Reasonable context window portion
		ResponseTooLargeMessage: "Response too large for context window. Consider using pagination, filtering, or requesting a summary instead of the full result.",
		Metadata:                make(map[string]any),
	}
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if c.Timeout < 0 {
		return fmt.Errorf("%w: timeout must be non-negative, got %v", ErrInvalidToolConfig, c.Timeout)
	}

	if c.MaxResponseTokens < 0 {
		return fmt.Errorf("%w: max response tokens must be non-negative, got %d", ErrInvalidToolConfig, c.MaxResponseTokens)
	}

	return nil
}

// applyOptions applies a list of options to the config with validation.
// Returns an error if any option fails validation.
func (c *Config) applyOptions(options ...Option) error {
	for i, opt := range options {
		err := opt(c)
		if err != nil {
			return fmt.Errorf("option %d failed: %w", i, err)
		}
	}

	// Final validation after all options are applied
	return c.Validate()
}

// WithTimeout sets the timeout for tool execution.
// Must be non-negative (0 = no timeout).
//
// Use this when:
// - Tool may run indefinitely (e.g., code execution, file processing)
// - You need to prevent resource exhaustion from stuck operations
// - Different tools have different expected completion times
//
// Example: WithTimeout(5*time.Minute) for large file uploads.
func WithTimeout(timeout time.Duration) Option {
	return func(c *Config) error {
		if timeout < 0 {
			return fmt.Errorf("%w: timeout must be non-negative, got %v", ErrInvalidToolConfig, timeout)
		}

		c.Timeout = timeout

		return nil
	}
}

// WithMaxResponseTokens sets the maximum response size in estimated tokens.
// Must be non-negative (0 = use registry default).
//
// Use this when:
// - Tool generates large outputs (logs, database dumps, search results)
// - You need to prevent context window overflow in LLM conversations
// - Different tools have different expected response sizes
//
// Common values:
// - 1000: Simple status/configuration tools
// - 10000: Complex query results, moderate file contents
// - 50000: Large data dumps, extensive logs (use with caution)
//
// Example: WithMaxResponseTokens(5000) for database query results.
func WithMaxResponseTokens(maxTokens int) Option {
	return func(c *Config) error {
		if maxTokens < 0 {
			return fmt.Errorf("%w: max response tokens must be non-negative, got %d", ErrInvalidToolConfig, maxTokens)
		}

		c.MaxResponseTokens = maxTokens

		return nil
	}
}

// WithResponseTooLargeMessage sets a custom message for when responses exceed token limits.
//
// Use this when:
// - You want to provide tool-specific guidance when output is truncated
// - Default message doesn't give users enough context about what to do
// - Tool has specific ways to reduce output size (filters, pagination, etc.)
//
// Examples:
// - "Query returned too many results. Try adding WHERE clauses or LIMIT."
// - "Log file too large. Use --since flag to limit time range."
// - "Search results truncated. Refine your search terms."
//
// Example: WithResponseTooLargeMessage("Use LIMIT clause to reduce result set").
func WithResponseTooLargeMessage(message string) Option {
	return func(c *Config) error {
		c.ResponseTooLargeMessage = message
		return nil
	}
}

// WithMetadata sets custom metadata for the tool.
// This metadata is for internal registry management and observability, NOT sent to the LLM.
//
// Use this for:
// - Tool categorization and management (category, risk_level, version)
// - A2A integration context (environment, approval_workflow, notifications)
// - Performance monitoring (expected_latency, cost_per_call, resource_requirements)
// - Governance and compliance (audit_required, data_classification, owner)
//
// Examples:
// - map[string]any{"category": "database", "risk_level": "high"}
// - map[string]any{"environment": "prod", "requires_approval": true}
// - map[string]any{"model_version": "v1.2", "gpu_required": true}
//
// Example: WithMetadata(map[string]any{"category": "external_api", "rate_limited": true}).
func WithMetadata(metadata map[string]any) Option {
	return func(c *Config) error {
		if metadata == nil {
			return errors.New("metadata cannot be nil")
		}

		c.Metadata = metadata

		return nil
	}
}
