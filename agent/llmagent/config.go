package llmagent

import (
	"errors"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// config holds the internal configuration for an LLMAgent.
type config struct {
	name            string
	description     string
	systemPrompt    string
	model           llm.Model
	tools           tool.Registry
	interceptors    []agent.Interceptor
	maxTurns        int
	toolConcurrency int
}

// validate checks that the configuration is valid.
func (c *config) validate() error {
	if c.name == "" {
		return errors.New("llmagent: name is required")
	}

	if c.systemPrompt == "" {
		return errors.New("llmagent: system prompt is required")
	}

	if c.model == nil {
		return errors.New("llmagent: model is required")
	}

	if c.maxTurns < 0 {
		return fmt.Errorf("llmagent: maxTurns must not be negative, got %d", c.maxTurns)
	}

	if c.toolConcurrency <= 0 {
		return fmt.Errorf("llmagent: toolConcurrency must be positive, got %d", c.toolConcurrency)
	}

	// Validate that all interceptors implement at least one interceptor interface
	for i, interceptor := range c.interceptors {
		if !agent.ImplementsAnyInterceptor(interceptor) {
			return fmt.Errorf("llmagent: interceptor at index %d does not implement any valid interface", i)
		}
	}

	return nil
}

// Option configures an LLMAgent.
type Option func(*config)

// WithDescription sets the agent's description.
// Used when wrapping agents as tools (agent-as-tool pattern).
func WithDescription(description string) Option {
	return func(c *config) {
		c.description = description
	}
}

// WithTools sets the registry of available tools.
func WithTools(tools tool.Registry) Option {
	return func(c *config) {
		c.tools = tools
	}
}

// WithMaxTurns sets the maximum number of turns per invocation.
// 0 keeps the default (25). Negative values are rejected at validation.
func WithMaxTurns(maxTurns int) Option {
	return func(c *config) {
		if maxTurns > 0 {
			c.maxTurns = maxTurns
		}
	}
}

// WithToolConcurrency limits parallel tool execution.
// Defaults to 3 if not specified.
func WithToolConcurrency(toolConcurrency int) Option {
	return func(c *config) {
		c.toolConcurrency = toolConcurrency
	}
}

// WithInterceptors sets the interceptors to be applied during agent execution.
// Interceptors can intercept and modify behavior at various points in the execution lifecycle.
func WithInterceptors(i ...agent.Interceptor) Option {
	return func(c *config) {
		c.interceptors = i
	}
}
