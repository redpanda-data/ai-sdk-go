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

package llmagent

import (
	"context"
	"errors"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// SystemPromptProvider is a function that returns the system prompt for a
// given request. It is called once per LLM call (i.e., every turn in the
// agentic loop), receiving the request's context.Context so callers can
// pass per-request data such as the authenticated user's identity.
//
// Use [WithSystemPromptProvider] to configure it. When set, it takes
// precedence over the static systemPrompt string.
type SystemPromptProvider func(ctx context.Context) (string, error)

// config holds the internal configuration for an LLMAgent.
type config struct {
	name                 string
	description          string
	systemPrompt         string
	systemPromptProvider SystemPromptProvider
	id                   string
	version              string
	model                llm.Model
	tools                tool.Registry
	interceptors         []agent.Interceptor
	maxTurns             int
	toolConcurrency      int
}

// validate checks that the configuration is valid.
func (c *config) validate() error {
	if c.name == "" {
		return errors.New("llmagent: name is required")
	}

	if c.systemPrompt == "" && c.systemPromptProvider == nil {
		return errors.New("llmagent: system prompt is required (set either systemPrompt or SystemPromptProvider)")
	}

	if c.model == nil {
		return errors.New("llmagent: model is required")
	}

	if c.maxTurns <= 0 {
		return fmt.Errorf("llmagent: maxTurns must be positive, got %d", c.maxTurns)
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

// WithSystemPromptProvider sets a dynamic system prompt provider.
//
// When set, the provider is called every turn to produce the system prompt,
// and the static systemPrompt argument to [New] is ignored. Pass an empty
// string for systemPrompt when using a provider.
//
// The provider receives the request's context.Context, so callers can inject
// per-request data (e.g., authenticated user identity) via [context.WithValue]
// and read it back inside the provider.
func WithSystemPromptProvider(p SystemPromptProvider) Option {
	return func(c *config) {
		c.systemPromptProvider = p
	}
}

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
// Defaults to 25 if not specified.
func WithMaxTurns(maxTurns int) Option {
	return func(c *config) {
		c.maxTurns = maxTurns
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

// WithID sets the agent's unique identifier (used for gen_ai.agent.id).
func WithID(id string) Option {
	return func(c *config) {
		c.id = id
	}
}

// WithVersion sets the agent's version (used for gen_ai.agent.version).
func WithVersion(version string) Option {
	return func(c *config) {
		c.version = version
	}
}
