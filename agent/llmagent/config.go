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
// given request. It is called when preparing each LLM call, and may also be
// called while recovering interrupted tool calls so their results can be
// budgeted against the same prompt. It receives both the request context and
// the invocation metadata so callers can draw from either source:
//
//   - ctx carries request-scoped values (e.g., authenticated identity
//     injected by HTTP middleware via [context.WithValue]).
//   - inv exposes session metadata, per-invocation metadata set by
//     interceptors, and the current turn number.
//
// Use [WithSystemPromptProvider] to configure it. When set, it takes
// precedence over the static systemPrompt string.
type SystemPromptProvider func(ctx context.Context, inv *agent.InvocationMetadata) (string, error)

// Compaction budget defaults, applied where the corresponding
// [CompactionConfig] field is zero.
const (
	// DefaultTriggerFraction of the usable window at which compaction runs.
	DefaultTriggerFraction = 0.8

	// DefaultTargetFraction of the usable window compaction reduces toward.
	// The trigger-target gap is what makes compaction rare and big-step, so
	// the prompt-prefix cache is invalidated occasionally, not per-turn.
	DefaultTargetFraction = 0.6
)

// CompactionConfig configures deterministic context compaction: when a
// conversation nears the model's context window, already-read tool results
// are pruned to compact markers first, then the oldest turns are dropped.
// The zero value of each field selects the derived default. A struct rather
// than functional options so fields can be added without breaking callers.
type CompactionConfig struct {
	// OutputReserve overrides the derived answer-room reservation.
	// 0 = min(MaxOutputTokens, max(4096, window/10)).
	OutputReserve int

	// TriggerFraction of the usable window at which compaction runs.
	// 0 = [DefaultTriggerFraction]. Must exceed the target fraction or
	// compaction would fire again immediately after reducing.
	TriggerFraction float64

	// TargetFraction of the usable window compaction reduces toward.
	// 0 = [DefaultTargetFraction]. Must stay below the trigger fraction.
	TargetFraction float64
}

// Validate checks the model-independent invariants of the configuration:
// both fractions in (0, 1), target below trigger (after defaulting), and a
// non-negative output reserve. [New] runs it as part of construction, plus
// window-dependent checks against the model; callers that populate the
// struct from external configuration can run it at load time to reject a
// bad knob with a field-named error instead of failing agent construction.
func (c CompactionConfig) Validate() error {
	trigger := c.TriggerFraction
	if trigger == 0 {
		trigger = DefaultTriggerFraction
	}

	target := c.TargetFraction
	if target == 0 {
		target = DefaultTargetFraction
	}

	if trigger <= 0 || trigger >= 1 {
		return fmt.Errorf("llmagent: compaction trigger fraction must be in (0, 1), got %v", c.TriggerFraction)
	}

	if target <= 0 || target >= 1 {
		return fmt.Errorf("llmagent: compaction target fraction must be in (0, 1), got %v", c.TargetFraction)
	}

	if target >= trigger {
		return fmt.Errorf("llmagent: compaction target fraction (%v) must be below the trigger fraction (%v), or compaction would fire again immediately after reducing", target, trigger)
	}

	if c.OutputReserve < 0 {
		return fmt.Errorf("llmagent: compaction output reserve must not be negative, got %d", c.OutputReserve)
	}

	return nil
}

// config holds the internal configuration for an LLMAgent.
type config struct {
	name                 string
	description          string
	systemPrompt         string
	systemPromptProvider SystemPromptProvider
	inputSchema          map[string]any
	id                   string
	version              string
	model                llm.Model
	tools                tool.Registry
	interceptors         []agent.Interceptor
	maxTurns             int
	toolConcurrency      int
	compaction           *CompactionConfig
	toolResultLimit      int
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

	if err := c.validateCompaction(); err != nil {
		return err
	}

	// Validate that all interceptors implement at least one interceptor interface
	for i, interceptor := range c.interceptors {
		if !agent.ImplementsAnyInterceptor(interceptor) {
			return fmt.Errorf("llmagent: interceptor at index %d does not implement any valid interface", i)
		}
	}

	return nil
}

// validateCompaction fails fast on a compaction or tool-result-limit
// configuration that could never budget correctly at runtime.
func (c *config) validateCompaction() error {
	if c.toolResultLimit < 0 {
		return fmt.Errorf("llmagent: tool result limit must be positive, got %d", c.toolResultLimit)
	}

	if c.compaction == nil {
		return nil
	}

	window := c.model.Constraints().MaxInputTokens
	if window <= 0 {
		return errors.New("llmagent: compaction requires a model with a known context window (Constraints().MaxInputTokens)")
	}

	if err := c.compaction.Validate(); err != nil {
		return err
	}

	if r := c.compaction.OutputReserve; r >= window {
		return fmt.Errorf("llmagent: compaction output reserve %d must be below the model's %d-token window", r, window)
	}

	return nil
}

// Option configures an LLMAgent.
type Option func(*config)

// WithCompaction enables deterministic context compaction: before every model
// call, a request whose estimated size crosses the trigger line is reduced by
// pruning already-read tool results to markers and, if still over, dropping
// the oldest messages. The unread frontier is never touched; a request that
// cannot fit the usable window even after reduction fails with a typed error
// wrapping llm.ErrContextOverflow.
//
// Off by default. The session's Messages are rewritten in place - an
// application that needs a full transcript must persist the event stream
// itself before enabling compaction.
func WithCompaction(cfg CompactionConfig) Option {
	return func(c *config) {
		c.compaction = &cfg
	}
}

// WithToolResultLimit sets the estimated-token threshold for collected tool
// results. An oversized result is replaced at collection time by a marker
// carrying the tool name, error status and a preview - never spliced bytes.
// When the descriptive marker does not fit, a minimal valid-JSON marker is
// used. Useful with or without compaction.
func WithToolResultLimit(tokens int) Option {
	return func(c *config) {
		c.toolResultLimit = tokens
	}
}

// WithSystemPromptProvider sets a dynamic system prompt provider.
//
// When set, the provider is called for every turn that reaches the model, and
// may also be called during interrupted-tool recovery for context budgeting.
// The static systemPrompt argument to [New] is ignored. Pass an empty string
// for systemPrompt when using a provider.
//
// The provider receives both context.Context (for request-scoped values like
// authenticated identity) and [agent.InvocationMetadata] (for session state,
// interceptor metadata, and turn number).
func WithSystemPromptProvider(p SystemPromptProvider) Option {
	return func(c *config) {
		c.systemPromptProvider = p
	}
}

// WithInputSchema overrides the JSON Schema this agent advertises for its input.
//
// It matters for a SUB-agent. agenttool derives the delegation tool's parameters
// from Agent.InputSchema, and the default is a single freeform `message` string —
// so a router has exactly one field to say everything in, and anything it forgets
// to mention is simply gone. A structured schema turns the delegation contract into
// named arguments the caller must fill, which is how OpenAI's Agents SDK typed
// handoffs and LangGraph's typed state work.
//
// The schema is advertised, not enforced beyond JSON Schema validation: agenttool
// hands the encoded arguments to the sub-agent as its user message, so the callee
// sees the JSON.
func WithInputSchema(schema map[string]any) Option {
	return func(c *config) {
		c.inputSchema = schema
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
