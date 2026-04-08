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

package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Registry manages tool registration, discovery, and execution.
// It serves as the central coordination point for all tool operations.
//
// Design Philosophy:
// - Registry handles the orchestration and lifecycle management
// - Individual Tool implementations focus only on their specific logic
// - Clean separation allows for different tool types (MCP, custom functions, external APIs).
type Registry interface {
	// Register adds a tool to the registry with optional configuration
	Register(tool Tool, opts ...Option) error

	// Unregister removes a tool by name
	Unregister(name string) error

	// List returns tool definitions for use in llm.Request.Tools
	// These definitions tell the LLM what tools are available
	List() []llm.ToolDefinition

	// Get retrieves a registered tool by name
	Get(name string) (Tool, error)

	// Execute runs a tool synchronously and returns the complete result.
	// Returns (nil, error) for validation errors; otherwise returns a ToolResponse with
	// execution errors encoded in the Error field.
	Execute(ctx context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error)

	// ExecuteAll runs multiple tool requests concurrently with optional concurrency limits.
	// Always returns len(reqs) responses in the same order. All failures (tool errors,
	// timeouts, cancellation) are encoded in ToolResponse.Error, never as a top-level error.
	// This ensures callers get a uniform response shape regardless of failure mode.
	ExecuteAll(ctx context.Context, reqs []*llm.ToolRequest, opts ...BatchOption) []*llm.ToolResponse
}

// RegistryConfig configures the overall registry behavior.
// Currently empty but reserved for future registry-level settings.
type RegistryConfig struct {
	// Reserved for future registry-wide settings
	// Tool-specific configuration is handled via functional options
}

// ExecutionContext provides additional context during tool execution
// This allows tools to access registry-level services and configuration.
type ExecutionContext struct {
	// Original LLM request for context
	ToolRequest *llm.ToolRequest

	// Tool configuration
	Config *Config
}

// registry is the concrete implementation of Registry interface.
type registry struct {
	mu     sync.RWMutex
	tools  map[string]*registeredTool
	config RegistryConfig
}

// registeredTool wraps a tool with its configuration.
type registeredTool struct {
	tool   Tool
	config Config
}

// NewRegistry creates a new tool registry with the given configuration.
func NewRegistry(config RegistryConfig) Registry {
	return &registry{
		tools:  make(map[string]*registeredTool),
		config: config,
	}
}

// Register adds a tool to the registry with optional configuration.
func (r *registry) Register(tool Tool, opts ...Option) error {
	if tool == nil {
		return ErrToolNil
	}

	definition := tool.Definition()
	if definition.Name == "" {
		return ErrToolNameEmpty
	}

	// Create config with tool defaults
	config := defaultToolConfig()

	// Apply user options with validation
	err := config.applyOptions(opts...)
	if err != nil {
		return fmt.Errorf("%w for tool %q: %w", ErrInvalidToolConfig, definition.Name, err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	// Check for duplicate names
	if _, exists := r.tools[definition.Name]; exists {
		return fmt.Errorf("%w: %q", ErrToolAlreadyRegistered, definition.Name)
	}

	r.tools[definition.Name] = &registeredTool{
		tool:   tool,
		config: *config,
	}

	return nil
}

// Unregister removes a tool by name.
func (r *registry) Unregister(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[name]; !exists {
		return fmt.Errorf("%w: %q", ErrToolNotFound, name)
	}

	delete(r.tools, name)

	return nil
}

// List returns tool definitions for use in llm.Request.Tools.
// Asynchronous tools (IsAsynchronous() == true) have a note appended to their
// description instructing the LLM not to re-invoke them after a pending status.
func (r *registry) List() []llm.ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()

	definitions := make([]llm.ToolDefinition, 0, len(r.tools))
	for _, registered := range r.tools {
		def := registered.tool.Definition()
		if registered.tool.IsAsynchronous() {
			def.Description += "\n\nNOTE: This is an asynchronous operation. " +
				"Do not call this tool again if it has already returned " +
				"an intermediate or pending status."
		}

		definitions = append(definitions, def)
	}

	return definitions
}

// Get retrieves a registered tool by name.
func (r *registry) Get(name string) (Tool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	registered, exists := r.tools[name]
	if !exists {
		return nil, fmt.Errorf("%w: %q", ErrToolNotFound, name)
	}

	return registered.tool, nil
}

// Execute runs a tool synchronously and returns the complete result.
func (r *registry) Execute(ctx context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
	if req == nil {
		return nil, ErrToolRequestNil
	}

	// Find the tool
	r.mu.RLock()
	registered, exists := r.tools[req.Name]
	r.mu.RUnlock()

	if !exists {
		return &llm.ToolResponse{
			ID:    req.ID,
			Name:  req.Name,
			Error: fmt.Sprintf("%v: %q", ErrToolNotFound, req.Name),
		}, nil
	}

	// Apply timeout if configured
	executeCtx := ctx

	if registered.config.Timeout > 0 {
		var cancel context.CancelFunc

		executeCtx, cancel = context.WithTimeout(ctx, registered.config.Timeout)
		defer cancel()
	}

	// Execute the tool
	result, err := registered.tool.Execute(executeCtx, req.Arguments)
	// Handle execution errors
	if err != nil {
		// Check if it's a timeout
		if errors.Is(executeCtx.Err(), context.DeadlineExceeded) {
			return &llm.ToolResponse{
				ID:    req.ID,
				Name:  req.Name,
				Error: fmt.Sprintf("%v after %s", ErrToolExecutionTimeout, registered.config.Timeout),
			}, nil
		}

		// Other execution errors
		return &llm.ToolResponse{
			ID:    req.ID,
			Name:  req.Name,
			Error: err.Error(),
		}, nil
	}

	// Check response size and apply limits
	processedResult, err := r.enforceResponseSizeLimit(result, &registered.config)
	if err != nil {
		return &llm.ToolResponse{
			ID:    req.ID,
			Name:  req.Name,
			Error: fmt.Sprintf("failed to process response: %v", err),
		}, nil
	}

	// Success
	return &llm.ToolResponse{
		ID:     req.ID,
		Name:   req.Name,
		Result: processedResult,
	}, nil
}

// ExecuteAll implements Registry.ExecuteAll.
// All errors are encoded in ToolResponse.Error fields, including:
// - Per-tool execution failures
// - Individual tool timeouts
// - Context cancellation (for tasks that never started or were interrupted)
//
// This "best-effort" pattern ensures callers always get a predictable response
// structure with len(reqs) entries, making it simpler to process results without
// checking for top-level errors.
func (r *registry) ExecuteAll(ctx context.Context, reqs []*llm.ToolRequest, opts ...BatchOption) []*llm.ToolResponse {
	n := len(reqs)
	if n == 0 {
		return []*llm.ToolResponse{}
	}

	cfg := defaultBatchConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	concurrency := cfg.concurrency
	if concurrency <= 0 || concurrency > n {
		concurrency = n
	}

	results := make([]*llm.ToolResponse, n)
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(concurrency)

	for i, req := range reqs {
		g.Go(func() error {
			// Per-tool errors are encoded in resp.Error, never propagated to errgroup.
			// This prevents one tool failure from canceling other concurrent executions.
			resp, _ := r.executeOne(gctx, req)
			results[i] = resp

			return nil
		})
	}

	_ = g.Wait() // always nil since we never return errors from g.Go

	// Fill any nil slots (tasks that never started due to context cancellation)
	// with error responses so we always return len(reqs) results.
	if ctx.Err() != nil {
		for i := range results {
			if results[i] == nil {
				results[i] = &llm.ToolResponse{
					Error: ctx.Err().Error(),
				}
			}
		}
	}

	return results
}

// enforceResponseSizeLimit checks response size and applies limits/fallbacks.
func (*registry) enforceResponseSizeLimit(result json.RawMessage, config *Config) (json.RawMessage, error) {
	if config.MaxResponseTokens <= 0 {
		return result, nil // No limit configured
	}

	// Rough token estimation: ~4 characters per token for JSON
	estimatedTokens := len(result) / 4

	if estimatedTokens <= config.MaxResponseTokens {
		return result, nil // Within limits
	}

	// Response too large - create fallback response
	message := config.ResponseTooLargeMessage
	if message == "" {
		message = ErrToolResponseTooLarge.Error()
	}

	fallbackResponse := map[string]any{
		"error":   "response_too_large",
		"message": message,
		"details": map[string]any{
			"estimated_tokens": estimatedTokens,
			"max_tokens":       config.MaxResponseTokens,
		},
	}

	return json.Marshal(fallbackResponse)
}

// executeOne is a helper that handles nil requests and calls Execute.
// It always returns a non-nil ToolResponse, with errors populated in the Error field.
func (r *registry) executeOne(ctx context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
	if req == nil {
		return &llm.ToolResponse{Error: ErrToolRequestNil.Error()}, ErrToolRequestNil
	}

	resp, err := r.Execute(ctx, req)
	if resp == nil {
		resp = &llm.ToolResponse{ID: req.ID, Name: req.Name}
	}

	if err != nil {
		resp.Error = err.Error()
	}

	return resp, err
}

// BatchOption configures ExecuteAll behavior.
type BatchOption func(*batchConfig)

type batchConfig struct {
	concurrency int
}

// WithMaxConcurrency limits the number of in-flight tool executions.
// Default is len(reqs), meaning all tools execute concurrently.
// Set to 1 for sequential execution.
func WithMaxConcurrency(n int) BatchOption {
	return func(c *batchConfig) { c.concurrency = n }
}

func defaultBatchConfig() batchConfig {
	return batchConfig{
		concurrency: 0, // Will be set to len(reqs) in ExecuteAll
	}
}
