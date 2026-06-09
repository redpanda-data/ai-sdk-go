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
//
// Run / RunAll / Resume return tool.ExecutionResult, the typed
// control-flow result that carries Execution{Output, Await, Actions}
// plus a runtime error. The agent loop and runner use these to detect
// pauses, persist pending state, and emit ToolPendingEvent. Callers
// that only need the model-visible payload reconcile via
// ExecutionResult.Response().
//
// Registry is a concrete type: components that want to abstract over it
// should declare their own narrow interface over the methods they use
// (see tool/mcp.ToolRegistry for an example).
type Registry struct {
	mu    sync.RWMutex
	tools map[string]*registeredTool
}

// ExecutionResult is the typed control-flow return from Registry.Run.
// It preserves the request index, ID, and name so callers can emit
// events as tools finish while still persisting the final message in
// request order.
type ExecutionResult struct {
	// Index is the position of this request in the original RunAll
	// input slice. Single-shot Run leaves this at 0.
	Index int

	// Request is the original tool request. ID and Name are guaranteed
	// non-empty for requests that came through Run; nil only for the
	// nil-request slot in RunAll.
	Request *llm.ToolRequestPart

	// Execution is the structured return from the tool. Output is the
	// model-visible JSON; Await drives pause/resume.
	Execution Execution

	// Err is a runtime/tool error. The runtime treats it as a model-
	// visible tool error (the model sees a ToolResponsePart with
	// IsError=true), not a fatal failure of the invocation.
	Err error
}

// Response reconciles e into the ToolResponsePart that should be added
// to session history. It is the single chokepoint where Execution.Output
// + Err become wire-format tool responses, so call sites cannot get the
// shape subtly wrong.
//
// Rules:
//   - Err != nil: IsError=true, Result is `{"error":"<msg>"}`.
//   - Execution.Output non-empty: IsError=false, Result is Output.
//   - Empty Output: IsError=false, Result is `{}` so providers always
//     receive a JSON object.
//
// The Await pointer does not change the shape of the persisted message:
// callers that care about pause must read e.Execution.Await directly.
func (e ExecutionResult) Response() *llm.ToolResponsePart {
	id, name := "", ""
	if e.Request != nil {
		id = e.Request.ID
		name = e.Request.Name
	}

	if e.Err != nil {
		return llm.NewToolErrorPart(id, name, e.Err.Error())
	}

	result := e.Execution.Output
	if len(result) == 0 {
		result = json.RawMessage(`{}`)
	}

	return llm.NewToolResponsePart(id, name, result)
}

// registeredTool wraps a tool with its configuration.
type registeredTool struct {
	tool   Tool
	config Config
}

// RegistryOption configures registry-wide behavior. None are defined
// yet; the variadic NewRegistry signature exists so registry-wide
// settings can be added without breaking callers.
type RegistryOption func(*Registry)

// NewRegistry creates a new tool registry.
func NewRegistry(opts ...RegistryOption) *Registry {
	r := &Registry{
		tools: make(map[string]*registeredTool),
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// Register adds a tool to the registry with optional configuration.
func (r *Registry) Register(t Tool, opts ...Option) error {
	if t == nil {
		return ErrToolNil
	}

	name := t.Name()
	if name == "" {
		return ErrToolNameEmpty
	}

	config := defaultToolConfig()
	if err := config.applyOptions(opts...); err != nil {
		return fmt.Errorf("%w for tool %q: %w", ErrInvalidToolConfig, name, err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("%w: %q", ErrToolAlreadyRegistered, name)
	}

	r.tools[name] = &registeredTool{tool: t, config: *config}

	return nil
}

// Unregister removes a tool by name.
func (r *Registry) Unregister(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[name]; !exists {
		return fmt.Errorf("%w: %q", ErrToolNotFound, name)
	}

	delete(r.tools, name)

	return nil
}

// List returns tool definitions for use in llm.Request.Tools.
func (r *Registry) List() []llm.ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()

	definitions := make([]llm.ToolDefinition, 0, len(r.tools))
	for _, registered := range r.tools {
		definitions = append(definitions, Definition(registered.tool))
	}

	return definitions
}

// Get retrieves a registered tool by name.
func (r *Registry) Get(name string) (Tool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	registered, exists := r.tools[name]
	if !exists {
		return nil, fmt.Errorf("%w: %q", ErrToolNotFound, name)
	}

	return registered.tool, nil
}

// Run executes a single tool request and returns the typed
// ExecutionResult. See ExecutionResult for the error vs. await contract.
func (r *Registry) Run(ctx context.Context, inv InvocationInfo, req *llm.ToolRequestPart) ExecutionResult {
	return r.run(ctx, inv, req, nil)
}

// Resume implements Registry.
func (r *Registry) Resume(ctx context.Context, inv InvocationInfo, req *llm.ToolRequestPart, payload *ResumePayload) ExecutionResult {
	return r.run(ctx, inv, req, payload)
}

// RunAll executes multiple tool requests concurrently. Returned results
// are in request order even though tools execute concurrently.
func (r *Registry) RunAll(ctx context.Context, inv InvocationInfo, reqs []*llm.ToolRequestPart, opts ...BatchOption) []ExecutionResult {
	n := len(reqs)
	if n == 0 {
		return []ExecutionResult{}
	}

	cfg := defaultBatchConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	concurrency := cfg.concurrency
	if concurrency <= 0 || concurrency > n {
		concurrency = n
	}

	results := make([]ExecutionResult, n)
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(concurrency)

	for i, req := range reqs {
		g.Go(func() error {
			res := r.Run(gctx, inv, req)
			res.Index = i
			results[i] = res

			return nil
		})
	}

	_ = g.Wait()

	// Fill any slot the dispatcher could not start due to ctx cancel.
	if ctx.Err() != nil {
		for i := range results {
			if results[i].Request == nil && results[i].Err == nil {
				results[i] = ExecutionResult{Index: i, Request: reqs[i], Err: ctx.Err()}
			}
		}
	}

	return results
}

// run is the shared execution path for Run and Resume.
func (r *Registry) run(ctx context.Context, inv InvocationInfo, req *llm.ToolRequestPart, resume *ResumePayload) ExecutionResult {
	if req == nil {
		return ExecutionResult{Err: ErrToolRequestNil}
	}

	r.mu.RLock()
	registered, exists := r.tools[req.Name]
	r.mu.RUnlock()

	out := ExecutionResult{Request: req}

	if !exists {
		out.Err = fmt.Errorf("%w: %q", ErrToolNotFound, req.Name)
		return out
	}

	executeCtx := ctx

	if registered.config.Timeout > 0 {
		var cancel context.CancelFunc

		executeCtx, cancel = context.WithTimeout(ctx, registered.config.Timeout)
		defer cancel()
	}

	call := Call{
		Request:    *req,
		Invocation: inv,
		Resume:     resume,
	}

	exec, err := registered.tool.Execute(executeCtx, call)
	if err != nil {
		if errors.Is(executeCtx.Err(), context.DeadlineExceeded) && !errors.Is(ctx.Err(), context.DeadlineExceeded) {
			// Per-tool timeout (not caller cancellation): report the
			// configured timeout for clarity.
			out.Err = fmt.Errorf("%w after %s", ErrToolExecutionTimeout, registered.config.Timeout)
			return out
		}

		out.Err = err

		return out
	}

	// Normalize then validate the Await shape before persisting the
	// pause. Normalize fills an empty Resume from the Reason default.
	exec.Await.Normalize()

	if err := exec.Await.Validate(); err != nil {
		out.Err = fmt.Errorf("%w: %w", ErrAwaitInvalid, err)
		return out
	}

	// A declared AsyncSpec constrains the pauses a tool may emit. This
	// applies to every Tool implementation (SpecOf follows Unwrap
	// chains), not just tool.Func.
	if err := validateAwaitAgainstSpec(registered.tool, req.Name, exec.Await); err != nil {
		out.Err = err
		return out
	}

	// Enforce response-size limit on the placeholder/final Output. The
	// limit applies regardless of whether the tool paused — paused tools
	// still ship Output to the model.
	limited, limitErr := r.enforceResponseSizeLimit(exec.Output, &registered.config)
	if limitErr != nil {
		out.Err = fmt.Errorf("response size: %w", limitErr)
		return out
	}

	exec.Output = limited
	out.Execution = exec

	return out
}

// enforceResponseSizeLimit checks response size and applies a fallback
// payload when the tool output exceeds the configured token budget.
// Token count is approximated as len/4 — accurate enough for "is this
// likely to blow up the context window" gating, which is the only
// decision this method needs to make.
func (*Registry) enforceResponseSizeLimit(result json.RawMessage, config *Config) (json.RawMessage, error) {
	if config.MaxResponseTokens <= 0 || len(result) == 0 {
		return result, nil
	}

	estimatedTokens := len(result) / 4
	if estimatedTokens <= config.MaxResponseTokens {
		return result, nil
	}

	message := config.ResponseTooLargeMessage
	if message == "" {
		message = ErrToolResponseTooLarge.Error()
	}

	fallback := map[string]any{
		"error":   "response_too_large",
		"message": message,
		"details": map[string]any{
			"estimated_tokens": estimatedTokens,
			"max_tokens":       config.MaxResponseTokens,
		},
	}

	return json.Marshal(fallback)
}

// BatchOption configures RunAll / ExecuteAll behavior.
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
		concurrency: 0,
	}
}

// validateAwaitAgainstSpec checks a returned Await against the tool's
// declared AsyncSpec, if any.
func validateAwaitAgainstSpec(t Tool, name string, a *Await) error {
	if a == nil {
		return nil
	}

	spec, ok := SpecOf(t)
	if !ok || spec.Async == nil {
		return nil
	}

	if a.Reason != spec.Async.Reason {
		return fmt.Errorf("%w: tool %q await reason %q does not match declared %q",
			ErrAwaitInvalid, name, a.Reason, spec.Async.Reason)
	}

	if a.Resume != spec.Async.Resume {
		return fmt.Errorf("%w: tool %q await resume %q does not match declared %q",
			ErrAwaitInvalid, name, a.Resume, spec.Async.Resume)
	}

	return nil
}
