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
// Two execution surfaces are exposed:
//
//   - Run / RunAll return tool.ExecutionResult, the typed control-flow
//     return that carries Execution{Output, Await, Actions} plus a
//     runtime error. The agent loop and runner use these to detect
//     pauses, persist pending state, and emit ToolPendingEvent.
//   - Execute / ExecuteAll are convenience wrappers that reconcile an
//     ExecutionResult into an llm.ToolResponsePart for callers that only
//     care about the model-visible payload (most tests, ad-hoc tool
//     invocations). They ignore Await: a paused tool surfaces as a
//     placeholder response.
type Registry interface {
	// Register adds a tool to the registry with optional configuration.
	Register(tool Tool, opts ...Option) error

	// Unregister removes a tool by name.
	Unregister(name string) error

	// List returns tool definitions for use in llm.Request.Tools. The
	// definitions tell the model what tools are available; they include
	// AsyncSpec-derived hints where set.
	List() []llm.ToolDefinition

	// Get retrieves a registered tool by name.
	Get(name string) (Tool, error)

	// Run executes a single tool request and returns the typed
	// ExecutionResult. Err is populated for runtime/tool errors (the
	// returned Execution is the zero value in that case); pauses appear
	// as Execution.Await != nil.
	Run(ctx context.Context, inv InvocationInfo, req *llm.ToolRequestPart) ExecutionResult

	// Resume re-enters a registered tool after a ResumeWithReentry pause,
	// with Call.Resume populated from payload. It applies the same
	// timeout, Await validation, and response-size limits as Run.
	Resume(ctx context.Context, inv InvocationInfo, req *llm.ToolRequestPart, payload *ResumePayload) ExecutionResult

	// RunAll executes multiple tool requests concurrently. Results are
	// returned in the SAME ORDER as reqs — fixing the previous
	// completion-order shape that broke provider tool-call ordering.
	RunAll(ctx context.Context, inv InvocationInfo, reqs []*llm.ToolRequestPart, opts ...BatchOption) []ExecutionResult

	// Execute reconciles Run's ExecutionResult into a single
	// llm.ToolResponsePart. Errors are encoded via IsError + a
	// `{"error":"..."}` payload. Returns (nil, ErrToolRequestNil) for a
	// nil request — that signals a bad caller, not a tool failure.
	Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error)

	// ExecuteAll is the response-part variant of RunAll. Per-request
	// failures are encoded in the returned parts; the slice always has
	// len(reqs) entries, in request order.
	ExecuteAll(ctx context.Context, reqs []*llm.ToolRequestPart, opts ...BatchOption) []*llm.ToolResponsePart
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

// RegistryConfig configures registry-wide behavior. Reserved for future
// settings; tool-specific configuration is handled via Option.
type RegistryConfig struct{}

// registry is the concrete implementation of Registry.
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
func (r *registry) Register(t Tool, opts ...Option) error {
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
func (r *registry) List() []llm.ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()

	definitions := make([]llm.ToolDefinition, 0, len(r.tools))
	for _, registered := range r.tools {
		definitions = append(definitions, Definition(registered.tool))
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

// Run executes a single tool request and returns the typed
// ExecutionResult. See ExecutionResult for the error vs. await contract.
func (r *registry) Run(ctx context.Context, inv InvocationInfo, req *llm.ToolRequestPart) ExecutionResult {
	return r.run(ctx, inv, req, nil)
}

// Resume implements Registry.
func (r *registry) Resume(ctx context.Context, inv InvocationInfo, req *llm.ToolRequestPart, payload *ResumePayload) ExecutionResult {
	return r.run(ctx, inv, req, payload)
}

// RunAll executes multiple tool requests concurrently. Returned results
// are in request order even though tools execute concurrently.
func (r *registry) RunAll(ctx context.Context, inv InvocationInfo, reqs []*llm.ToolRequestPart, opts ...BatchOption) []ExecutionResult {
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

// Execute is the response-part wrapper around Run. See Registry docs
// for when to prefer Run vs. Execute.
func (r *registry) Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	if req == nil {
		return nil, ErrToolRequestNil
	}

	res := r.Run(ctx, InvocationInfo{}, req)

	return res.Response(), nil
}

// ExecuteAll is the response-part wrapper around RunAll.
func (r *registry) ExecuteAll(ctx context.Context, reqs []*llm.ToolRequestPart, opts ...BatchOption) []*llm.ToolResponsePart {
	results := r.RunAll(ctx, InvocationInfo{}, reqs, opts...)
	out := make([]*llm.ToolResponsePart, len(results))

	for i, res := range results {
		out[i] = res.Response()
	}

	return out
}

// run is the shared execution path for Run and Resume.
func (r *registry) run(ctx context.Context, inv InvocationInfo, req *llm.ToolRequestPart, resume *ResumePayload) ExecutionResult {
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
		Args:       req.Arguments,
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

	// Validate Await shape before persisting the pause.
	if err := exec.Await.Validate(); err != nil {
		out.Err = fmt.Errorf("%w: %w", ErrAwaitInvalid, err)
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
func (*registry) enforceResponseSizeLimit(result json.RawMessage, config *Config) (json.RawMessage, error) {
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
