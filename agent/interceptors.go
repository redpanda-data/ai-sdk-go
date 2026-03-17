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

package agent

import (
	"context"
	"iter"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Interceptor is a marker interface for all interceptors.
//
// Interceptors can implement any combination of the interceptor interfaces defined in this package
// (TurnInterceptor, ModelInterceptor, ToolInterceptor). The agent checks at runtime which
// interfaces your interceptor implements and calls them at the appropriate points during execution.
//
// This allows you to:
//   - Only intercept behavior you care about
//   - Share state between related interception points (e.g., tracer instance)
//   - Compose multiple interceptors cleanly
//
// IMPORTANT: Interceptors must be safe for concurrent use. Multiple goroutines may call your
// methods simultaneously. Use immutable state or proper synchronization.
//
// Interceptors use an interceptor pattern where each interceptor wraps the execution and can:
//   - Modify inputs before passing to next handler
//   - Skip execution entirely (e.g., caching)
//   - Retry by calling next multiple times
//   - Transform outputs after execution
//
// # Example
//
// A simple logging interceptor that tracks tool executions:
//
//	type ToolLogger struct {
//	    logger *slog.Logger
//	}
//
//	func (t *ToolLogger) InterceptToolExecution(
//	    ctx context.Context,
//	    info *agent.ToolCallInfo,
//	    next agent.ToolExecutionNext,
//	) (*llm.ToolResponse, error) {
//	    t.logger.Info("tool starting", "name", info.Req.Name, "id", info.Req.ID)
//	    resp, err := next(ctx, info)
//	    if err != nil {
//	        t.logger.Error("tool failed", "name", info.Req.Name, "error", err)
//	    } else {
//	        t.logger.Info("tool completed", "name", info.Req.Name)
//	    }
//	    return resp, err
//	}
//
//	// Register with agent
//	agent, _ := llmagent.New(
//	    "assistant",
//	    "You are helpful",
//	    model,
//	    llmagent.WithInterceptors(&ToolLogger{logger: slog.Default()}),
//	)
//
// See the individual interceptor interfaces for details on when they are called and what they can do.
type Interceptor any

// NOTE: When adding new interceptor interfaces, update ImplementsAnyInterceptor below.

// TurnInfo contains context for turn interception.
// New fields can be added without breaking existing interceptor implementations.
type TurnInfo struct {
	// Inv provides invocation metadata (session, turn, usage, custom metadata).
	Inv *InvocationMetadata
}

// TurnNext is the continuation function for turn interception.
type TurnNext func(ctx context.Context, info *TurnInfo) (FinishReason, error)

// TurnInterceptor intercepts individual turns in the agentic loop.
//
// This interceptor wraps each turn (one iteration of the agent loop). It can:
//   - Add per-turn logging or tracing
//   - Implement early stopping conditions
//   - Manage context windows
//   - Track per-turn costs
//
// Use cases:
//   - Turn-specific logging
//   - Early stopping conditions
//   - Context window management
//   - Per-turn metrics
//   - Conversation quality checks
type TurnInterceptor interface {
	// InterceptTurn wraps a single turn execution.
	//
	// Return a non-empty FinishReason without calling next to end the invocation early.
	// Call next to continue turn execution.
	//
	// Parameters:
	//   - ctx: Standard Go context for cancellation, deadlines, and request-scoped values
	//   - info: Turn context (invocation metadata); new fields may be added in future versions
	//   - next: Continuation function to call the next interceptor or the base turn execution
	//
	// Interceptors can:
	//   - Derive a new context (e.g., add timeout) and pass it to next
	//   - Read from info.Inv (turn number, session, metadata set by previous interceptors)
	//   - Write to info.Inv.Metadata() to pass data to subsequent interceptors
	//   - Modify info.Inv.Session() (e.g., inject system messages) - use with caution
	InterceptTurn(ctx context.Context, info *TurnInfo, next TurnNext) (FinishReason, error)
}

// ModelCallHandler represents the behavioral surface of an LLM model, separating
// generation logic (Generate/GenerateEvents) from identity metadata (Name/Capabilities).
//
// This separation allows ModelInterceptor to wrap generation behavior without requiring
// interceptors to implement the identity methods, eliminating boilerplate. The framework
// automatically combines the wrapped handler with the base model's identity.
type ModelCallHandler interface {
	llm.Generator       // Generate(ctx context.Context, req *Request) (*Response, error)
	llm.EventsGenerator // GenerateEvents(ctx context.Context, req *Request) iter.Seq2[Event, error]
}

// ModelCallInfo contains context for model call interception.
// New fields can be added without breaking existing interceptor implementations.
type ModelCallInfo struct {
	// InvocationMetadata provides invocation metadata (session, turn, usage, custom metadata).
	InvocationMetadata *InvocationMetadata

	// Model provides read-only access to model identity (Name, Capabilities, Constraints).
	// Use this for observability (e.g., OTel span attributes) or routing decisions.
	Model llm.ModelInfo

	// Req is the LLM request to be sent to the model.
	Req *llm.Request
}

// ModelInterceptor intercepts LLM API calls for both synchronous and streaming generation.
//
// This interceptor wraps the model's generation behavior (Generate/GenerateEvents)
// without requiring implementation of metadata methods (Name/Capabilities).
// The interceptor MUST handle both synchronous and streaming paths explicitly
// by returning a handler that implements both methods.
//
// Use cases:
//   - Content Redaction: Modify content in both sync and streaming (primary use case)
//   - Response Caching: Cache responses and replay for both modes
//   - Request Validation: Validate before either path
//   - Dynamic Routing: Choose model based on request
//   - A/B Testing: Modify prompts or route to different models
//   - Observability: Access model name/capabilities for tracing attributes
type ModelInterceptor interface {
	// InterceptModel wraps the model's generation behavior.
	//
	// The interceptor receives model call context and a ModelCallHandler (just Generate/GenerateEvents)
	// and returns a wrapped version that implements both methods. The framework
	// handles combining the wrapped handler with the original model metadata.
	//
	// Parameters:
	//   - ctx: Standard Go context for cancellation, deadlines, and request-scoped values
	//   - info: Model call context (invocation, model identity, request); new fields may be added
	//   - next: The next model handler in the chain
	//
	// The wrapper MUST implement both Generate() and GenerateEvents() to handle
	// both synchronous and streaming execution paths.
	//
	// The wrapper MAY:
	//   - Share logic between Generate and GenerateEvents
	//   - Modify info.Req before passing to next
	//   - Transform responses/events after receiving from next
	//   - Skip calling next entirely (e.g., cache hit)
	//   - Call next multiple times (e.g., retries)
	//   - Read from info.Inv (turn number, session, metadata set by previous interceptors)
	//   - Write to info.Inv.Metadata() to pass data to subsequent interceptors
	//   - Read info.Model.Name(), info.Model.Capabilities() for observability/routing
	InterceptModel(ctx context.Context, info *ModelCallInfo, next ModelCallHandler) ModelCallHandler
}

// ToolCallInfo contains context for tool execution interception.
// New fields can be added without breaking existing interceptor implementations.
type ToolCallInfo struct {
	// Inv provides invocation metadata (session, turn, usage, custom metadata).
	Inv *InvocationMetadata

	// Req is the tool request from the LLM.
	Req *llm.ToolRequest

	// Definition is the full tool definition including description and type.
	// Used by interceptors for observability (e.g., OpenTelemetry attributes).
	// May be nil if tool definition is not available.
	Definition *llm.ToolDefinition
}

// ToolExecutionNext is the continuation function for tool execution interception.
type ToolExecutionNext func(ctx context.Context, info *ToolCallInfo) (*llm.ToolResponse, error)

// ToolInterceptor intercepts tool executions.
//
// This interceptor wraps tool calls requested by the LLM. It has full control over
// the execution and can:
//   - Authorize or deny tool execution
//   - Validate or sanitize arguments
//   - Mock tool results (testing)
//   - Transform results
//   - Implement tool-specific retries
//
// Use cases:
//   - Tool approval/authorization
//   - Argument validation/sanitization
//   - Tool execution mocking (testing)
//   - Tool call logging
//   - Cost/rate limiting per tool
//   - Result transformation
//   - Sensitive data masking
type ToolInterceptor interface {
	// InterceptToolExecution wraps a tool execution.
	//
	// Parameters:
	//   - ctx: Standard Go context for cancellation, deadlines, and request-scoped values
	//   - info: Tool call context (invocation metadata, tool request); new fields may be added
	//   - next: Continuation function to call the next interceptor or the base tool execution
	//
	// You can:
	//   - Modify info.Req before passing to next
	//   - Skip calling next (e.g., deny execution, return mock result)
	//   - Call next multiple times (e.g., retries)
	//   - Transform the response after next returns
	//   - Read from info.Inv (turn number, session, metadata set by previous interceptors)
	//   - Write to info.Inv.Metadata() to pass data to subsequent interceptors
	//
	// Tool execution errors are not terminal - they are sent to the LLM as tool errors.
	// Return an error to indicate the tool failed; the error message will be sent to the LLM.
	//
	// IMPORTANT: Always pass ctx (or a child context) to next, never context.Background().
	InterceptToolExecution(ctx context.Context, info *ToolCallInfo, next ToolExecutionNext) (*llm.ToolResponse, error)
}

// ImplementsAnyInterceptor checks if interceptor i implements at least one interceptor interface.
// Used during interceptor registration to catch mistakes early.
func ImplementsAnyInterceptor(i Interceptor) bool {
	switch i.(type) {
	case TurnInterceptor,
		ModelInterceptor,
		ToolInterceptor:
		return true
	}

	return false
}

// ApplyModelInterceptors wraps a model with the model interceptor chain.
//
// This applies ModelInterceptor interceptors to intercept Generate/GenerateEvents while
// preserving the base model's Name/Capabilities.
//
// Returns the base model unchanged if no ModelInterceptor interceptors are present.
func ApplyModelInterceptors(
	ctx context.Context,
	info *ModelCallInfo,
	base llm.Model,
	interceptors []Interceptor,
) llm.Model {
	// Start with base model as the handler
	var handler ModelCallHandler = base

	// Apply interceptors in reverse order (first interceptor = outermost wrapper)
	for i := len(interceptors) - 1; i >= 0; i-- {
		if ic, ok := interceptors[i].(ModelInterceptor); ok {
			next := handler

			handler = ic.InterceptModel(ctx, info, next)
		}
	}

	// No interception happened - return original
	if handler == base {
		return base
	}

	// Bridge: handler only has Generate/GenerateEvents, but agent needs full llm.Model
	// So we wrap it to add ModelInfo methods from the base model via embedding
	return &interceptedModel{
		ModelInfo: base,
		handler:   handler,
	}
}

// interceptedModel implements llm.Model by combining:
//   - Base model's ModelInfo methods (Name/Capabilities/Constraints) via embedding (preserved identity)
//   - Handler chain's Generate/GenerateEvents (intercepted behavior)
type interceptedModel struct {
	llm.ModelInfo // Embedded - all ModelInfo methods automatically promoted

	handler ModelCallHandler // For Generate() and GenerateEvents()
}

// Generate delegates to the intercepted handler chain.
func (m *interceptedModel) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	return m.handler.Generate(ctx, req)
}

// GenerateEvents delegates to the intercepted handler chain.
func (m *interceptedModel) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return m.handler.GenerateEvents(ctx, req)
}

// ApplyTurnInterceptors wraps a turn execution function with the turn interceptor chain.
//
// This applies TurnInterceptor interceptors to intercept individual turns in the agentic loop.
//
// Returns the base turn function unchanged if no TurnInterceptor interceptors are present.
func ApplyTurnInterceptors(
	interceptors []Interceptor,
	baseTurn TurnNext,
) TurnNext {
	// Start with base turn function
	turnFunc := baseTurn

	// Apply interceptors in reverse order (first interceptor = outermost wrapper)
	for i := len(interceptors) - 1; i >= 0; i-- {
		if interceptor, ok := interceptors[i].(TurnInterceptor); ok {
			next := turnFunc
			ic := interceptor

			// Create a wrapper that calls the interceptor
			turnFunc = func(ctx context.Context, info *TurnInfo) (FinishReason, error) {
				return ic.InterceptTurn(ctx, info, next)
			}
		}
	}

	return turnFunc
}

// ApplyToolInterceptors wraps a tool execution function with the tool interceptor chain.
//
// This applies ToolInterceptor interceptors to intercept tool execution.
//
// Returns the base executor unchanged if no ToolInterceptor interceptors are present.
func ApplyToolInterceptors(
	interceptors []Interceptor,
	baseExecutor ToolExecutionNext,
) ToolExecutionNext {
	// Start with base executor
	executor := baseExecutor

	// Apply interceptors in reverse order (first interceptor = outermost wrapper)
	for i := len(interceptors) - 1; i >= 0; i-- {
		if interceptor, ok := interceptors[i].(ToolInterceptor); ok {
			next := executor
			ic := interceptor

			// Create a wrapper that calls the interceptor
			executor = func(ctx context.Context, info *ToolCallInfo) (*llm.ToolResponse, error) {
				return ic.InterceptToolExecution(ctx, info, next)
			}
		}
	}

	return executor
}
