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
//	    req *llm.ToolRequest,
//	    next agent.ToolExecutionNext,
//	) (*llm.ToolResponse, error) {
//	    t.logger.Info("tool starting", "name", req.Name, "id", req.ID)
//	    resp, err := next(ctx, req)
//	    if err != nil {
//	        t.logger.Error("tool failed", "name", req.Name, "error", err)
//	    } else {
//	        t.logger.Info("tool completed", "name", req.Name)
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

// TurnNext is the continuation function for turn interception.
type TurnNext func(ctx context.Context) (FinishReason, error)

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
	// The turn number is available via ctx (implementation-specific).
	InterceptTurn(ctx context.Context, next TurnNext) (FinishReason, error)
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
type ModelInterceptor interface {
	// InterceptModel wraps the model's generation behavior.
	//
	// The interceptor receives a ModelCallHandler (just Generate/GenerateEvents)
	// and returns a wrapped version that implements both methods. The framework
	// handles combining the wrapped handler with the original model metadata.
	//
	// The wrapper MUST implement both Generate() and GenerateEvents() to handle
	// both synchronous and streaming execution paths.
	//
	// The wrapper MAY:
	//   - Share logic between Generate and GenerateEvents
	//   - Modify requests before passing to next
	//   - Transform responses/events after receiving from next
	//   - Skip calling next entirely (e.g., cache hit)
	//   - Call next multiple times (e.g., retries)
	InterceptModel(ctx context.Context, req *llm.Request, next ModelCallHandler) ModelCallHandler
}

// ToolExecutionNext is the continuation function for tool execution interception.
type ToolExecutionNext func(ctx context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error)

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
	// You can:
	//   - Modify req before passing to next
	//   - Skip calling next (e.g., deny execution, return mock result)
	//   - Call next multiple times (e.g., retries)
	//   - Transform the response after next returns
	//
	// Tool execution errors are not terminal - they are sent to the LLM as tool errors.
	// Return an error to indicate the tool failed; the error message will be sent to the LLM.
	//
	// IMPORTANT: Always pass ctx (or a child context) to next, never context.Background().
	InterceptToolExecution(ctx context.Context, req *llm.ToolRequest, next ToolExecutionNext) (*llm.ToolResponse, error)
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
	req *llm.Request,
	base llm.Model,
	interceptors []Interceptor,
) llm.Model {
	// Start with base model as the handler
	var handler ModelCallHandler = base

	// Apply interceptors in reverse order (first interceptor = outermost wrapper)
	for i := len(interceptors) - 1; i >= 0; i-- {
		if ic, ok := interceptors[i].(ModelInterceptor); ok {
			next := handler

			handler = ic.InterceptModel(ctx, req, next)
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
	_ context.Context,
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
			turnFunc = func(ctx context.Context) (FinishReason, error) {
				return ic.InterceptTurn(ctx, next)
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
	_ context.Context,
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
			executor = func(ctx context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
				return ic.InterceptToolExecution(ctx, req, next)
			}
		}
	}

	return executor
}
