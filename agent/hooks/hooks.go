// Package hooks provides extensible hook points for observing and modifying agent execution.
//
// The hook system uses a marker interface pattern where hooks implement only the interfaces
// they care about (HookBeforeInvocation, HookBeforeModelCall, etc.). This enables plugin
// architectures, state sharing between hook points, and clean composition.
//
// See individual hook interfaces below for available hook points and usage examples.
package hooks

import (
	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Hook is a marker interface for all hooks.
//
// Hooks can implement any combination of the hook interfaces defined in this package.
// The runner and agent check at runtime which interfaces your hook implements and call
// them at the appropriate points during execution.
//
// This allows you to:
//   - Only hook into behavior you care about
//   - Share state between related hook points
//   - Compose multiple hooks cleanly
//
// Hooks must be safe for concurrent use. It is expected that hooks are fast; if a hook
// needs to take time, copy what you need and ensure the hook is async.
//
// See the individual hook interfaces for details on when they are called and what they can do.
type Hook any

// NOTE: When adding new hook interfaces, update implementsAnyHook below.

// HookBeforeInvocation is called before agent execution starts.
// Use for audit logging, rate limiting, session validation, or feature flag setup.
// Returning an error rejects the invocation.
type HookBeforeInvocation interface {
	// OnBeforeInvocation is called before the agent begins execution.
	// Return an error to reject the invocation.
	OnBeforeInvocation(ctx HookContext, userMessage llm.Message) error
}

// HookAfterInvocation is called after agent execution completes.
// This is observe-only for metrics, audit logging, or billing.
// For response transformation, use HookAfterModelCall instead.
type HookAfterInvocation interface {
	// OnAfterInvocation is called after execution completes.
	// Return an error only if the hook itself fails (e.g., metrics endpoint unreachable).
	OnAfterInvocation(ctx HookContext, result InvocationResult) error
}

// InvocationResult contains the complete result of an agent invocation.
type InvocationResult struct {
	// FinishReason indicates why the invocation ended.
	FinishReason agent.FinishReason

	// FinalMessage is the last assistant message from this invocation.
	// May be nil if invocation failed before generating a message.
	FinalMessage *llm.Message

	// TotalUsage is the cumulative token usage across all turns.
	TotalUsage llm.TokenUsage

	// Events contains all events emitted during this invocation.
	Events []agent.Event

	// Error is non-nil if the invocation failed.
	Error error
}

// HookBeforeTurn is called at the start of each agentic loop turn.
//
// This hook can be used for:
//   - Turn-specific logging
//   - Early stopping conditions
//   - Context window management
//
// Returning a non-empty finish reason ends the invocation early (skip this turn).
type HookBeforeTurn interface {
	// OnBeforeTurn is called at the start of each turn.
	//
	// The hook receives the turn number (0-indexed).
	//
	// Return non-empty finishReason to end invocation early.
	// Return error to terminate with error.
	// Return ("", nil) to continue turn execution.
	OnBeforeTurn(ctx HookContext, turn int) (finishReason agent.FinishReason, err error)
}

// HookAfterTurn is called after each turn completes.
//
// This hook can be used for:
//   - Per-turn metrics
//   - Cost tracking per turn
//   - Conversation quality checks
//
// Returning a non-empty finish reason ends the invocation.
type HookAfterTurn interface {
	// OnAfterTurn is called after each turn completes.
	//
	// The hook receives the turn number and usage for this turn.
	//
	// Return non-empty finishReason to end invocation.
	// Return error to terminate with error.
	// Return ("", nil) to continue to next turn.
	OnAfterTurn(ctx HookContext, turn int, turnUsage llm.TokenUsage) (finishReason agent.FinishReason, err error)
}

// HookBeforeModelCall is called before each LLM API call.
//
// This hook can be used for:
//   - LLM response caching (return cached response to skip call)
//   - Request logging
//   - Dynamic prompt injection
//   - Request validation/sanitization
//   - A/B testing (modify system prompt)
//
// The hook can skip the LLM call entirely (caching) or modify the request.
type HookBeforeModelCall interface {
	// OnBeforeModelCall is called before calling the LLM.
	//
	// The hook receives the request that will be sent to the LLM.
	//
	// Return response to skip LLM call (e.g., cache hit). The response will be used
	// as if it came from the LLM.
	//
	// Return modifiedRequest to change the request. Multiple hooks can chain
	// modifications - each hook receives the potentially-modified request from
	// previous hooks.
	//
	// Return error to terminate with error.
	//
	// Return (nil, nil, nil) to continue with original/current request.
	OnBeforeModelCall(ctx HookContext, request *llm.Request) (
		response *llm.Response,
		modifiedRequest *llm.Request,
		err error,
	)
}

// HookAfterModelCall is called after each LLM API call completes (success or failure).
//
// This hook can be used for:
//   - Response caching (store for future use)
//   - Content moderation (filter responses)
//   - Response transformation (format changes)
//   - Usage tracking
//   - Response validation
//
// The hook can replace the response or convert success to failure.
type HookAfterModelCall interface {
	// OnAfterModelCall is called after the LLM returns.
	//
	// The hook receives the response (may be nil if call failed) and any error
	// from the LLM call.
	//
	// Return response to replace the original response.
	// Return error to treat as LLM failure (original response discarded).
	// Return (nil, nil) to use original response unchanged.
	OnAfterModelCall(ctx HookContext, response *llm.Response, responseErr error) (*llm.Response, error)
}

// HookBeforeToolExecution is called before each tool execution.
//
// This hook can be used for:
//   - Tool approval/authorization
//   - Argument validation/sanitization
//   - Tool execution mocking (testing)
//   - Tool call logging
//   - Cost/rate limiting per tool
//
// The hook can skip tool execution entirely (mock result, deny) or modify arguments.
type HookBeforeToolExecution interface {
	// OnBeforeToolExecution is called before executing a tool.
	//
	// The hook receives the tool request including name, ID, and arguments.
	//
	// Return response to skip tool execution (e.g., mock result, denial).
	// The response will be sent to the LLM as if the tool executed.
	//
	// Return modifiedRequest to change the arguments. Multiple hooks can chain
	// modifications - each hook receives the potentially-modified request from
	// previous hooks.
	//
	// Return error to fail the tool execution. The error will be sent to the LLM
	// as a tool error (not a terminal error).
	//
	// Return (nil, nil, nil) to continue with original/current request.
	OnBeforeToolExecution(ctx HookContext, request *llm.ToolRequest) (
		response *llm.ToolResponse,
		modifiedRequest *llm.ToolRequest,
		err error,
	)
}

// HookAfterToolExecution is called after each tool execution completes (success or failure).
//
// This hook can be used for:
//   - Tool result logging
//   - Result transformation (format conversion)
//   - Sensitive data masking
//   - Error enrichment
//   - Tool execution metrics
//
// The hook can replace the result or convert success to failure.
type HookAfterToolExecution interface {
	// OnAfterToolExecution is called after a tool executes.
	//
	// The hook receives the original request, the response (may be nil if
	// execution failed), and any error from tool execution.
	//
	// Return response to replace the original result.
	// Return error to replace success with error (original result discarded).
	// Return (nil, nil) to use original result unchanged.
	OnAfterToolExecution(
		ctx HookContext,
		request *llm.ToolRequest,
		response *llm.ToolResponse,
		executionErr error,
	) (*llm.ToolResponse, error)
}

// ImplementsAnyHook checks if hook h implements at least one hook interface.
// Used during hook registration to catch mistakes early.
func ImplementsAnyHook(h Hook) bool {
	switch h.(type) {
	case HookBeforeInvocation,
		HookAfterInvocation,
		HookBeforeTurn,
		HookAfterTurn,
		HookBeforeModelCall,
		HookAfterModelCall,
		HookBeforeToolExecution,
		HookAfterToolExecution:
		return true
	}

	return false
}
