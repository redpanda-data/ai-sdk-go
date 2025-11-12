// Package agent provides the core interfaces and types for agent execution.
//
// Key concepts:
//   - Agent: Stateless executor implementing a specific execution strategy
//   - InvocationContext: Per-invocation state (turn, usage) and execution handle
//   - Event: Sealed interface for all runtime events yielded during execution
//
// All agent implementations yield events during execution for real-time observability.
// The event stream always ends with InvocationEndEvent.
package agent

import (
	"iter"
)

// Agent is the core execution interface. Different agent types
// (LLM-based, remote, sequential, etc.) implement this interface.
//
// Agents are stateless executors that operate on an InvocationContext.
// The context provides access to the session state and other execution data.
//
// # Event Streaming
//
// Agent.Run() yields events during execution, following the same pattern
// as Runner.Run(). This enables:
//   - Real-time progress updates
//   - Streaming token deltas
//   - Tool execution visibility
//   - Protocol adapter integration
//
// The event stream ends with a terminal event (InvocationEndEvent).
type Agent interface {
	// Name returns the agent's identifier.
	Name() string

	// Description describes the agent's purpose and capabilities.
	Description() string

	// Run executes the agent with the given invocation context.
	//
	// The agent reads from invCtx.Session() (messages, turn, usage) and
	// updates it as execution progresses. Events are yielded during
	// execution to provide real-time progress updates.
	//
	// # Error Handling
	//
	// Run uses iter.Seq2[Event, error] to distinguish two failure modes:
	//
	// Terminal Errors - yield(nil, error):
	//   - Execution cannot proceed (config invalid, auth failed, infra unavailable)
	//   - Stream terminates immediately, no InvocationEndEvent emitted
	//   - Examples: ErrToolRegistry, ErrSessionLoad, context.Canceled
	//
	// Graceful Failures - ErrorEvent + InvocationEndEvent(FinishReasonError):
	//   - Execution completes with error outcome (rate limit, content filter, max turns)
	//   - Stream ends normally with InvocationEndEvent for observability
	//   - Examples: rate limits, content policy violations, model timeouts
	//
	// On success, the stream always ends with InvocationEndEvent.
	Run(ctx *InvocationContext) iter.Seq2[Event, error]

	// InputSchema returns the expected input schema for this agent.
	// Used when wrapping agents as tools (agent-as-tool pattern).
	InputSchema() map[string]any
}
