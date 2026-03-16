// Package agent provides the core interfaces and types for agent execution.
//
// Key concepts:
//   - Agent: Stateless executor implementing a specific execution strategy
//   - InvocationMetadata: Per-invocation state (turn, usage) and session reference
//   - Event: Sealed interface for all runtime events yielded during execution
//
// All agent implementations yield events during execution for real-time observability.
// The event stream always ends with InvocationEndEvent.
package agent

import (
	"context"
	"iter"
)

// Agent is the core execution interface. Different agent types
// (LLM-based, remote, sequential, etc.) implement this interface.
//
// Agents are stateless executors that operate with a context.Context for
// control flow and InvocationMetadata for domain state. This separation
// follows idiomatic Go patterns and prevents context derivation footguns.
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
	// Info returns the agent's identity snapshot.
	//
	// The returned Info struct contains stable, human-readable metadata
	// used by observability integrations (OpenTelemetry gen_ai.agent.*),
	// agent-as-tool wrappers, and other SDK features.
	//
	// For LLM-based agents, Info also includes ModelName and ProviderName
	// so that invocation spans can carry gen_ai.request.model and
	// gen_ai.provider.name from creation, even before a model call occurs.
	//
	// Adding fields to Info is backward-compatible (zero value).
	Info() Info

	// Run executes the agent with the given context and invocation metadata.
	//
	// The agent reads from inv.Session() (messages) and updates it as
	// execution progresses. Events are yielded during execution to provide
	// real-time progress updates.
	//
	// Parameters:
	//   - ctx: Standard Go context for cancellation, deadlines, and request-scoped values
	//   - inv: Invocation metadata (session, turn, usage, metadata)
	//
	// # Error Handling
	//
	// Run uses iter.Seq2[Event, error] following the principle:
	// "errors in events are data, errors in iterators are control flow"
	//
	// Terminal Errors - yield(nil, error):
	//   System failures that prevent continuation (control flow)
	//   Examples: ErrToolRegistry, authentication failures, session store down
	//
	// Observable Errors - ErrorEvent + InvocationEndEvent (both with nil error):
	//   Application-level errors visible to users (data)
	//   Examples: rate limits, content filters, max turns reached
	//
	// Consumer pattern:
	//   for evt, err := range agent.Run(ctx, inv) {
	//       if err != nil {
	//           // CONTROL FLOW: Fatal error, system can't continue
	//           return
	//       }
	//       switch e := evt.(type) {
	//       case agent.ErrorEvent:
	//           // DATA: Observable error for logging/display
	//       case agent.InvocationEndEvent:
	//           // Completion (check FinishReason)
	//       }
	//   }
	Run(ctx context.Context, inv *InvocationMetadata) iter.Seq2[Event, error]

	// InputSchema returns the expected input schema for this agent.
	// Used when wrapping agents as tools (agent-as-tool pattern).
	InputSchema() map[string]any
}
