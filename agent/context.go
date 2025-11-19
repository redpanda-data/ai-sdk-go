package agent

import (
	"context"

	"github.com/rs/xid"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// InvocationContext provides the execution context for an agent invocation.
//
// It embeds context.Context for standard Go context functionality (cancellation,
// deadlines, values) and provides access to the session state and invocation metadata.
//
// InvocationContext is created by Runner at the start of each execution and passed
// to Agent.Run(). The agent reads from and writes to the referenced session.
//
// Lifecycle: An invocation begins with a user message and continues until:
//   - The agent completes naturally (FinishReasonComplete)
//   - Maximum turns are reached (FinishReasonMaxTurns)
//   - An error occurs (FinishReasonError)
//   - Execution is canceled (FinishReasonCanceled)
//
// An invocation can span multiple user messages if the agent requires input
// (e.g., require_input tool). In this case, the same invocation ID persists
// across Continue() calls.
//
// Per-Invocation vs Per-Session State:
// InvocationContext tracks per-invocation state (Turn, TotalUsage) that is
// ephemeral and not persisted. The Session tracks persistent conversation history.
//
//   - Turn: Resets to 0 for each invocation
//   - TotalUsage: Resets to 0 for each invocation
//   - Session.Messages: Accumulates across invocations
//
// This separation enables:
//   - "This request cost X tokens" (per-invocation usage)
//   - "This conversation has N messages" (per-session history)
//   - Agent handoffs (same session, different agent)
//   - Protocol adapters (map invocation to protocol lifecycle)
type InvocationContext struct {
	context.Context //nolint:containedctx // Intentional: InvocationContext is both a context and a state container

	// invocationID uniquely identifies this invocation
	invocationID string

	// session references the persistent conversation state
	session *session.State

	// turn is the current turn number for this invocation (0-indexed)
	// A turn represents one model call + tool execution cycle
	turn int

	// totalUsage tracks cumulative token consumption for this invocation
	totalUsage llm.TokenUsage

	// metadata holds invocation-specific data (feature flags, etc.)
	metadata map[string]any
}

// NewInvocationContext creates a new invocation context.
//
// The invocation ID is automatically generated. The provided context.Context
// is embedded for cancellation support, and the session reference is stored
// for state access.
func NewInvocationContext(
	ctx context.Context,
	sess *session.State,
) *InvocationContext {
	return &InvocationContext{
		Context:      ctx,
		invocationID: generateInvocationID(),
		session:      sess,
		metadata:     make(map[string]any),
	}
}

// InvocationID returns the unique identifier for this invocation.
//
// The invocation ID is stable across an entire invocation, including
// any pauses for user input (require_input tool). This enables:
//   - Correlating events and logs for a single invocation
//   - Protocol adapters tracking invocation state
//   - Analytics and debugging
func (ic *InvocationContext) InvocationID() string {
	return ic.invocationID
}

// Session returns the session state.
//
// The session contains persistent conversation data:
//   - ID: Session identifier
//   - Messages: Conversation history
//   - Metadata: Session-level metadata
//
// Note: Turn and TotalUsage are per-invocation state on InvocationContext,
// not part of the persisted session.
func (ic *InvocationContext) Session() *session.State {
	return ic.session
}

// Metadata returns the invocation-specific metadata.
//
// This is separate from session metadata - it's scoped to this specific
// invocation and is not persisted. Use cases:
//   - Feature flags for this execution
//   - A/B test assignments
//   - Request-specific context (user IP, locale, etc.)
func (ic *InvocationContext) Metadata() map[string]any {
	return ic.metadata
}

// SetMetadata sets a metadata value for this invocation.
func (ic *InvocationContext) SetMetadata(key string, value any) {
	ic.metadata[key] = value
}

// Turn returns the current turn number for this invocation.
//
// Turns are 0-indexed. A turn represents one complete cycle of:
// model call -> tool execution -> next model call.
func (ic *InvocationContext) Turn() int {
	return ic.turn
}

// IncrementTurn advances to the next turn.
//
// This is typically called by the agent after completing a turn's
// tool executions and before making the next model call.
func (ic *InvocationContext) IncrementTurn() {
	ic.turn++
}

// TotalUsage returns the cumulative token usage for this invocation.
//
// This includes all model calls made during this invocation.
// Usage is reset to 0 at the start of each new invocation.
func (ic *InvocationContext) TotalUsage() llm.TokenUsage {
	return ic.totalUsage
}

// AddUsage accumulates token usage from a model call.
//
// This should be called after each model generation to track
// cumulative token consumption for this invocation.
func (ic *InvocationContext) AddUsage(usage *llm.TokenUsage) {
	ic.totalUsage.InputTokens += usage.InputTokens
	ic.totalUsage.OutputTokens += usage.OutputTokens
	ic.totalUsage.ReasoningTokens += usage.ReasoningTokens
	ic.totalUsage.CachedTokens += usage.CachedTokens
	ic.totalUsage.TotalTokens += usage.TotalTokens
	// MaxInputTokens is a model constraint, not cumulative - use the first non-zero value
	if ic.totalUsage.MaxInputTokens == 0 && usage.MaxInputTokens > 0 {
		ic.totalUsage.MaxInputTokens = usage.MaxInputTokens
	}
}

// generateInvocationID generates a unique invocation identifier.
//
// Format: "inv-" + UUID v4
// This follows ADK's pattern of "e-" + UUID but uses "inv-" for clarity.
func generateInvocationID() string {
	return "inv-" + xid.New().String()
}
