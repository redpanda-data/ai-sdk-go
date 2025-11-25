package agent

import (
	"github.com/rs/xid"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// InvocationMetadata holds invocation-specific state and metadata.
//
// This type separates invocation domain data from context.Context control flow,
// following the idiomatic Go pattern of passing context and domain objects separately.
//
// # Mutability Contract
//
// IMMUTABLE FIELDS (framework-managed, read-only for interceptors):
//   - InvocationID() - set once at creation, uniquely identifies this invocation
//   - Turn() - incremented by framework between turns
//   - TotalUsage() - accumulated by framework after model calls
//
// MUTABLE FIELDS (safe for interceptor use):
//   - Metadata() - use for interceptor-to-interceptor communication
//   - Session() - can be modified, but be careful with message history
//
// Interceptors should NOT attempt to mutate immutable fields.
// They are unexported and only accessible via getter methods.
//
// # Lifecycle
//
// InvocationMetadata is created by Runner at the start of each execution and
// passed to Agent.Run() along with context.Context. The agent and interceptors
// read from and write to the referenced session and metadata.
//
// An invocation begins with a user message and continues until:
//   - The agent completes naturally (FinishReasonComplete)
//   - Maximum turns are reached (FinishReasonMaxTurns)
//   - An error occurs (FinishReasonError)
//   - Execution is canceled (FinishReasonCanceled)
//
// # Per-Invocation vs Per-Session State
//
// InvocationMetadata tracks per-invocation state (Turn, TotalUsage) that is
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
type InvocationMetadata struct {
	// Immutable fields (unexported, accessed via getters)
	invocationID string
	turn         int
	totalUsage   llm.TokenUsage

	// Session reference - accessible but modifications should be careful
	session *session.State

	// Mutable metadata for interceptor communication
	metadata map[string]any
}

// NewInvocationMetadata creates a new invocation metadata.
//
// The invocation ID is automatically generated. The provided session reference
// is stored for state access. Turn and usage start at zero.
//
// This function is typically called by Runner at the start of execution.
func NewInvocationMetadata(sess *session.State) *InvocationMetadata {
	return &InvocationMetadata{
		invocationID: generateInvocationID(),
		session:      sess,
		turn:         0,
		totalUsage:   llm.TokenUsage{},
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
//
// This field is immutable and set once at creation.
func (m *InvocationMetadata) InvocationID() string {
	return m.invocationID
}

// Session returns the session state.
//
// The session contains persistent conversation data:
//   - ID: Session identifier
//   - Messages: Conversation history
//   - Metadata: Session-level metadata
//
// Note: Turn and TotalUsage are per-invocation state on InvocationMetadata,
// not part of the persisted session.
//
// WARNING: The session is mutable. Interceptors can modify it (e.g., injecting
// system messages), but should be careful not to corrupt message history or
// violate framework invariants.
func (m *InvocationMetadata) Session() *session.State {
	return m.session
}

// Turn returns the current turn number for this invocation.
//
// Turns are 0-indexed. A turn represents one complete cycle of:
// model call -> tool execution -> next model call.
//
// This field is immutable from the perspective of interceptors.
// The framework increments it between turns via unexported methods.
func (m *InvocationMetadata) Turn() int {
	return m.turn
}

// TotalUsage returns the cumulative token usage for this invocation.
//
// This includes all model calls made during this invocation.
// Usage is reset to 0 at the start of each new invocation.
//
// This field is immutable from the perspective of interceptors.
// The framework updates it after each model call via unexported methods.
func (m *InvocationMetadata) TotalUsage() llm.TokenUsage {
	return m.totalUsage
}

// GetMetadata retrieves a metadata value by key.
//
// Returns nil if the key doesn't exist. Metadata is safe for interceptors
// to read and write for interceptor-to-interceptor communication.
//
// Example use cases:
//   - Auth interceptor sets user_id, other interceptors read it
//   - Tracing interceptor sets trace_id for logging
//   - Rate limiting interceptor sets rate_limit_remaining
func (m *InvocationMetadata) GetMetadata(key string) any {
	return m.metadata[key]
}

// SetMetadata sets a metadata value for interceptor communication.
//
// This is safe for interceptors to call. Use metadata to pass information
// between interceptors in the chain.
//
// Example:
//   - inv.SetMetadata("user_id", "user-123")
//   - inv.SetMetadata("trace_id", span.SpanContext().TraceID().String())
func (m *InvocationMetadata) SetMetadata(key string, value any) {
	m.metadata[key] = value
}

// Metadata returns the entire metadata map.
//
// Interceptors can read from and write to this map directly.
// The map is safe for concurrent access from a single invocation's
// goroutine (interceptors are called sequentially in a chain).
func (m *InvocationMetadata) Metadata() map[string]any {
	return m.metadata
}

// --- Internal mutators (unexported, framework use only) ---

// incrementTurn advances to the next turn.
//
// This is called by the agent after completing a turn's tool executions
// and before making the next model call. It is not exported because
// interceptors should not modify the turn counter.
func (m *InvocationMetadata) incrementTurn() {
	m.turn++
}

// addUsage accumulates token usage from a model call.
//
// This should be called after each model generation to track cumulative
// token consumption for this invocation. It is not exported because
// interceptors should not modify usage tracking.
func (m *InvocationMetadata) addUsage(usage *llm.TokenUsage) {
	m.totalUsage.InputTokens += usage.InputTokens
	m.totalUsage.OutputTokens += usage.OutputTokens
	m.totalUsage.ReasoningTokens += usage.ReasoningTokens
	m.totalUsage.CachedTokens += usage.CachedTokens
	m.totalUsage.TotalTokens += usage.TotalTokens
	// MaxInputTokens is a model constraint, not cumulative - use the first non-zero value
	if m.totalUsage.MaxInputTokens == 0 && usage.MaxInputTokens > 0 {
		m.totalUsage.MaxInputTokens = usage.MaxInputTokens
	}
}

// IncrementTurn advances the turn counter.
//
// This is a framework function for use by agent implementations.
// It should be called after completing a turn's tool executions
// and before making the next model call.
func IncrementTurn(inv *InvocationMetadata) {
	inv.incrementTurn()
}

// AddUsage accumulates token usage from a model call.
//
// This is a framework function for use by agent implementations.
// It should be called after each model generation to track
// cumulative token consumption for this invocation.
func AddUsage(inv *InvocationMetadata, usage *llm.TokenUsage) {
	inv.addUsage(usage)
}

// generateInvocationID generates a unique invocation ID.
func generateInvocationID() string {
	return "inv-" + xid.New().String()
}
