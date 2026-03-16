package agent

import (
	"github.com/rs/xid"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// Info captures the agent's identity at the start of an invocation.
//
// This is an immutable snapshot taken when InvocationMetadata is created, providing
// stable agent identity for observability, debugging, auditing, and telemetry purposes.
//
// # What's Included
//
// Info contains only the agent's identity (name, description) which are
// not available elsewhere in the invocation context. Other agent configuration:
//   - System prompt: available in session.Messages (first system role message)
//   - Available tools: available in llm.Request.Tools (sent with each model call)
//
// # Why a Info?
//
// The live agent instance may change over time, but interceptors and telemetry systems
// need a consistent view of agent identity. This snapshot ensures that:
//   - OpenTelemetry spans have stable agent metadata (gen_ai.agent.name, gen_ai.agent.description)
//   - Debugging tools can identify which agent handled this invocation
//   - Auditing systems can track agent identity for compliance
//
// # Immutability Contract
//
// Info is immutable after construction. It represents the agent identity
// at invocation start.
//
// # Design Rationale
//
// This struct is separate from InvocationMetadata's root fields to:
//   - Keep the root namespace clean (avoid "God Object" antipattern)
//   - Clearly distinguish "agent identity" from "runtime state" (turn, usage)
//   - Provide room to evolve (add Labels, etc. if needed)
//   - Make mutability clear (entire struct is immutable)
type Info struct {
	// Name is the agent's application-defined name (used for gen_ai.agent.name).
	// It should be stable and human-readable, but does not need to be globally unique.
	Name string

	// Description is the agent's purpose and capabilities (used for gen_ai.agent.description)
	Description string

	// SystemPrompt contains the agent's system instructions (used for gen_ai.system_instructions).
	// This captures the base system prompt configured for the agent, distinct from dynamic
	// instructions that may be part of the conversation history.
	SystemPrompt string

	// ID is an optional unique agent identifier (used for gen_ai.agent.id).
	// It should be stable across invocations and uniquely identify the logical agent.
	ID string

	// Version is an optional agent version (used for gen_ai.agent.version).
	// Use a stable release or configuration version, not a per-request value.
	Version string

	// ModelName is the model identifier for LLM-based agents (used for gen_ai.request.model).
	// Empty for non-LLM agents.
	ModelName string

	// ProviderName is the provider name for LLM-based agents (used for gen_ai.provider.name).
	// Empty for non-LLM agents.
	ProviderName string
}

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

	// Immutable agent snapshot - captured at invocation start
	agent Info

	// Session reference - accessible but modifications should be careful
	session *session.State

	// Mutable metadata for interceptor communication
	metadata map[string]any
}

// NewInvocationMetadata creates a new invocation metadata with agent context.
//
// The invocation ID is automatically generated. The provided session reference
// is stored for state access. Turn and usage start at zero.
//
// The agent snapshot captures the agent's configuration (name, description, system prompt,
// available tools) at invocation start, providing a stable baseline for observability,
// debugging, and telemetry.
//
// This function is typically called by agent implementations (e.g., llmagent) at the
// start of execution. For convenience, use NewInvocationMetadataFromAgent if you have
// an Agent interface.
func NewInvocationMetadata(sess *session.State, agent Info) *InvocationMetadata {
	return &InvocationMetadata{
		invocationID: generateInvocationID(),
		agent:        agent,
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

// Agent returns the agent snapshot for this invocation.
//
// The snapshot captures the agent's configuration (name, description, system prompt,
// available tools) at invocation start. This is immutable and provides a stable
// baseline for:
//   - Observability (OpenTelemetry gen_ai.agent.* attributes)
//   - Debugging (what system prompt and tools were active?)
//   - Auditing (compliance tracking of agent configuration)
//   - Replay (reconstructing agent state for testing)
//
// Per-call variations (e.g., dynamic prompts, tool subsets per turn) should be
// tracked in ModelInterceptor spans, not here.
//
// This field is immutable and set once at creation.
func (m *InvocationMetadata) Agent() Info {
	return m.agent
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
