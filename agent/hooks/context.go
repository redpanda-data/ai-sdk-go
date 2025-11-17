package hooks

import (
	"context"
	"time"

	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// HookContext provides execution state access for hooks.
//
// Hooks modify behavior via return values, not by mutating context.
// Session() provides read access - hooks should not modify session state directly
// as that could lead to inconsistencies. Use return values from hooks to indicate
// desired changes.
//
// The metadata map is for inter-hook communication - hooks can store data that
// subsequent hooks in the chain can read. Metadata is scoped to the current
// invocation and is not persisted.
type HookContext interface {
	// context.Context for standard Go context functionality (cancellation, deadlines, values)
	context.Context

	// InvocationID returns the unique identifier for this invocation.
	// Format: "inv-" + xid
	InvocationID() string

	// SessionID returns the session identifier this invocation belongs to.
	SessionID() string

	// Turn returns the current turn number (0-indexed).
	// A turn represents one complete cycle: model call -> tool execution -> next model call.
	Turn() int

	// At returns when this hook was invoked (UTC).
	// Useful for timing/duration calculations.
	At() time.Time

	// Session returns read-only access to the session state.
	//
	// IMPORTANT: Hooks should NOT modify session state directly. Use hook return
	// values to indicate desired changes (e.g., return a modified message from
	// AfterModelCall hook). Direct mutation could lead to inconsistent state.
	Session() *session.State

	// Metadata returns the metadata map for inter-hook communication.
	//
	// Hooks can store arbitrary data in metadata to pass information to subsequent
	// hooks in the chain. For example, a BeforeModelCall hook might store a cache
	// key that an AfterModelCall hook then uses to cache the response.
	//
	// Metadata is scoped to the current invocation and is not persisted.
	Metadata() map[string]any

	// SetMetadata sets a metadata value.
	//
	// This is the primary way hooks communicate with each other. Use this to
	// pass data between related hook points (e.g., BeforeModelCall -> AfterModelCall).
	SetMetadata(key string, value any)
}

// hookContext is the internal implementation of HookContext.
type hookContext struct {
	context.Context
	invocationID string
	sessionID    string
	turn         int
	at           time.Time
	session      *session.State
	metadata     map[string]any
}

// NewHookContext creates a new hook context.
//
// This is typically called by the runner or agent at the start of each hook point
// to provide hooks with current execution state.
func NewHookContext(
	ctx context.Context,
	invocationID string,
	sessionID string,
	turn int,
	at time.Time,
	session *session.State,
) HookContext {
	return &hookContext{
		Context:      ctx,
		invocationID: invocationID,
		sessionID:    sessionID,
		turn:         turn,
		at:           at,
		session:      session,
		metadata:     make(map[string]any),
	}
}

// InvocationID returns the unique identifier for this invocation.
func (c *hookContext) InvocationID() string {
	return c.invocationID
}

// SessionID returns the session identifier.
func (c *hookContext) SessionID() string {
	return c.sessionID
}

// Turn returns the current turn number.
func (c *hookContext) Turn() int {
	return c.turn
}

// At returns when this hook was invoked.
func (c *hookContext) At() time.Time {
	return c.at
}

// Session returns the session state.
func (c *hookContext) Session() *session.State {
	return c.session
}

// Metadata returns the metadata map.
func (c *hookContext) Metadata() map[string]any {
	return c.metadata
}

// SetMetadata sets a metadata value.
func (c *hookContext) SetMetadata(key string, value any) {
	c.metadata[key] = value
}
