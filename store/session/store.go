// Package session provides interfaces and implementations for managing agent conversation sessions.
//
// A session represents the persistent state of a conversation between a user and an agent,
// including the message history and associated metadata. Session stores are responsible for
// loading, saving, and deleting these sessions.
//
// # Core Interface
//
// The Store interface defines the basic operations for session persistence:
//
//	type Store interface {
//	    Load(ctx context.Context, sessionID string) (*State, error)
//	    Save(ctx context.Context, state *State) error
//	    Delete(ctx context.Context, sessionID string) error
//	}
//
// # Concurrency
//
// Implementations must be safe for concurrent use. The InMemoryStore implementation uses
// a read-write mutex for this purpose. When multiple goroutines access the same sessionID
// concurrently, the last write wins.
//
// # Implementations
//
// The package provides:
//   - InMemoryStore: A simple in-memory implementation suitable for development and testing
//
// Additional implementations (PostgreSQL, Redis, Kafka, etc.) can be provided by implementing
// the Store interface.
package session

import (
	"context"
	"errors"
	"maps"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Store manages the persistence of agent conversation sessions.
//
// All methods must be safe for concurrent use. Implementations should handle
// context cancellation appropriately and return context.Canceled or context.DeadlineExceeded
// when the context is cancelled or times out.
type Store interface {
	// Load retrieves a session by its ID.
	// Returns ErrNotFound if the session does not exist.
	Load(ctx context.Context, sessionID string) (*State, error)

	// Save persists the given session state.
	// If a session with the same ID already exists, it is completely replaced.
	// Implementations should store a copy of the state to prevent modifications
	// after Save returns.
	Save(ctx context.Context, state *State) error

	// Delete removes a session by its ID.
	// Returns nil if the session doesn't exist (idempotent).
	Delete(ctx context.Context, sessionID string) error
}

// State represents the persistent state of a conversation session.
//
// The Messages slice contains the conversation history excluding any system prompts,
// which are managed by the runtime. Metadata can store arbitrary session-specific
// data such as user preferences, feature flags, or tracking information.
type State struct {
	// ID is the unique identifier for this session.
	ID string

	// Messages contains the conversation history (excluding system prompts).
	// The slice should be treated as append-only to maintain temporal ordering.
	Messages []llm.Message

	// Metadata contains arbitrary key-value pairs associated with the session.
	// Common uses include user settings, locale, feature flags, and analytics data.
	Metadata map[string]any
}

// Clone creates a deep copy of the session state.
// Returns nil if the receiver is nil.
func (s *State) Clone() *State {
	if s == nil {
		return nil
	}

	clone := &State{
		ID:       s.ID,
		Messages: make([]llm.Message, len(s.Messages)),
		Metadata: make(map[string]any, len(s.Metadata)),
	}
	copy(clone.Messages, s.Messages)
	maps.Copy(clone.Metadata, s.Metadata)

	return clone
}

// ErrNotFound indicates that the requested session does not exist.
var ErrNotFound = errors.New("session: not found")
