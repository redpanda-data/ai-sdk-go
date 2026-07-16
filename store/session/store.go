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
//	    List(ctx context.Context, req *ListRequest) (*ListResponse, error)
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
	"time"

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

	// List returns a page of session summaries ordered by UpdatedAt
	// descending, then ID ascending (a stable total order for keyset
	// pagination). Summaries omit the message payload — use Load to fetch a
	// full session. PageToken is opaque and implementation-defined; callers
	// pass back the NextPageToken from the previous response.
	//
	// Stores whose backend cannot enumerate sessions (e.g. a plain key-value
	// backend keyed by session ID) return ErrListNotSupported.
	List(ctx context.Context, req *ListRequest) (*ListResponse, error)
}

// Metadata keys recording parent→sub-agent linkage on a session's Metadata.
//
// When a sub-agent is invoked in-process as part of a parent agent's turn it
// shares the parent's session ID (so the two group into one conversation for
// tracing/reconstruction). These keys carry the back-reference and identity
// needed to tell the sub-agent's activity apart from the parent's. Their
// presence is itself the signal that a session is a sub-agent run.
const (
	// MetadataParentInvocationID (string) references the parent agent's invocation ID.
	MetadataParentInvocationID = "parent_invocation_id"
	// MetadataAgentPath (string) identifies which sub-agent produced the session.
	MetadataAgentPath = "agent_path"
)

// State represents the persistent state of a conversation session.
//
// The Messages slice contains the conversation history excluding any system prompts,
// which are managed by the runtime. Metadata can store arbitrary session-specific
// data such as user preferences, feature flags, or tracking information.
type State struct {
	// ID is the unique identifier for this session.
	ID string `json:"id"`

	// Messages contains the conversation history (excluding system prompts).
	// The slice should be treated as append-only to maintain temporal ordering.
	Messages []llm.Message `json:"messages"`

	// Metadata contains arbitrary key-value pairs associated with the session.
	// Common uses include user settings, locale, feature flags, and analytics data.
	Metadata map[string]any `json:"metadata,omitempty"`

	// UpdatedAt is the time the session was last persisted. It is
	// storage-managed and output-only: Save ignores whatever value it holds
	// (the store stamps the write time), while Load and List populate it.
	// Stores without a clock may leave it zero.
	UpdatedAt time.Time `json:"updated_at,omitzero"`
}

// Clone creates a deep copy of the session state.
// Returns nil if the receiver is nil. Each Message in s.Messages is
// duplicated via llm.CloneMessage so persisted state never shares mutable
// pointers (Part interfaces, json.RawMessage slices, metadata maps) with the
// live caller.
func (s *State) Clone() *State {
	if s == nil {
		return nil
	}

	clone := &State{
		ID:        s.ID,
		Messages:  make([]llm.Message, len(s.Messages)),
		Metadata:  make(map[string]any, len(s.Metadata)),
		UpdatedAt: s.UpdatedAt,
	}

	for i, msg := range s.Messages {
		clone.Messages[i] = llm.CloneMessage(msg)
	}

	maps.Copy(clone.Metadata, s.Metadata)

	return clone
}

// ErrNotFound indicates that the requested session does not exist.
var ErrNotFound = errors.New("session: not found")

// ErrListNotSupported indicates that a Store's backend cannot enumerate
// sessions (e.g. a plain key-value backend keyed by session ID). Such stores
// return it from List.
var ErrListNotSupported = errors.New("session: listing not supported by this store")

// Summary is the list view of a session: its identity and metadata without
// the message payload. It is what List returns per session, keeping list
// queries cheap regardless of conversation size — fetch the messages with
// Load only when a specific session is opened.
type Summary struct {
	// ID is the unique identifier for the session.
	ID string

	// Metadata is the session's metadata, as in State.
	Metadata map[string]any

	// UpdatedAt is the time the session was last persisted.
	UpdatedAt time.Time
}

// ListRequest selects a page of session summaries.
type ListRequest struct {
	// PageSize is the maximum number of summaries to return. A non-positive
	// value lets the store apply its own default; stores may cap it.
	PageSize int32

	// PageToken continues a previous List; empty starts from the first page.
	// The token is opaque and must come from a prior ListResponse.
	PageToken string
}

// ListResponse is one page of session summaries.
type ListResponse struct {
	// Sessions is the page, ordered by UpdatedAt descending, ID ascending.
	Sessions []Summary

	// NextPageToken is the token for the following page, or empty when this
	// is the last page.
	NextPageToken string
}
