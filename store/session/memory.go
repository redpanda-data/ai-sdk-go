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

package session

import (
	"context"
	"time"

	"github.com/twmb/go-cache/cache"
)

const (
	// DefaultSessionTTL is the default time-to-live for sessions in memory.
	// After this duration of inactivity, sessions are automatically expired.
	DefaultSessionTTL = 24 * time.Hour

	// DefaultCleanupInterval is how often the cache automatically cleans up expired sessions.
	DefaultCleanupInterval = 1 * time.Hour
)

// InMemoryStore provides an in-memory session storage implementation with automatic expiration.
//
// Sessions are automatically expired after a configurable TTL period of inactivity.
// This prevents unbounded memory growth in long-running applications.
//
// This implementation is suitable for development, testing, and single-instance
// deployments. For production use with multiple instances or persistence requirements,
// consider implementing a Store backed by a database or distributed cache.
//
// InMemoryStore is safe for concurrent use from multiple goroutines.
type InMemoryStore struct {
	cache *cache.Cache[string, *State]
}

// InMemoryStoreOption configures an InMemoryStore.
type InMemoryStoreOption func(*inMemoryStoreConfig)

type inMemoryStoreConfig struct {
	ttl             time.Duration
	cleanupInterval time.Duration
}

// WithSessionTTL sets the time-to-live for sessions.
// After this duration of inactivity, sessions are automatically expired.
func WithSessionTTL(ttl time.Duration) InMemoryStoreOption {
	return func(c *inMemoryStoreConfig) {
		c.ttl = ttl
	}
}

// WithCleanupInterval sets how often the cache automatically cleans up expired sessions.
func WithCleanupInterval(interval time.Duration) InMemoryStoreOption {
	return func(c *inMemoryStoreConfig) {
		c.cleanupInterval = interval
	}
}

// NewInMemoryStore creates a new in-memory session store with automatic expiration.
//
// By default, sessions expire after 24 hours of inactivity and cleanup runs every hour.
// These defaults can be customized using WithSessionTTL and WithCleanupInterval options.
func NewInMemoryStore(opts ...InMemoryStoreOption) *InMemoryStore {
	cfg := &inMemoryStoreConfig{
		ttl:             DefaultSessionTTL,
		cleanupInterval: DefaultCleanupInterval,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	return &InMemoryStore{
		cache: cache.New[string, *State](
			cache.MaxAge(cfg.ttl),
			cache.AutoCleanInterval(cfg.cleanupInterval),
		),
	}
}

// Load retrieves a session by ID.
//
// Returns ErrNotFound if the session does not exist or has expired.
func (s *InMemoryStore) Load(_ context.Context, sessionID string) (*State, error) {
	state, err, keyState := s.cache.TryGet(sessionID)
	if err != nil {
		return nil, err
	}

	if keyState.IsMiss() {
		return nil, ErrNotFound
	}

	// Return a clone to prevent external modifications
	return state.Clone(), nil
}

// Save persists a session and resets its expiration timer.
//
// If a session with the same ID already exists, it is completely replaced.
// The state is cloned before storing to prevent external modifications.
func (s *InMemoryStore) Save(_ context.Context, state *State) error {
	// Clone to prevent external modifications
	s.cache.Set(state.ID, state.Clone())
	return nil
}

// Delete removes a session.
//
// Returns nil if the session doesn't exist (idempotent operation).
func (s *InMemoryStore) Delete(_ context.Context, sessionID string) error {
	//nolint:dogsled // cache.Delete returns (value, existed, evicted); we don't need any of these
	_, _, _ = s.cache.Delete(sessionID)
	return nil
}
