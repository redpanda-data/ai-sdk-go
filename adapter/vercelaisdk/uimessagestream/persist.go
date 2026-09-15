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

package uimessagestream

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"sync"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// loadOrCreate loads the session for key, creating a fresh empty state when the
// store has none — a new chat's first message and a follow-up take the same
// path, mirroring runner's load-or-create semantics.
func loadOrCreate(ctx context.Context, store session.Store, key string) (*session.State, error) {
	sess, err := store.Load(ctx, key)
	if err == nil {
		return sess, nil
	}

	if errors.Is(err, session.ErrNotFound) {
		return &session.State{ID: key, Metadata: make(map[string]any)}, nil
	}

	return nil, err
}

// keyedMutex serializes work per session key so concurrent POSTs to the same
// chat cannot interleave load-modify-save and lose messages. Entries are
// refcounted and removed when idle, so the map does not grow with the number of
// chats ever seen. In-process only: multi-replica deployments must serialize
// per session externally (sticky routing or a store-level guard).
type keyedMutex struct {
	mu    sync.Mutex
	locks map[string]*keyLock
}

type keyLock struct {
	mu   sync.Mutex
	refs int
}

func newKeyedMutex() *keyedMutex {
	return &keyedMutex{locks: make(map[string]*keyLock)}
}

// lock acquires the key's mutex and returns the corresponding unlock.
func (km *keyedMutex) lock(key string) func() {
	km.mu.Lock()

	kl := km.locks[key]
	if kl == nil {
		kl = &keyLock{}
		km.locks[key] = kl
	}

	kl.refs++

	km.mu.Unlock()

	kl.mu.Lock()

	return func() {
		kl.mu.Unlock()
		km.mu.Lock()

		kl.refs--
		if kl.refs == 0 {
			delete(km.locks, key)
		}

		km.mu.Unlock()
	}
}

// lastGenuineUserIdx returns the index of the last message a human authored: a
// user-role message with at least one part that is not a tool response. Tool
// results are persisted as user-role messages carrying only ToolResponseParts
// (that is how providers expect them), and they must not count — regenerating
// from one would re-run from a mid-turn state.
func lastGenuineUserIdx(msgs []llm.Message) int {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != llm.RoleUser {
			continue
		}

		for _, p := range msgs[i].Content {
			if _, ok := p.(*llm.ToolResponsePart); !ok {
				return i
			}
		}
	}

	return -1
}

// truncateForRegenerate drops everything after the last genuine user message so
// the run regenerates the assistant's answer to it. Returns false when the
// session holds no user message to regenerate from.
func truncateForRegenerate(sess *session.State) bool {
	i := lastGenuineUserIdx(sess.Messages)
	if i < 0 {
		return false
	}

	sess.Messages = sess.Messages[:i+1]

	return true
}

// persistingAgent wraps an agent.Agent with session persistence: the session is
// saved after every completed assistant message and once more when the run
// ends, whatever the reason. This is the seam that keeps StreamAgent and the
// whole event-to-chunk layer stateless and unchanged.
//
// Not the runner: its append-one-message shape cannot express regenerate
// (truncate without append), and its deferred save uses the request context,
// which is already canceled in exactly the case that most needs persisting — a
// client disconnect. All saves here use context.WithoutCancel.
type persistingAgent struct {
	inner  agent.Agent
	store  session.Store
	sess   *session.State
	logger *slog.Logger
}

func (p *persistingAgent) Info() agent.Info { return p.inner.Info() }

func (p *persistingAgent) InputSchema() map[string]any { return p.inner.InputSchema() }

func (p *persistingAgent) Run(ctx context.Context, inv *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		saveCtx := context.WithoutCancel(ctx)

		// Final save: covers tool-response messages appended after the last
		// MessageEvent, and interrupted runs. Every path reaching this defer has
		// already terminated the stream, so a failure can only be logged.
		defer func() {
			if err := p.store.Save(saveCtx, p.sess); err != nil {
				p.logger.Error("failed to save session", "sessionId", p.sess.ID, "error", err)
			}
		}()

		for event, err := range p.inner.Run(ctx, inv) {
			if err != nil {
				if !yield(nil, err) {
					return
				}

				continue
			}

			// The agent appends the assistant message to the session before
			// yielding its MessageEvent; persist incrementally so a crash or
			// disconnect keeps every completed turn.
			if _, ok := event.(agent.MessageEvent); ok {
				if serr := p.store.Save(saveCtx, p.sess); serr != nil {
					yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionSave, serr))
					return
				}
			}

			if !yield(event, nil) {
				return
			}
		}
	}
}
