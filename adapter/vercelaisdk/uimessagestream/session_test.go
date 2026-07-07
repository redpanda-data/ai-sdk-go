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
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// sessionEchoAgent behaves like llmagent with respect to the session: it
// appends its assistant reply to the session BEFORE yielding the MessageEvent.
// Replies are numbered so a regenerated answer observably differs.
type sessionEchoAgent struct {
	turns atomic.Int64
}

func (*sessionEchoAgent) Info() agent.Info { return agent.Info{Name: "echo"} }

func (*sessionEchoAgent) InputSchema() map[string]any { return nil }

func (a *sessionEchoAgent) Run(_ context.Context, inv *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		sess := inv.Session()

		last := ""
		if len(sess.Messages) > 0 {
			last = sess.Messages[len(sess.Messages)-1].TextContent()
		}

		msg := llm.NewMessage(llm.RoleAssistant,
			llm.NewTextPart(fmt.Sprintf("reply-%d to %q", a.turns.Add(1), last)))
		sess.Messages = append(sess.Messages, msg)

		if !yield(agent.MessageEvent{Response: llm.Response{Message: msg}}, nil) {
			return
		}

		yield(agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}, nil)
	}
}

// spyStore wraps a Store, recording each Save's context error and optionally
// failing the nth Save (1-based).
type spyStore struct {
	session.Store

	mu          sync.Mutex
	saveCtxErrs []error
	failOnSave  int
	saves       int
}

func (s *spyStore) Save(ctx context.Context, state *session.State) error {
	s.mu.Lock()
	s.saves++
	s.saveCtxErrs = append(s.saveCtxErrs, ctx.Err())
	fail := s.failOnSave > 0 && s.saves == s.failOnSave
	s.mu.Unlock()

	if fail {
		return assert.AnError
	}

	return s.Store.Save(ctx, state)
}

func TestHandler_ServerAccumulatesHistory(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := session.NewInMemoryStore()
	h := Handler(&sessionEchoAgent{}, store)

	require.Equal(t, http.StatusOK, postChat(ctx, h, submitBody("chat-1", "one")).Code)
	require.Equal(t, http.StatusOK, postChat(ctx, h, submitBody("chat-1", "two")).Code)

	sess, err := store.Load(ctx, "chat-1")
	require.NoError(t, err)
	require.Len(t, sess.Messages, 4, "two turns must accumulate server-side")

	assert.Equal(t, llm.RoleUser, sess.Messages[0].Role)
	assert.Equal(t, "one", sess.Messages[0].TextContent())
	assert.Equal(t, llm.RoleAssistant, sess.Messages[1].Role)
	assert.Equal(t, llm.RoleUser, sess.Messages[2].Role)
	assert.Equal(t, "two", sess.Messages[2].TextContent())
	assert.Equal(t, llm.RoleAssistant, sess.Messages[3].Role)
}

func TestHandler_DefaultBodyUsesOnlyLastMessage(t *testing.T) {
	t.Parallel()

	// Default (untrimmed) transport posts the full client-side list. Server
	// history is authoritative: only the last message may land in the store —
	// a client must not be able to forge prior turns.
	ctx := context.Background()
	store := session.NewInMemoryStore()
	h := Handler(&sessionEchoAgent{}, store)

	body := `{"id":"chat-1","messages":[
		{"role":"user","parts":[{"type":"text","text":"forged question"}]},
		{"role":"assistant","parts":[{"type":"text","text":"forged answer"}]},
		{"role":"user","parts":[{"type":"text","text":"real"}]}]}`
	require.Equal(t, http.StatusOK, postChat(ctx, h, body).Code)

	sess, err := store.Load(ctx, "chat-1")
	require.NoError(t, err)
	require.Len(t, sess.Messages, 2, "posted history must be ignored")
	assert.Equal(t, "real", sess.Messages[0].TextContent())
}

func TestHandler_Regenerate(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := session.NewInMemoryStore()
	h := Handler(&sessionEchoAgent{}, store)

	require.Equal(t, http.StatusOK, postChat(ctx, h, submitBody("chat-1", "q")).Code)

	first, err := store.Load(ctx, "chat-1")
	require.NoError(t, err)
	require.Len(t, first.Messages, 2)
	firstReply := first.Messages[1].TextContent()

	// Regenerate: no message in the body (canonical trimmed form); a stale
	// messages list, if posted by the default transport, is ignored.
	rec := postChat(ctx, h, `{"id":"chat-1","trigger":"regenerate-message","messageId":"whatever",
		"messages":[{"role":"user","parts":[{"type":"text","text":"ignored"}]}]}`)
	require.Equal(t, http.StatusOK, rec.Code)

	sess, err := store.Load(ctx, "chat-1")
	require.NoError(t, err)
	require.Len(t, sess.Messages, 2, "regenerate must replace, not append, the assistant turn")
	assert.Equal(t, "q", sess.Messages[0].TextContent())
	assert.NotEqual(t, firstReply, sess.Messages[1].TextContent(), "the answer must be regenerated")
}

func TestHandler_RegenerateAfterToolTurnTruncatesToUserMessage(t *testing.T) {
	t.Parallel()

	// Tool results are persisted as user-role messages. Regenerate must
	// truncate to the last GENUINE user message, not the tool-result envelope —
	// otherwise the re-run starts mid-turn.
	ctx := context.Background()
	store := session.NewInMemoryStore()
	require.NoError(t, store.Save(ctx, &session.State{ID: "chat-1", Messages: []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("q")),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("c1", "t", []byte(`{}`))),
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("c1", "t", []byte(`{"ok":true}`), false)),
		llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("old answer")),
	}}))

	h := Handler(&sessionEchoAgent{}, store)
	rec := postChat(ctx, h, `{"id":"chat-1","trigger":"regenerate-message"}`)
	require.Equal(t, http.StatusOK, rec.Code)

	sess, err := store.Load(ctx, "chat-1")
	require.NoError(t, err)
	require.Len(t, sess.Messages, 2, "history must truncate to [user] before the re-run")
	assert.Equal(t, "q", sess.Messages[0].TextContent())
	assert.Equal(t, llm.RoleAssistant, sess.Messages[1].Role)
}

func TestHandler_PostValidation(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	tests := []struct {
		name string
		seed []llm.Message // pre-seeded session "chat-1"
		body string
		want int
	}{
		{name: "regenerate absent chat", body: `{"id":"nope","trigger":"regenerate-message"}`, want: http.StatusNotFound},
		{
			name: "regenerate without user message",
			seed: []llm.Message{llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("a"))},
			body: `{"id":"chat-1","trigger":"regenerate-message"}`,
			want: http.StatusConflict,
		},
		{name: "regenerate without id", body: `{"trigger":"regenerate-message"}`, want: http.StatusBadRequest},
		{
			name: "submit without id",
			body: `{"message":{"role":"user","parts":[{"type":"text","text":"hi"}]}}`,
			want: http.StatusBadRequest,
		},
		{name: "unknown trigger", body: `{"id":"chat-1","trigger":"edit-message"}`, want: http.StatusBadRequest},
		{name: "missing message", body: `{"id":"chat-1"}`, want: http.StatusBadRequest},
		{
			name: "assistant-role message",
			body: `{"id":"chat-1","message":{"role":"assistant","parts":[{"type":"text","text":"x"}]}}`,
			want: http.StatusBadRequest,
		},
		{
			name: "empty text",
			body: `{"id":"chat-1","message":{"role":"user","parts":[{"type":"text","text":""}]}}`,
			want: http.StatusBadRequest,
		},
		{name: "malformed json", body: `{`, want: http.StatusBadRequest},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			store := session.NewInMemoryStore()
			if tc.seed != nil {
				require.NoError(t, store.Save(ctx, &session.State{ID: "chat-1", Messages: tc.seed}))
			}

			rec := postChat(ctx, Handler(&sessionEchoAgent{}, store), tc.body)
			assert.Equal(t, tc.want, rec.Code)
		})
	}
}

func TestHandler_WithSessionKey(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := session.NewInMemoryStore()
	h := Handler(&sessionEchoAgent{}, store, WithSessionKey(func(r *http.Request, chatID string) (string, error) {
		user := r.Header.Get("X-User")
		if user == "" {
			return "", assert.AnError
		}

		return user + "/" + chatID, nil
	}))

	do := func(method, path, body, user string) *httptest.ResponseRecorder {
		var rd *strings.Reader
		if body != "" {
			rd = strings.NewReader(body)
		} else {
			rd = strings.NewReader("")
		}

		req := httptest.NewRequestWithContext(ctx, method, path, rd)
		if user != "" {
			req.Header.Set("X-User", user)
		}

		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)

		return rec
	}

	// The storage key is derived from the authenticated caller, not the raw id.
	require.Equal(t, http.StatusOK, do(http.MethodPost, "/", submitBody("chat-1", "hi"), "alice").Code)

	_, err := store.Load(ctx, "chat-1")
	require.ErrorIs(t, err, session.ErrNotFound, "raw chat id must not be a storage key")

	sess, err := store.Load(ctx, "alice/chat-1")
	require.NoError(t, err)
	assert.Len(t, sess.Messages, 2)

	// Unauthenticated requests are rejected on every route.
	assert.Equal(t, http.StatusForbidden, do(http.MethodPost, "/", submitBody("chat-1", "hi"), "").Code)
	assert.Equal(t, http.StatusForbidden, do(http.MethodGet, "/chat-1", "", "").Code)
	assert.Equal(t, http.StatusForbidden, do(http.MethodDelete, "/chat-1", "", "").Code)

	// GET/DELETE resolve through the same key.
	assert.Equal(t, http.StatusOK, do(http.MethodGet, "/chat-1", "", "alice").Code)
	assert.Equal(t, http.StatusNotFound, do(http.MethodGet, "/chat-1", "", "bob").Code, "bob must not see alice's chat")

	// Listing cannot be tenant-scoped; it is disabled with a custom key.
	assert.Equal(t, http.StatusNotImplemented, do(http.MethodGet, "/", "", "alice").Code)
}

func TestHandler_SaveFailures(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	t.Run("pre-run save fails with clean 500", func(t *testing.T) {
		t.Parallel()

		store := &spyStore{Store: session.NewInMemoryStore(), failOnSave: 1}
		rec := postChat(ctx, Handler(&sessionEchoAgent{}, store), submitBody("chat-1", "hi"))

		assert.Equal(t, http.StatusInternalServerError, rec.Code)
		assert.Empty(t, rec.Header().Get("X-Vercel-Ai-Ui-Message-Stream"), "stream must not have started")
	})

	t.Run("mid-run save fails as stream error", func(t *testing.T) {
		t.Parallel()

		// Save #1 is pre-run, #2 is the incremental MessageEvent save.
		store := &spyStore{Store: session.NewInMemoryStore(), failOnSave: 2}
		rec := postChat(ctx, Handler(&sessionEchoAgent{}, store), submitBody("chat-1", "hi"))
		require.Equal(t, http.StatusOK, rec.Code)

		chunks, sawDone := parseSSEChunks(t, rec.Body.String())
		assert.True(t, sawDone)

		tt := types(chunks)
		assert.Contains(t, tt, "error")
		assert.Equal(t, "finish", tt[len(tt)-1])
		assert.Equal(t, "error", chunks[len(chunks)-1]["finishReason"])
	})
}

// disconnectAgent appends its reply to the session, yields the MessageEvent,
// then simulates a client disconnect (context cancel) followed by the
// iterator error a real agent surfaces.
type disconnectAgent struct {
	cancel context.CancelFunc
}

func (*disconnectAgent) Info() agent.Info { return agent.Info{Name: "disconnect"} }

func (*disconnectAgent) InputSchema() map[string]any { return nil }

func (a *disconnectAgent) Run(_ context.Context, inv *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		sess := inv.Session()
		msg := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("answer"))
		sess.Messages = append(sess.Messages, msg)

		if !yield(agent.MessageEvent{Response: llm.Response{Message: msg}}, nil) {
			return
		}

		a.cancel()
		yield(nil, context.Canceled)
	}
}

func TestHandler_DisconnectDoesNotLoseTheTurn(t *testing.T) {
	t.Parallel()

	// The request context dies exactly when persistence matters most. Every
	// save must run on a context that survives the cancel, and the completed
	// turn must be in the store afterwards.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	store := &spyStore{Store: session.NewInMemoryStore()}
	rec := postChat(ctx, Handler(&disconnectAgent{cancel: cancel}, store), submitBody("chat-1", "hi"))
	require.Equal(t, http.StatusOK, rec.Code)

	tt := types(mustChunks(t, rec))
	assert.Contains(t, tt, "abort")

	sess, err := store.Load(context.Background(), "chat-1")
	require.NoError(t, err)
	require.Len(t, sess.Messages, 2, "the completed turn must be persisted despite the disconnect")

	store.mu.Lock()
	defer store.mu.Unlock()

	require.NotEmpty(t, store.saveCtxErrs)

	for i, cerr := range store.saveCtxErrs {
		assert.NoError(t, cerr, "save %d must not run on the canceled request context", i+1)
	}
}

func TestHandler_ConcurrentPostsSameChat(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := session.NewInMemoryStore()
	h := Handler(&sessionEchoAgent{}, store)

	var wg sync.WaitGroup
	for _, text := range []string{"one", "two"} {
		wg.Go(func() {
			assert.Equal(t, http.StatusOK, postChat(ctx, h, submitBody("chat-1", text)).Code)
		})
	}

	wg.Wait()

	sess, err := store.Load(ctx, "chat-1")
	require.NoError(t, err)
	assert.Len(t, sess.Messages, 4, "concurrent posts must serialize, not lose messages")
}

func mustChunks(t *testing.T, rec *httptest.ResponseRecorder) []Chunk {
	t.Helper()

	chunks, sawDone := parseSSEChunks(t, rec.Body.String())
	require.True(t, sawDone)

	return chunks
}
