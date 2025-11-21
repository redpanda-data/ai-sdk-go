package runner_test

import (
	"context"
	"errors"
	"iter"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// TestNew_Validation tests runner configuration validation.
func TestNew_Validation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		agent        agent.Agent
		sessionStore session.Store
		wantErr      error
	}{
		{
			name:         "valid configuration",
			agent:        &mockAgent{name: "test"},
			sessionStore: session.NewInMemoryStore(),
			wantErr:      nil,
		},
		{
			name:         "missing agent",
			agent:        nil,
			sessionStore: session.NewInMemoryStore(),
			wantErr:      agent.ErrNoAgent,
		},
		{
			name:         "missing session store",
			agent:        &mockAgent{name: "test"},
			sessionStore: nil,
			wantErr:      agent.ErrNoSessionStore,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			r, err := runner.New(tt.agent, tt.sessionStore)

			if tt.wantErr != nil {
				require.Error(t, err)
				require.ErrorIs(t, err, tt.wantErr)
				assert.Nil(t, r)
			} else {
				require.NoError(t, err)
				require.NotNil(t, r)
			}
		})
	}
}

// TestRun_NewSession verifies new session creation.
func TestRun_NewSession(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()
	ag := &mockAgent{
		name: "test-agent",
		runFunc: func(invCtx *agent.InvocationContext) iter.Seq2[agent.Event, error] {
			return func(yield func(agent.Event, error) bool) {
				// Verify session is empty initially
				sess := invCtx.Session()
				if len(sess.Messages) != 1 { // Only user message
					yield(nil, errors.New("expected 1 message"))
					return
				}

				// Emit completion event
				yield(agent.InvocationEndEvent{
					Envelope: agent.EventEnvelope{
						InvocationID: invCtx.InvocationID(),
						SessionID:    sess.ID,
						Turn:         0,
						At:           time.Now().UTC(),
					},
					FinishReason: agent.FinishReasonStop,
					Usage:        &llm.TokenUsage{TotalTokens: 100},
				}, nil)
			}
		},
	}

	r, err := runner.New(ag, store)
	require.NoError(t, err)

	// Execute with new session
	ctx := context.Background()
	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))

	events := collectEvents(t, r.Run(ctx, "", "new-session", userMsg))

	// Assert: Should have completion event
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonStop, endEvent.FinishReason)

	// Verify session was saved
	savedSess, err := store.Load(ctx, "new-session")
	require.NoError(t, err)
	assert.Equal(t, "new-session", savedSess.ID)
	assert.Len(t, savedSess.Messages, 1) // User message
}

// TestRun_ExistingSession verifies existing session loading.
func TestRun_ExistingSession(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()

	// Pre-populate session with existing messages
	ctx := context.Background()
	existingMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Previous message"))
	err := store.Save(ctx, &session.State{
		ID:       "existing-session",
		Messages: []llm.Message{existingMsg},
	})
	require.NoError(t, err)

	ag := &mockAgent{
		name: "test-agent",
		runFunc: func(invCtx *agent.InvocationContext) iter.Seq2[agent.Event, error] {
			return func(yield func(agent.Event, error) bool) {
				sess := invCtx.Session()

				// Verify session has both old and new messages
				if len(sess.Messages) != 2 {
					yield(nil, errors.New("expected 2 messages"))
					return
				}

				// Add assistant response to session
				sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Response")))

				yield(agent.InvocationEndEvent{
					Envelope: agent.EventEnvelope{
						InvocationID: invCtx.InvocationID(),
						SessionID:    sess.ID,
						Turn:         0,
						At:           time.Now().UTC(),
					},
					FinishReason: agent.FinishReasonStop,
				}, nil)
			}
		},
	}

	r, err := runner.New(ag, store)
	require.NoError(t, err)

	// Execute with existing session
	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("New message"))
	events := collectEvents(t, r.Run(ctx, "", "existing-session", userMsg))

	// Assert: Should complete
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)

	// Verify session was updated
	savedSess, err := store.Load(ctx, "existing-session")
	require.NoError(t, err)
	assert.Len(t, savedSess.Messages, 3) // Previous + new user + assistant
}

// TestRun_MessageAccumulation verifies messages accumulate correctly.
func TestRun_MessageAccumulation(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()

	ag := &mockAgent{
		name: "test-agent",
		runFunc: func(invCtx *agent.InvocationContext) iter.Seq2[agent.Event, error] {
			return func(yield func(agent.Event, error) bool) {
				sess := invCtx.Session()

				// Add assistant response
				sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Response")))

				yield(agent.InvocationEndEvent{
					Envelope: agent.EventEnvelope{
						InvocationID: invCtx.InvocationID(),
						SessionID:    sess.ID,
						Turn:         0,
						At:           time.Now().UTC(),
					},
					FinishReason: agent.FinishReasonStop,
				}, nil)
			}
		},
	}

	r, err := runner.New(ag, store)
	require.NoError(t, err)

	ctx := context.Background()

	// Execute first invocation
	userMsg1 := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Message 1"))
	collectEvents(t, r.Run(ctx, "", "test-session", userMsg1))

	// Execute second invocation
	userMsg2 := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Message 2"))
	collectEvents(t, r.Run(ctx, "", "test-session", userMsg2))

	// Execute third invocation
	userMsg3 := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Message 3"))
	collectEvents(t, r.Run(ctx, "", "test-session", userMsg3))

	// Verify all messages accumulated
	savedSess, err := store.Load(ctx, "test-session")
	require.NoError(t, err)
	assert.Len(t, savedSess.Messages, 6) // 3 user + 3 assistant

	// Verify order is preserved
	assert.Equal(t, llm.RoleUser, savedSess.Messages[0].Role)
	assert.Equal(t, "Message 1", savedSess.Messages[0].TextContent())
	assert.Equal(t, llm.RoleAssistant, savedSess.Messages[1].Role)
	assert.Equal(t, llm.RoleUser, savedSess.Messages[2].Role)
	assert.Equal(t, "Message 2", savedSess.Messages[2].TextContent())
	assert.Equal(t, llm.RoleAssistant, savedSess.Messages[3].Role)
}

// TestRun_EventForwarding verifies events are forwarded correctly.
func TestRun_EventForwarding(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()

	// Agent that emits multiple event types
	ag := &mockAgent{
		name: "test-agent",
		runFunc: func(invCtx *agent.InvocationContext) iter.Seq2[agent.Event, error] {
			return func(yield func(agent.Event, error) bool) {
				sess := invCtx.Session()
				envelope := agent.EventEnvelope{
					InvocationID: invCtx.InvocationID(),
					SessionID:    sess.ID,
					Turn:         0,
					At:           time.Now().UTC(),
				}

				// Emit status event
				if !yield(agent.StatusEvent{
					Envelope: envelope,
					Stage:    agent.StatusStageTurnStarted,
					Details:  "turn started",
				}, nil) {
					return
				}

				// Emit message event
				if !yield(agent.MessageEvent{
					Envelope: envelope,
					Response: llm.Response{
						Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Response")),
					},
				}, nil) {
					return
				}

				// Emit completion event
				yield(agent.InvocationEndEvent{
					Envelope:     envelope,
					FinishReason: agent.FinishReasonStop,
				}, nil)
			}
		},
	}

	r, err := runner.New(ag, store)
	require.NoError(t, err)

	ctx := context.Background()
	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))

	// Collect events
	events := collectEvents(t, r.Run(ctx, "", "test-session", userMsg))

	// Verify all events were forwarded
	require.Len(t, events, 3) // Status, Message, InvocationEnd

	assert.IsType(t, agent.StatusEvent{}, events[0])
	assert.IsType(t, agent.MessageEvent{}, events[1])
	assert.IsType(t, agent.InvocationEndEvent{}, events[2])
}

// TestRun_SessionLoadError verifies session load errors are handled.
func TestRun_SessionLoadError(t *testing.T) {
	t.Parallel()

	// Store that returns error on Load
	store := &mockSessionStore{
		loadFunc: func(_ context.Context, _ string) (*session.State, error) {
			return nil, errors.New("database connection failed")
		},
	}

	ag := &mockAgent{name: "test-agent"}

	r, err := runner.New(ag, store)
	require.NoError(t, err)

	ctx := context.Background()
	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))

	// Execute - should get error
	var gotError error

	for _, err := range r.Run(ctx, "", "test-session", userMsg) {
		if err != nil {
			gotError = err
			break
		}
	}

	require.Error(t, gotError)
	assert.ErrorIs(t, gotError, agent.ErrSessionLoad)
}

// TestRun_SessionSaveError verifies session save errors are handled.
func TestRun_SessionSaveError(t *testing.T) {
	t.Parallel()

	// Store that returns error on Save
	store := &mockSessionStore{
		loadFunc: func(_ context.Context, _ string) (*session.State, error) {
			return nil, session.ErrNotFound
		},
		saveFunc: func(_ context.Context, _ *session.State) error {
			return errors.New("database write failed")
		},
	}

	ag := &mockAgent{
		name: "test-agent",
		runFunc: func(invCtx *agent.InvocationContext) iter.Seq2[agent.Event, error] {
			return func(yield func(agent.Event, error) bool) {
				yield(agent.InvocationEndEvent{
					Envelope: agent.EventEnvelope{
						InvocationID: invCtx.InvocationID(),
						SessionID:    invCtx.Session().ID,
						Turn:         0,
						At:           time.Now().UTC(),
					},
					FinishReason: agent.FinishReasonStop,
				}, nil)
			}
		},
	}

	r, err := runner.New(ag, store)
	require.NoError(t, err)

	ctx := context.Background()
	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))

	// Collect events - should get save error after completion
	var gotError error

	for _, err := range r.Run(ctx, "", "test-session", userMsg) {
		if err != nil {
			gotError = err
			// Continue to check if error comes after completion
		}
	}

	require.Error(t, gotError)
	assert.ErrorIs(t, gotError, agent.ErrSessionSave)
}

// TestRun_ContextCancellation verifies context cancellation.
func TestRun_ContextCancellation(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()

	ag := &mockAgent{
		name: "test-agent",
		runFunc: func(invCtx *agent.InvocationContext) iter.Seq2[agent.Event, error] {
			return func(yield func(agent.Event, error) bool) {
				// Check if context is already canceled
				if invCtx.Err() != nil {
					yield(agent.InvocationEndEvent{
						Envelope: agent.EventEnvelope{
							InvocationID: invCtx.InvocationID(),
							SessionID:    invCtx.Session().ID,
							Turn:         0,
							At:           time.Now().UTC(),
						},
						FinishReason: agent.FinishReasonInterrupted,
					}, nil)

					return
				}

				// Normal completion
				yield(agent.InvocationEndEvent{
					Envelope: agent.EventEnvelope{
						InvocationID: invCtx.InvocationID(),
						SessionID:    invCtx.Session().ID,
						Turn:         0,
						At:           time.Now().UTC(),
					},
					FinishReason: agent.FinishReasonStop,
				}, nil)
			}
		},
	}

	r, err := runner.New(ag, store)
	require.NoError(t, err)

	// Create canceled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))

	// Execute with canceled context
	events := collectEvents(t, r.Run(ctx, "", "test-session", userMsg))

	// Assert: Should get canceled finish reason
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonInterrupted, endEvent.FinishReason)
}

// Helper functions

// collectEvents collects all events from an iterator.
// Fails immediately on any error - use collectEventsWithErrors for tests that expect errors.
func collectEvents(t *testing.T, iter func(func(agent.Event, error) bool)) []agent.Event {
	t.Helper()

	var events []agent.Event //nolint:prealloc // size unknown, depends on iterator

	for evt, err := range iter {
		require.NoError(t, err, "unexpected error in event stream")

		events = append(events, evt)
	}

	return events
}

// findInvocationEndEvent finds the InvocationEndEvent in events.
func findInvocationEndEvent(events []agent.Event) *agent.InvocationEndEvent {
	for i := len(events) - 1; i >= 0; i-- {
		if endEvt, ok := events[i].(agent.InvocationEndEvent); ok {
			return &endEvt
		}
	}

	return nil
}

// mockAgent is a test implementation of agent.Agent.
type mockAgent struct {
	name        string
	description string
	runFunc     func(*agent.InvocationContext) iter.Seq2[agent.Event, error]
}

func (m *mockAgent) Name() string {
	return m.name
}

func (m *mockAgent) Description() string {
	return m.description
}

func (m *mockAgent) Run(invCtx *agent.InvocationContext) iter.Seq2[agent.Event, error] {
	if m.runFunc != nil {
		return m.runFunc(invCtx)
	}

	// Default implementation
	return func(yield func(agent.Event, error) bool) {
		yield(agent.InvocationEndEvent{
			Envelope: agent.EventEnvelope{
				InvocationID: invCtx.InvocationID(),
				SessionID:    invCtx.Session().ID,
				Turn:         0,
				At:           time.Now().UTC(),
			},
			FinishReason: agent.FinishReasonStop,
		}, nil)
	}
}

func (m *mockAgent) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"message": map[string]any{
				"type": "string",
			},
		},
	}
}

// mockSessionStore is a test implementation of session.Store.
type mockSessionStore struct {
	loadFunc   func(context.Context, string) (*session.State, error)
	saveFunc   func(context.Context, *session.State) error
	deleteFunc func(context.Context, string) error
}

func (m *mockSessionStore) Load(ctx context.Context, sessionID string) (*session.State, error) {
	if m.loadFunc != nil {
		return m.loadFunc(ctx, sessionID)
	}

	return nil, session.ErrNotFound
}

func (m *mockSessionStore) Save(ctx context.Context, state *session.State) error {
	if m.saveFunc != nil {
		return m.saveFunc(ctx, state)
	}

	return nil
}

func (m *mockSessionStore) Delete(ctx context.Context, sessionID string) error {
	if m.deleteFunc != nil {
		return m.deleteFunc(ctx, sessionID)
	}

	return nil
}
