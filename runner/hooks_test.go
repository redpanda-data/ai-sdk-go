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
	"github.com/redpanda-data/ai-sdk-go/agent/hooks"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// Test hook types.
type testBeforeHook struct {
	onBefore func(hooks.HookContext, llm.Message) error
}

func (h *testBeforeHook) OnBeforeInvocation(ctx hooks.HookContext, msg llm.Message) error {
	if h.onBefore != nil {
		return h.onBefore(ctx, msg)
	}

	return nil
}

type testAfterHook struct {
	onAfter func(hooks.HookContext, hooks.InvocationResult) error
}

func (h *testAfterHook) OnAfterInvocation(ctx hooks.HookContext, result hooks.InvocationResult) error {
	if h.onAfter != nil {
		return h.onAfter(ctx, result)
	}

	return nil
}

type emptyInvalidHook struct{}

// TestNew_InvalidHook verifies that runner construction fails with invalid hook.
func TestNew_InvalidHook(t *testing.T) {
	t.Parallel()

	ag := &mockAgent{name: "test"}
	store := session.NewInMemoryStore()

	r, err := runner.New(ag, store, runner.WithHook(emptyInvalidHook{}))

	require.Error(t, err)
	require.ErrorIs(t, err, agent.ErrInvalidHook)
	assert.Nil(t, r)
}

// TestBeforeInvocationHooks tests BeforeInvocation hook execution.
func TestBeforeInvocationHooks(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		hooks          []hooks.Hook
		expectRunError bool
		verify         func(t *testing.T, hookContexts []hooks.HookContext)
	}{
		{
			name: "single hook executes with correct context",
			hooks: []hooks.Hook{
				&testBeforeHook{
					onBefore: func(ctx hooks.HookContext, msg llm.Message) error {
						// Store context for verification
						metadata := ctx.Metadata()
						metadata["invocation_id"] = ctx.InvocationID()
						metadata["session_id"] = ctx.SessionID()
						metadata["turn"] = ctx.Turn()

						return nil
					},
				},
			},
			expectRunError: false,
			verify: func(t *testing.T, hookContexts []hooks.HookContext) {
				require.Len(t, hookContexts, 1)
				ctx := hookContexts[0]
				assert.NotEmpty(t, ctx.InvocationID())
				assert.NotEmpty(t, ctx.SessionID())
				assert.Equal(t, 0, ctx.Turn())
			},
		},
		{
			name: "multiple hooks execute in order",
			hooks: []hooks.Hook{
				&testBeforeHook{
					onBefore: func(ctx hooks.HookContext, msg llm.Message) error {
						ctx.SetMetadata("order", []string{"hook1"})
						return nil
					},
				},
				&testBeforeHook{
					onBefore: func(ctx hooks.HookContext, msg llm.Message) error {
						order := ctx.Metadata()["order"].([]string)
						order = append(order, "hook2")
						ctx.SetMetadata("order", order)

						return nil
					},
				},
			},
			expectRunError: false,
			verify: func(t *testing.T, hookContexts []hooks.HookContext) {
				require.Len(t, hookContexts, 2)
				// Both hooks should see the same context
				order := hookContexts[1].Metadata()["order"].([]string)
				assert.Equal(t, []string{"hook1", "hook2"}, order)
			},
		},
		{
			name: "first hook error stops execution",
			hooks: []hooks.Hook{
				&testBeforeHook{
					onBefore: func(ctx hooks.HookContext, msg llm.Message) error {
						return errors.New("hook failed")
					},
				},
				&testBeforeHook{
					onBefore: func(ctx hooks.HookContext, msg llm.Message) error {
						t.Error("second hook should not execute")
						return nil
					},
				},
			},
			expectRunError: true,
			verify:         func(t *testing.T, hookContexts []hooks.HookContext) {},
		},
		{
			name: "metadata passed between hooks",
			hooks: []hooks.Hook{
				&testBeforeHook{
					onBefore: func(ctx hooks.HookContext, msg llm.Message) error {
						ctx.SetMetadata("key1", "value1")
						ctx.SetMetadata("key2", 42)

						return nil
					},
				},
				&testBeforeHook{
					onBefore: func(ctx hooks.HookContext, msg llm.Message) error {
						// Second hook can read metadata set by first hook
						val1 := ctx.Metadata()["key1"]
						val2 := ctx.Metadata()["key2"]
						ctx.SetMetadata("verified", val1 == "value1" && val2 == 42)

						return nil
					},
				},
			},
			expectRunError: false,
			verify: func(t *testing.T, hookContexts []hooks.HookContext) {
				require.Len(t, hookContexts, 2)
				verified := hookContexts[1].Metadata()["verified"].(bool)
				assert.True(t, verified, "second hook should have verified first hook's metadata")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Capture hook contexts for verification
			var capturedContexts []hooks.HookContext

			// Create mock agent
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

			// Wrap hooks to capture contexts
			wrappedHooks := make([]hooks.Hook, len(tt.hooks))
			for i, h := range tt.hooks {
				beforeHook := h.(*testBeforeHook)
				originalOnBefore := beforeHook.onBefore
				wrappedHooks[i] = &testBeforeHook{
					onBefore: func(ctx hooks.HookContext, msg llm.Message) error {
						capturedContexts = append(capturedContexts, ctx)
						if originalOnBefore != nil {
							return originalOnBefore(ctx, msg)
						}

						return nil
					},
				}
			}

			// Create runner with hooks
			opts := make([]runner.Option, len(wrappedHooks))
			for i, h := range wrappedHooks {
				opts[i] = runner.WithHook(h)
			}

			store := session.NewInMemoryStore()
			r, err := runner.New(ag, store, opts...)
			require.NoError(t, err)

			// Execute
			ctx := context.Background()
			userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("test"))

			var runErr error

			for _, err := range r.Run(ctx, "", "test-session", userMsg) {
				if err != nil {
					runErr = err
					break
				}
			}

			if tt.expectRunError {
				assert.Error(t, runErr)
			} else {
				assert.NoError(t, runErr)
				tt.verify(t, capturedContexts)
			}
		})
	}
}

// TestAfterInvocationHooks tests AfterInvocation hook execution.
func TestAfterInvocationHooks(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		hooks  []hooks.Hook
		verify func(t *testing.T, results []hooks.InvocationResult)
	}{
		{
			name: "receives complete invocation result",
			hooks: []hooks.Hook{
				&testAfterHook{
					onAfter: func(ctx hooks.HookContext, result hooks.InvocationResult) error {
						// Store result in context for verification
						ctx.SetMetadata("finish_reason", result.FinishReason)
						ctx.SetMetadata("has_message", result.FinalMessage != nil)
						ctx.SetMetadata("total_tokens", result.TotalUsage.TotalTokens)

						return nil
					},
				},
			},
			verify: func(t *testing.T, results []hooks.InvocationResult) {
				require.Len(t, results, 1)
				result := results[0]
				assert.Equal(t, agent.FinishReasonStop, result.FinishReason)
				assert.NotNil(t, result.FinalMessage)
				assert.GreaterOrEqual(t, result.TotalUsage.TotalTokens, 0)
			},
		},
		{
			name: "multiple hooks execute in order",
			hooks: []hooks.Hook{
				&testAfterHook{
					onAfter: func(ctx hooks.HookContext, result hooks.InvocationResult) error {
						ctx.SetMetadata("hook_order", []string{"first"})
						return nil
					},
				},
				&testAfterHook{
					onAfter: func(ctx hooks.HookContext, result hooks.InvocationResult) error {
						order := ctx.Metadata()["hook_order"].([]string)
						order = append(order, "second")
						ctx.SetMetadata("hook_order", order)

						return nil
					},
				},
			},
			verify: func(t *testing.T, results []hooks.InvocationResult) {
				require.Len(t, results, 2)
				// Both hooks see the same result
				assert.Equal(t, results[0].FinishReason, results[1].FinishReason)
				assert.Equal(t, results[0].TotalUsage, results[1].TotalUsage)
			},
		},
		{
			name: "hook error does not affect result",
			hooks: []hooks.Hook{
				&testAfterHook{
					onAfter: func(ctx hooks.HookContext, result hooks.InvocationResult) error {
						// Even if hook returns error, result should be complete
						return errors.New("hook processing failed")
					},
				},
			},
			verify: func(t *testing.T, results []hooks.InvocationResult) {
				require.Len(t, results, 1)
				result := results[0]
				// Result should still be valid
				assert.Equal(t, agent.FinishReasonStop, result.FinishReason)
				assert.NotNil(t, result.FinalMessage)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Capture invocation results
			var capturedResults []hooks.InvocationResult

			// Create mock agent that generates a message
			ag := &mockAgent{
				name: "test-agent",
				runFunc: func(invCtx *agent.InvocationContext) iter.Seq2[agent.Event, error] {
					return func(yield func(agent.Event, error) bool) {
						envelope := agent.EventEnvelope{
							InvocationID: invCtx.InvocationID(),
							SessionID:    invCtx.Session().ID,
							Turn:         0,
							At:           time.Now().UTC(),
						}

						// Emit message event
						if !yield(agent.MessageEvent{
							Envelope: envelope,
							Response: llm.Response{
								Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Response")),
								Usage:   &llm.TokenUsage{TotalTokens: 100},
							},
						}, nil) {
							return
						}

						// Emit completion event
						yield(agent.InvocationEndEvent{
							Envelope:     envelope,
							FinishReason: agent.FinishReasonStop,
							Usage:        &llm.TokenUsage{TotalTokens: 100},
						}, nil)
					}
				},
			}

			// Wrap hooks to capture results
			wrappedHooks := make([]hooks.Hook, len(tt.hooks))
			for i, h := range tt.hooks {
				afterHook := h.(*testAfterHook)
				originalOnAfter := afterHook.onAfter
				wrappedHooks[i] = &testAfterHook{
					onAfter: func(ctx hooks.HookContext, result hooks.InvocationResult) error {
						capturedResults = append(capturedResults, result)
						if originalOnAfter != nil {
							return originalOnAfter(ctx, result)
						}

						return nil
					},
				}
			}

			// Create runner with hooks
			opts := make([]runner.Option, len(wrappedHooks))
			for i, h := range wrappedHooks {
				opts[i] = runner.WithHook(h)
			}

			store := session.NewInMemoryStore()
			r, err := runner.New(ag, store, opts...)
			require.NoError(t, err)

			// Execute
			ctx := context.Background()
			userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("test"))

			// Collect all events (ignore any hook errors for this test)
			for range r.Run(ctx, "", "test-session", userMsg) {
				// Just consume events
			}

			tt.verify(t, capturedResults)
		})
	}
}
