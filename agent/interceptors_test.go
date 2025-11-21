package agent_test

import (
	"context"
	"errors"
	"iter"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Test helpers

// orderRecorder provides thread-safe recording of execution order.
type orderRecorder struct {
	mu    sync.Mutex
	calls []string
}

func newOrderRecorder() *orderRecorder {
	return &orderRecorder{calls: make([]string, 0)}
}

func (r *orderRecorder) record(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.calls = append(r.calls, name)
}

func (r *orderRecorder) get() []string {
	r.mu.Lock()
	defer r.mu.Unlock()

	return append([]string(nil), r.calls...)
}

// newTestModelRequest creates a standard test model request.
func newTestModelRequest() *llm.Request {
	return &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("test"))},
	}
}

// newTestToolRequest creates a standard test tool request.
func newTestToolRequest() *llm.ToolRequest {
	return &llm.ToolRequest{
		ID:   "test-tool",
		Name: "test",
	}
}

// newCanceledContext creates a pre-canceled context for testing.
func newCanceledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	return ctx
}

// newOrderingModelInterceptor creates a model interceptor that records execution order.
func newOrderingModelInterceptor(name string, recorder *orderRecorder) *testModelInterceptor {
	return &testModelInterceptor{
		name: name,
		//nolint:revive // ctx and req are used in nested closures
		intercept: func(ctx context.Context, req *llm.Request, next agent.ModelCallHandler) agent.ModelCallHandler {
			return &testModelCallHandler{
				generateFunc: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
					recorder.record(name + "-before")

					resp, err := next.Generate(ctx, req)

					recorder.record(name + "-after")

					return resp, err
				},
				generateEventsFunc: func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
					return next.GenerateEvents(ctx, req)
				},
			}
		},
	}
}

// newOrderingToolInterceptor creates a tool interceptor that records execution order.
func newOrderingToolInterceptor(name string, recorder *orderRecorder) *testToolInterceptor {
	return &testToolInterceptor{
		intercept: func(ctx context.Context, req *llm.ToolRequest, next agent.ToolExecutionNext) (*llm.ToolResponse, error) {
			recorder.record(name + "-before")

			resp, err := next(ctx, req)

			recorder.record(name + "-after")

			return resp, err
		},
	}
}

// TestModelInterceptor_Ordering verifies that model interceptors are applied
// in reverse order (first registered interceptor = outermost wrapper).
func TestModelInterceptor_Ordering(t *testing.T) {
	t.Parallel()

	recorder := newOrderRecorder()

	// Create three interceptors that record order
	interceptor1 := newOrderingModelInterceptor("interceptor1", recorder)
	interceptor2 := newOrderingModelInterceptor("interceptor2", recorder)
	interceptor3 := newOrderingModelInterceptor("interceptor3", recorder)

	baseModel := &testModel{
		name: "base",
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			recorder.record("base")

			return &llm.Response{
				Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("response")),
			}, nil
		},
	}

	// Apply interceptors in order: interceptor1, interceptor2, interceptor3
	// Expected chain: interceptor1 -> interceptor2 -> interceptor3 -> base
	interceptors := []agent.Interceptor{interceptor1, interceptor2, interceptor3}

	ctx := context.Background()
	req := newTestModelRequest()

	interceptedModel := agent.ApplyModelInterceptors(ctx, req, baseModel, interceptors)

	// Execute
	_, err := interceptedModel.Generate(ctx, req)
	require.NoError(t, err)

	// Verify: first registered interceptor (interceptor1) should be outermost
	expected := []string{
		"interceptor1-before",
		"interceptor2-before",
		"interceptor3-before",
		"base",
		"interceptor3-after",
		"interceptor2-after",
		"interceptor1-after",
	}

	assert.Equal(t, expected, recorder.get(), "Interceptor execution order should be: first registered = outermost wrapper")
}

// TestToolInterceptor_Ordering verifies that tool interceptors are applied
// in reverse order (first registered interceptor = outermost wrapper).
func TestToolInterceptor_Ordering(t *testing.T) {
	t.Parallel()

	recorder := newOrderRecorder()

	// Create three interceptors that record order
	interceptor1 := newOrderingToolInterceptor("interceptor1", recorder)
	interceptor2 := newOrderingToolInterceptor("interceptor2", recorder)
	interceptor3 := newOrderingToolInterceptor("interceptor3", recorder)

	baseExecutor := func(_ context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
		recorder.record("base")

		return &llm.ToolResponse{
			ID:     req.ID,
			Name:   req.Name,
			Result: []byte(`"result"`),
		}, nil
	}

	// Apply interceptors in order: interceptor1, interceptor2, interceptor3
	interceptors := []agent.Interceptor{interceptor1, interceptor2, interceptor3}

	ctx := context.Background()
	executor := agent.ApplyToolInterceptors(ctx, interceptors, baseExecutor)

	// Execute
	req := newTestToolRequest()
	_, err := executor(ctx, req)
	require.NoError(t, err)

	// Verify: first registered interceptor (interceptor1) should be outermost
	expected := []string{
		"interceptor1-before",
		"interceptor2-before",
		"interceptor3-before",
		"base",
		"interceptor3-after",
		"interceptor2-after",
		"interceptor1-after",
	}

	assert.Equal(t, expected, recorder.get(), "Interceptor execution order should be: first registered = outermost wrapper")
}

// TestModelInterceptor_ShortCircuit verifies that a model interceptor can
// short-circuit execution by not calling next (e.g., cache hit).
func TestModelInterceptor_ShortCircuit(t *testing.T) {
	t.Parallel()

	baseCalled := atomic.Bool{}
	cacheHitResponse := &llm.Response{
		Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("cached response")),
	}

	// Cache interceptor that never calls next
	cacheInterceptor := &testModelInterceptor{
		name: "cache",
		intercept: func(_ context.Context, _ *llm.Request, _ agent.ModelCallHandler) agent.ModelCallHandler {
			return &testModelCallHandler{
				generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
					// Return cached response without calling next
					return cacheHitResponse, nil
				},
				generateEventsFunc: func(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
					// For streaming, also return without calling next
					return func(yield func(llm.Event, error) bool) {
						yield(llm.ContentPartEvent{
							Index: 0,
							Part:  llm.NewTextPart("cached response"),
						}, nil)
					}
				},
			}
		},
	}

	baseModel := &testModel{
		name: "base",
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			baseCalled.Store(true)

			return &llm.Response{
				Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("real response")),
			}, nil
		},
	}

	ctx := context.Background()
	req := newTestModelRequest()

	interceptedModel := agent.ApplyModelInterceptors(ctx, req, baseModel, []agent.Interceptor{cacheInterceptor})

	// Execute
	resp, err := interceptedModel.Generate(ctx, req)
	require.NoError(t, err)

	// Verify: base model should NOT have been called
	assert.False(t, baseCalled.Load(), "Base model should not be called when interceptor short-circuits")
	assert.Equal(t, "cached response", resp.Message.TextContent(), "Should get cached response")
}

// TestToolInterceptor_ShortCircuit verifies that a tool interceptor can
// short-circuit execution by not calling next (e.g., deny execution).
func TestToolInterceptor_ShortCircuit(t *testing.T) {
	t.Parallel()

	baseCalled := atomic.Bool{}

	// Authorization interceptor that denies execution
	authInterceptor := &testToolInterceptor{
		intercept: func(_ context.Context, req *llm.ToolRequest, _ agent.ToolExecutionNext) (*llm.ToolResponse, error) {
			// Return error without calling next
			return &llm.ToolResponse{
				ID:    req.ID,
				Name:  req.Name,
				Error: "execution denied by policy",
			}, nil
		},
	}

	baseExecutor := func(_ context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
		baseCalled.Store(true)

		return &llm.ToolResponse{
			ID:     req.ID,
			Name:   req.Name,
			Result: []byte(`"real result"`),
		}, nil
	}

	ctx := context.Background()
	executor := agent.ApplyToolInterceptors(ctx, []agent.Interceptor{authInterceptor}, baseExecutor)

	// Execute
	req := newTestToolRequest()
	req.Name = "dangerous_tool"
	resp, err := executor(ctx, req)
	require.NoError(t, err)

	// Verify: base executor should NOT have been called
	assert.False(t, baseCalled.Load(), "Base executor should not be called when interceptor short-circuits")
	assert.NotEmpty(t, resp.Error, "Response should have error")
	assert.Contains(t, resp.Error, "execution denied", "Should get denial message")
}

// TestModelInterceptor_ErrorPropagation verifies error handling in the interceptor chain.
func TestModelInterceptor_ErrorPropagation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                string
		interceptorError    error
		baseError           error
		expectInterceptCall bool
		expectBaseCall      bool
		wantErr             string
	}{
		{
			name:                "interceptor returns error without calling next",
			interceptorError:    errors.New("interceptor validation failed"),
			baseError:           nil,
			expectInterceptCall: true,
			expectBaseCall:      false,
			wantErr:             "interceptor validation failed",
		},
		{
			name:                "base returns error",
			interceptorError:    nil,
			baseError:           errors.New("model API error"),
			expectInterceptCall: true,
			expectBaseCall:      true,
			wantErr:             "model API error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			interceptorCalled := atomic.Bool{}
			baseCalled := atomic.Bool{}

			testInterceptor := &testModelInterceptor{
				name: "test",
				//nolint:revive // ctx and req are used in nested closures
				intercept: func(ctx context.Context, req *llm.Request, next agent.ModelCallHandler) agent.ModelCallHandler {
					return &testModelCallHandler{
						generateFunc: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
							interceptorCalled.Store(true)

							if tt.interceptorError != nil {
								return nil, tt.interceptorError
							}

							return next.Generate(ctx, req)
						},
						generateEventsFunc: func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
							return next.GenerateEvents(ctx, req)
						},
					}
				},
			}

			baseModel := &testModel{
				name: "base",
				generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
					baseCalled.Store(true)

					if tt.baseError != nil {
						return nil, tt.baseError
					}

					return &llm.Response{
						Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("response")),
					}, nil
				},
			}

			ctx := context.Background()
			req := newTestModelRequest()

			interceptedModel := agent.ApplyModelInterceptors(ctx, req, baseModel, []agent.Interceptor{testInterceptor})

			// Execute
			_, err := interceptedModel.Generate(ctx, req)

			// Verify error
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
			} else {
				require.NoError(t, err)
			}

			// Verify call expectations
			assert.Equal(t, tt.expectInterceptCall, interceptorCalled.Load(), "Interceptor call expectation")
			assert.Equal(t, tt.expectBaseCall, baseCalled.Load(), "Base call expectation")
		})
	}
}

// TestToolInterceptor_ErrorPropagation verifies error handling in tool execution.
func TestToolInterceptor_ErrorPropagation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                string
		interceptorError    error
		baseError           error
		expectInterceptCall bool
		expectBaseCall      bool
		wantErr             string
	}{
		{
			name:                "interceptor returns error without calling next",
			interceptorError:    errors.New("interceptor validation failed"),
			baseError:           nil,
			expectInterceptCall: true,
			expectBaseCall:      false,
			wantErr:             "interceptor validation failed",
		},
		{
			name:                "base returns error",
			interceptorError:    nil,
			baseError:           errors.New("tool execution failed"),
			expectInterceptCall: true,
			expectBaseCall:      true,
			wantErr:             "tool execution failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			interceptorCalled := atomic.Bool{}
			baseCalled := atomic.Bool{}

			testInterceptor := &testToolInterceptor{
				intercept: func(ctx context.Context, req *llm.ToolRequest, next agent.ToolExecutionNext) (*llm.ToolResponse, error) {
					interceptorCalled.Store(true)

					if tt.interceptorError != nil {
						return nil, tt.interceptorError
					}

					return next(ctx, req)
				},
			}

			baseExecutor := func(_ context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
				baseCalled.Store(true)

				if tt.baseError != nil {
					return nil, tt.baseError
				}

				return &llm.ToolResponse{
					ID:     req.ID,
					Name:   req.Name,
					Result: []byte(`"result"`),
				}, nil
			}

			ctx := context.Background()
			executor := agent.ApplyToolInterceptors(ctx, []agent.Interceptor{testInterceptor}, baseExecutor)

			// Execute
			req := newTestToolRequest()
			_, err := executor(ctx, req)

			// Verify error
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
			} else {
				require.NoError(t, err)
			}

			// Verify call expectations
			assert.Equal(t, tt.expectInterceptCall, interceptorCalled.Load(), "Interceptor call expectation")
			assert.Equal(t, tt.expectBaseCall, baseCalled.Load(), "Base call expectation")
		})
	}
}

// TestModelInterceptor_ContextPropagation verifies that context is properly
// propagated through the interceptor chain and respects cancellation.
func TestModelInterceptor_ContextPropagation(t *testing.T) {
	t.Parallel()

	interceptorSeenContext := atomic.Bool{}
	baseSeenContext := atomic.Bool{}

	testInterceptor := &testModelInterceptor{
		name: "context-checker",
		//nolint:revive // ctx and req are used in nested closures
		intercept: func(ctx context.Context, req *llm.Request, next agent.ModelCallHandler) agent.ModelCallHandler {
			return &testModelCallHandler{
				generateFunc: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
					if ctx.Err() != nil {
						interceptorSeenContext.Store(true)
						return nil, ctx.Err()
					}

					return next.Generate(ctx, req)
				},
				generateEventsFunc: func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
					return next.GenerateEvents(ctx, req)
				},
			}
		},
	}

	baseModel := &testModel{
		name: "base",
		generateFunc: func(ctx context.Context, _ *llm.Request) (*llm.Response, error) {
			if ctx.Err() != nil {
				baseSeenContext.Store(true)
				return nil, ctx.Err()
			}

			return &llm.Response{
				Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("response")),
			}, nil
		},
	}

	// Create canceled context
	ctx := newCanceledContext()
	req := newTestModelRequest()

	interceptedModel := agent.ApplyModelInterceptors(ctx, req, baseModel, []agent.Interceptor{testInterceptor})

	// Execute with canceled context
	_, err := interceptedModel.Generate(ctx, req)

	// Verify cancellation error
	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)

	// Verify interceptor short-circuits on cancellation
	assert.True(t, interceptorSeenContext.Load(), "Interceptor should see canceled context")
	assert.False(t, baseSeenContext.Load(), "Base should not be called when interceptor short-circuits on cancellation")
}

// TestToolInterceptor_ContextPropagation verifies context propagation for tool execution.
func TestToolInterceptor_ContextPropagation(t *testing.T) {
	t.Parallel()

	interceptorSeenContext := atomic.Bool{}
	baseSeenContext := atomic.Bool{}

	testInterceptor := &testToolInterceptor{
		intercept: func(ctx context.Context, req *llm.ToolRequest, next agent.ToolExecutionNext) (*llm.ToolResponse, error) {
			if ctx.Err() != nil {
				interceptorSeenContext.Store(true)
				return nil, ctx.Err()
			}

			return next(ctx, req)
		},
	}

	baseExecutor := func(ctx context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
		if ctx.Err() != nil {
			baseSeenContext.Store(true)
			return nil, ctx.Err()
		}

		return &llm.ToolResponse{
			ID:     req.ID,
			Name:   req.Name,
			Result: []byte(`"result"`),
		}, nil
	}

	// Create canceled context
	ctx := newCanceledContext()

	executor := agent.ApplyToolInterceptors(ctx, []agent.Interceptor{testInterceptor}, baseExecutor)

	// Execute with canceled context
	req := newTestToolRequest()
	_, err := executor(ctx, req)

	// Verify cancellation error
	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)

	// Verify interceptor short-circuits on cancellation
	assert.True(t, interceptorSeenContext.Load(), "Interceptor should see canceled context")
	assert.False(t, baseSeenContext.Load(), "Base should not be called when interceptor short-circuits on cancellation")
}

// TestImplementsAnyInterceptor verifies interceptor validation.
func TestImplementsAnyInterceptor(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		interceptor agent.Interceptor
		wantOK      bool
	}{
		{
			name:        "valid turn interceptor",
			interceptor: &testTurnInterceptor{},
			wantOK:      true,
		},
		{
			name:        "valid model interceptor",
			interceptor: &testModelInterceptor{},
			wantOK:      true,
		},
		{
			name:        "valid tool interceptor",
			interceptor: &testToolInterceptor{},
			wantOK:      true,
		},
		{
			name:        "multi-interface interceptor",
			interceptor: &multiInterceptor{},
			wantOK:      true,
		},
		{
			name:        "invalid interceptor (implements nothing)",
			interceptor: &struct{}{},
			wantOK:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			ok := agent.ImplementsAnyInterceptor(tt.interceptor)
			assert.Equal(t, tt.wantOK, ok)
		})
	}
}

// TestTurnInterceptor_Ordering verifies that turn interceptors are applied
// in reverse order (first registered interceptor = outermost wrapper).
func TestTurnInterceptor_Ordering(t *testing.T) {
	t.Parallel()

	recorder := newOrderRecorder()

	// Create three interceptors that record order
	interceptor1 := &testTurnInterceptor{
		intercept: func(ctx context.Context, next agent.TurnNext) (agent.FinishReason, error) {
			recorder.record("interceptor1-before")

			reason, err := next(ctx)

			recorder.record("interceptor1-after")

			return reason, err
		},
	}
	interceptor2 := &testTurnInterceptor{
		intercept: func(ctx context.Context, next agent.TurnNext) (agent.FinishReason, error) {
			recorder.record("interceptor2-before")

			reason, err := next(ctx)

			recorder.record("interceptor2-after")

			return reason, err
		},
	}
	interceptor3 := &testTurnInterceptor{
		intercept: func(ctx context.Context, next agent.TurnNext) (agent.FinishReason, error) {
			recorder.record("interceptor3-before")

			reason, err := next(ctx)

			recorder.record("interceptor3-after")

			return reason, err
		},
	}

	baseTurn := func(_ context.Context) (agent.FinishReason, error) {
		recorder.record("base")
		return agent.FinishReasonStop, nil
	}

	// Apply interceptors in order: interceptor1, interceptor2, interceptor3
	// Expected chain: interceptor1 -> interceptor2 -> interceptor3 -> base
	interceptors := []agent.Interceptor{interceptor1, interceptor2, interceptor3}

	ctx := context.Background()
	turnFunc := agent.ApplyTurnInterceptors(ctx, interceptors, baseTurn)

	// Execute
	reason, err := turnFunc(ctx)
	require.NoError(t, err)
	assert.Equal(t, agent.FinishReasonStop, reason)

	// Verify: first registered interceptor (interceptor1) should be outermost
	expected := []string{
		"interceptor1-before",
		"interceptor2-before",
		"interceptor3-before",
		"base",
		"interceptor3-after",
		"interceptor2-after",
		"interceptor1-after",
	}

	assert.Equal(t, expected, recorder.get(), "Interceptor execution order should be: first registered = outermost wrapper")
}

// TestTurnInterceptor_EarlyStopping verifies that a turn interceptor can
// stop execution early by returning a finish reason without calling next.
func TestTurnInterceptor_EarlyStopping(t *testing.T) {
	t.Parallel()

	baseCalled := atomic.Bool{}

	// Interceptor that stops early
	earlyStopInterceptor := &testTurnInterceptor{
		intercept: func(_ context.Context, _ agent.TurnNext) (agent.FinishReason, error) {
			// Return early without calling next
			return agent.FinishReasonMaxTurns, nil
		},
	}

	baseTurn := func(_ context.Context) (agent.FinishReason, error) {
		baseCalled.Store(true)
		return agent.FinishReasonStop, nil
	}

	ctx := context.Background()
	turnFunc := agent.ApplyTurnInterceptors(ctx, []agent.Interceptor{earlyStopInterceptor}, baseTurn)

	// Execute
	reason, err := turnFunc(ctx)
	require.NoError(t, err)

	// Verify: should get the early stop reason
	assert.Equal(t, agent.FinishReasonMaxTurns, reason)

	// Verify: base turn should NOT have been called
	assert.False(t, baseCalled.Load(), "Base turn should not be called when interceptor stops early")
}

// TestEmptyInterceptors verifies that empty interceptors list is a no-op.
func TestEmptyInterceptors(t *testing.T) {
	t.Parallel()

	t.Run("model interceptor with empty interceptors", func(t *testing.T) {
		t.Parallel()

		baseModel := &testModel{
			name: "base",
			generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
				return &llm.Response{
					Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("response")),
				}, nil
			},
		}

		ctx := context.Background()
		req := newTestModelRequest()

		// Apply with empty interceptors list
		interceptedModel := agent.ApplyModelInterceptors(ctx, req, baseModel, []agent.Interceptor{})

		// Should return base model unchanged
		assert.Equal(t, baseModel, interceptedModel, "Empty interceptors should return base model unchanged")
	})

	t.Run("tool interceptor with empty interceptors", func(t *testing.T) {
		t.Parallel()

		called := atomic.Bool{}
		baseExecutor := func(_ context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
			called.Store(true)

			return &llm.ToolResponse{
				ID:     req.ID,
				Name:   req.Name,
				Result: []byte(`"result"`),
			}, nil
		}

		ctx := context.Background()

		// Apply with empty interceptors list
		executor := agent.ApplyToolInterceptors(ctx, []agent.Interceptor{}, baseExecutor)

		// Execute
		req := newTestToolRequest()
		_, err := executor(ctx, req)
		require.NoError(t, err)

		assert.True(t, called.Load(), "Base executor should be called with empty interceptors")
	})

	t.Run("turn interceptor with empty interceptors", func(t *testing.T) {
		t.Parallel()

		called := atomic.Bool{}
		baseTurn := func(_ context.Context) (agent.FinishReason, error) {
			called.Store(true)
			return agent.FinishReasonStop, nil
		}

		ctx := context.Background()

		// Apply with empty interceptors list
		turnFunc := agent.ApplyTurnInterceptors(ctx, []agent.Interceptor{}, baseTurn)

		// Execute
		reason, err := turnFunc(ctx)
		require.NoError(t, err)
		assert.Equal(t, agent.FinishReasonStop, reason)

		assert.True(t, called.Load(), "Base turn should be called with empty interceptors")
	})
}

// testTurnInterceptor is a test implementation of TurnInterceptor.
type testTurnInterceptor struct {
	intercept func(ctx context.Context, next agent.TurnNext) (agent.FinishReason, error)
}

func (i *testTurnInterceptor) InterceptTurn(ctx context.Context, next agent.TurnNext) (agent.FinishReason, error) {
	if i.intercept != nil {
		return i.intercept(ctx, next)
	}

	return next(ctx)
}

// testModelInterceptor is a test implementation of ModelInterceptor.
type testModelInterceptor struct {
	name      string
	intercept func(ctx context.Context, req *llm.Request, next agent.ModelCallHandler) agent.ModelCallHandler
}

func (i *testModelInterceptor) InterceptModel(ctx context.Context, req *llm.Request, next agent.ModelCallHandler) agent.ModelCallHandler {
	if i.intercept != nil {
		return i.intercept(ctx, req, next)
	}

	return next
}

// testToolInterceptor is a test implementation of ToolInterceptor.
type testToolInterceptor struct {
	intercept func(ctx context.Context, req *llm.ToolRequest, next agent.ToolExecutionNext) (*llm.ToolResponse, error)
}

func (i *testToolInterceptor) InterceptToolExecution(ctx context.Context, req *llm.ToolRequest, next agent.ToolExecutionNext) (*llm.ToolResponse, error) {
	if i.intercept != nil {
		return i.intercept(ctx, req, next)
	}

	return next(ctx, req)
}

// multiInterceptor implements all three interceptor interfaces.
type multiInterceptor struct{}

func (i *multiInterceptor) InterceptTurn(ctx context.Context, next agent.TurnNext) (agent.FinishReason, error) {
	return next(ctx)
}

func (i *multiInterceptor) InterceptModel(_ context.Context, _ *llm.Request, next agent.ModelCallHandler) agent.ModelCallHandler {
	return next
}

func (i *multiInterceptor) InterceptToolExecution(ctx context.Context, req *llm.ToolRequest, next agent.ToolExecutionNext) (*llm.ToolResponse, error) {
	return next(ctx, req)
}

// testModelCallHandler is a test implementation of ModelCallHandler.
type testModelCallHandler struct {
	generateFunc       func(ctx context.Context, req *llm.Request) (*llm.Response, error)
	generateEventsFunc func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error]
}

func (h *testModelCallHandler) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if h.generateFunc != nil {
		return h.generateFunc(ctx, req)
	}

	return &llm.Response{
		Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("default response")),
	}, nil
}

func (h *testModelCallHandler) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	if h.generateEventsFunc != nil {
		return h.generateEventsFunc(ctx, req)
	}

	return func(yield func(llm.Event, error) bool) {
		yield(llm.ContentPartEvent{
			Index: 0,
			Part:  llm.NewTextPart("default response"),
		}, nil)
	}
}

// testModel is a test implementation of llm.Model.
type testModel struct {
	name               string
	generateFunc       func(ctx context.Context, req *llm.Request) (*llm.Response, error)
	generateEventsFunc func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error]
}

func (m *testModel) Name() string {
	return m.name
}

func (m *testModel) Capabilities() llm.ModelCapabilities {
	return llm.ModelCapabilities{
		Streaming: true,
		Tools:     true,
	}
}

func (m *testModel) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, req)
	}

	return &llm.Response{
		Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("default response")),
	}, nil
}

func (m *testModel) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	if m.generateEventsFunc != nil {
		return m.generateEventsFunc(ctx, req)
	}

	return func(yield func(llm.Event, error) bool) {
		yield(llm.ContentPartEvent{
			Index: 0,
			Part:  llm.NewTextPart("default response"),
		}, nil)
	}
}
