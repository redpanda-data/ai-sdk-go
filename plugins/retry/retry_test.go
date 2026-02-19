package retry

import (
	"context"
	"errors"
	"iter"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// --- Mock model ---

type mockModel struct {
	generateFunc       func(ctx context.Context, req *llm.Request) (*llm.Response, error)
	generateEventsFunc func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error]
}

func (m *mockModel) Name() string                        { return "mock" }
func (m *mockModel) Provider() string                    { return "test" }
func (m *mockModel) Capabilities() llm.ModelCapabilities { return llm.ModelCapabilities{} }
func (m *mockModel) Constraints() llm.ModelConstraints   { return llm.ModelConstraints{} }

func (m *mockModel) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	return m.generateFunc(ctx, req)
}

func (m *mockModel) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return m.generateEventsFunc(ctx, req)
}

// --- Helpers ---

func retryableErr() error {
	return &llm.ProviderError{
		Base:      llm.ErrServerError,
		Code:      "server_error",
		Message:   "Internal server error",
		Retryable: true,
	}
}

func nonRetryableErr() error {
	return &llm.ProviderError{
		Base:      llm.ErrInvalidInput,
		Code:      "invalid_input",
		Message:   "Bad request",
		Retryable: false,
	}
}

func successResponse() *llm.Response {
	return &llm.Response{
		ID: "resp-1",
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: []*llm.Part{llm.NewTextPart("Hello")},
		},
		FinishReason: llm.FinishReasonStop,
	}
}

// collectEvents drains an iterator into slices.
func collectEvents(seq iter.Seq2[llm.Event, error]) ([]llm.Event, error) {
	events := make([]llm.Event, 0)

	for event, err := range seq {
		if err != nil {
			return events, err
		}

		events = append(events, event)
	}

	return events, nil
}

// --- Generate tests ---

func TestGenerate_Success(t *testing.T) {
	t.Parallel()

	mock := &mockModel{
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			return successResponse(), nil
		},
	}

	model := WrapModel(mock, WithMaxRetries(3), WithInitialDelay(time.Millisecond))
	resp, err := model.Generate(context.Background(), &llm.Request{})

	require.NoError(t, err)
	assert.Equal(t, "resp-1", resp.ID)
}

func TestGenerate_SuccessAfterRetries(t *testing.T) {
	t.Parallel()

	callCount := 0
	mock := &mockModel{
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			callCount++
			if callCount <= 2 {
				return nil, retryableErr()
			}

			return successResponse(), nil
		},
	}

	model := WrapModel(mock, WithMaxRetries(3), WithInitialDelay(time.Millisecond))
	resp, err := model.Generate(context.Background(), &llm.Request{})

	require.NoError(t, err)
	assert.Equal(t, "resp-1", resp.ID)
	assert.Equal(t, 3, callCount)
}

func TestGenerate_Exhausted(t *testing.T) {
	t.Parallel()

	callCount := 0
	mock := &mockModel{
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			callCount++
			return nil, retryableErr()
		},
	}

	model := WrapModel(mock, WithMaxRetries(2), WithInitialDelay(time.Millisecond))
	_, err := model.Generate(context.Background(), &llm.Request{})

	require.Error(t, err)
	assert.True(t, llm.IsRetryable(err))
	assert.Equal(t, 3, callCount) // 1 initial + 2 retries
}

func TestGenerate_NonRetryableSkipsRetry(t *testing.T) {
	t.Parallel()

	callCount := 0
	mock := &mockModel{
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			callCount++
			return nil, nonRetryableErr()
		},
	}

	model := WrapModel(mock, WithMaxRetries(3), WithInitialDelay(time.Millisecond))
	_, err := model.Generate(context.Background(), &llm.Request{})

	require.Error(t, err)
	assert.Equal(t, 1, callCount)
}

func TestGenerate_ContextCancel(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	callCount := 0

	mock := &mockModel{
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			callCount++
			if callCount == 1 {
				cancel()
			}

			return nil, retryableErr()
		},
	}

	model := WrapModel(mock, WithMaxRetries(3), WithInitialDelay(time.Millisecond))
	_, err := model.Generate(ctx, &llm.Request{})

	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)
}

func TestGenerate_RateLimit(t *testing.T) {
	t.Parallel()

	callCount := 0
	mock := &mockModel{
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			callCount++
			if callCount == 1 {
				return nil, &llm.ProviderError{
					Base:      llm.ErrRateLimitExceeded,
					Code:      "rate_limit",
					Message:   "Too many requests",
					Retryable: true,
				}
			}

			return successResponse(), nil
		},
	}

	model := WrapModel(mock, WithMaxRetries(3), WithInitialDelay(time.Millisecond))
	resp, err := model.Generate(context.Background(), &llm.Request{})

	require.NoError(t, err)
	assert.Equal(t, "resp-1", resp.ID)
	assert.Equal(t, 2, callCount)
}

// --- GenerateEvents tests ---

func TestGenerateEvents_Success(t *testing.T) {
	t.Parallel()

	mock := &mockModel{
		generateEventsFunc: func(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
			return func(yield func(llm.Event, error) bool) {
				yield(llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("Hello")}, nil)
				yield(llm.StreamEndEvent{Response: successResponse()}, nil)
			}
		},
	}

	model := WrapModel(mock, WithMaxRetries(3), WithInitialDelay(time.Millisecond))
	events, err := collectEvents(model.GenerateEvents(context.Background(), &llm.Request{}))

	require.NoError(t, err)
	require.Len(t, events, 2)
	assert.IsType(t, llm.ContentPartEvent{}, events[0])
	assert.IsType(t, llm.StreamEndEvent{}, events[1])
}

func TestGenerateEvents_MidStreamErrorEmitsResetThenRetries(t *testing.T) {
	t.Parallel()

	callCount := 0
	mock := &mockModel{
		generateEventsFunc: func(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
			callCount++

			return func(yield func(llm.Event, error) bool) {
				if callCount <= 1 {
					// First attempt: emit some content then fail
					yield(llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("partial")}, nil)
					yield(nil, retryableErr())

					return
				}

				// Second attempt: succeed
				yield(llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("complete")}, nil)
				yield(llm.StreamEndEvent{Response: successResponse()}, nil)
			}
		},
	}

	model := WrapModel(mock, WithMaxRetries(3), WithInitialDelay(time.Millisecond))
	events, err := collectEvents(model.GenerateEvents(context.Background(), &llm.Request{}))

	require.NoError(t, err)
	assert.Equal(t, 2, callCount)

	// Events: partial content, StreamResetEvent, complete content, StreamEndEvent
	require.Len(t, events, 4)
	assert.IsType(t, llm.ContentPartEvent{}, events[0])
	resetEvt, ok := events[1].(llm.StreamResetEvent)
	require.True(t, ok)
	assert.Equal(t, 1, resetEvt.Attempt)
	assert.Contains(t, resetEvt.Reason, "Internal server error")
	assert.IsType(t, llm.ContentPartEvent{}, events[2])
	assert.IsType(t, llm.StreamEndEvent{}, events[3])
}

func TestGenerateEvents_ExhaustedRetries(t *testing.T) {
	t.Parallel()

	callCount := 0
	mock := &mockModel{
		generateEventsFunc: func(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
			callCount++

			return func(yield func(llm.Event, error) bool) {
				yield(nil, retryableErr())
			}
		},
	}

	model := WrapModel(mock, WithMaxRetries(2), WithInitialDelay(time.Millisecond))
	events, err := collectEvents(model.GenerateEvents(context.Background(), &llm.Request{}))

	require.Error(t, err)
	assert.Equal(t, 3, callCount) // 1 initial + 2 retries
	// Should have 2 StreamResetEvents
	resetCount := 0

	for _, e := range events {
		if _, ok := e.(llm.StreamResetEvent); ok {
			resetCount++
		}
	}

	assert.Equal(t, 2, resetCount)
}

func TestGenerateEvents_NonRetryablePropagated(t *testing.T) {
	t.Parallel()

	callCount := 0
	mock := &mockModel{
		generateEventsFunc: func(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
			callCount++

			return func(yield func(llm.Event, error) bool) {
				yield(nil, nonRetryableErr())
			}
		},
	}

	model := WrapModel(mock, WithMaxRetries(3), WithInitialDelay(time.Millisecond))
	_, err := collectEvents(model.GenerateEvents(context.Background(), &llm.Request{}))

	require.Error(t, err)
	assert.Equal(t, 1, callCount)
}

// --- WrapModel tests ---

func TestWrapModel_PreservesIdentity(t *testing.T) {
	t.Parallel()

	mock := &mockModel{
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			return successResponse(), nil
		},
	}

	model := WrapModel(mock, WithMaxRetries(3))

	assert.Equal(t, "mock", model.Name())
	assert.Equal(t, "test", model.Provider())
}

// --- Config tests ---

func TestCustomDecision(t *testing.T) {
	t.Parallel()

	callCount := 0
	mock := &mockModel{
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			callCount++
			return nil, errors.New("custom error")
		},
	}

	// Custom decision: always retry
	model := WrapModel(mock,
		WithMaxRetries(2),
		WithInitialDelay(time.Millisecond),
		WithRetryDecision(func(_ error, _ int) bool { return true }),
	)
	_, err := model.Generate(context.Background(), &llm.Request{})

	require.Error(t, err)
	assert.Equal(t, 3, callCount)
}

func TestLogger(t *testing.T) {
	t.Parallel()

	callCount := 0
	mock := &mockModel{
		generateFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			callCount++
			if callCount == 1 {
				return nil, retryableErr()
			}

			return successResponse(), nil
		},
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))
	model := WrapModel(mock,
		WithMaxRetries(3),
		WithInitialDelay(time.Millisecond),
		WithLogger(logger),
	)

	resp, err := model.Generate(context.Background(), &llm.Request{})
	require.NoError(t, err)
	assert.Equal(t, "resp-1", resp.ID)
}

func TestDefaults(t *testing.T) {
	t.Parallel()

	cfg := defaultConfig()
	assert.Equal(t, 3, cfg.maxRetries)
	assert.Equal(t, 200*time.Millisecond, cfg.initialDelay)
	assert.Equal(t, 30*time.Second, cfg.maxDelay)
	assert.Nil(t, cfg.retryDecision)
	assert.Nil(t, cfg.logger)
}

func TestCalculateDelay(t *testing.T) {
	t.Parallel()

	h := &retryHandler{cfg: config{
		initialDelay: 100 * time.Millisecond,
		maxDelay:     5 * time.Second,
	}}

	// Test that delays grow exponentially (within jitter bounds)
	for attempt := 1; attempt <= 5; attempt++ {
		delay := h.calculateDelay(attempt)
		expectedBase := min(100*time.Millisecond*time.Duration(1<<(attempt-1)), 5*time.Second)

		// Allow ±25% jitter
		minDelay := time.Duration(float64(expectedBase) * 0.75)

		maxDelay := time.Duration(float64(expectedBase) * 1.25)

		assert.GreaterOrEqual(t, delay, minDelay, "attempt %d: delay %v < min %v", attempt, delay, minDelay)
		assert.LessOrEqual(t, delay, maxDelay, "attempt %d: delay %v > max %v", attempt, delay, maxDelay)
	}
}

// --- Interceptor tests ---

func TestInterceptor_ImplementsInterface(t *testing.T) {
	t.Parallel()

	interceptor := New()
	assert.Implements(t, (*agent.ModelInterceptor)(nil), interceptor)
}
