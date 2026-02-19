// Package retry provides a retry interceptor for LLM model calls.
//
// It implements [agent.ModelInterceptor] for use with the agent framework and
// also provides [WrapModel] for standalone use without the agent framework.
//
// The interceptor retries both synchronous (Generate) and streaming (GenerateEvents)
// calls when the error is classified as retryable (using [llm.IsRetryable]).
// For streaming, it emits [llm.StreamResetEvent] on retry so consumers can
// discard partial content before the stream restarts.
//
// # Usage with agent framework
//
//	retrier := retry.New(
//	    retry.WithMaxRetries(3),
//	    retry.WithLogger(slog.Default()),
//	)
//	agent, _ := llmagent.New("assistant", "You are helpful", model,
//	    llmagent.WithInterceptors(retrier),
//	)
//
// # Standalone usage
//
//	resilientModel := retry.WrapModel(baseModel,
//	    retry.WithMaxRetries(5),
//	)
//	resp, err := resilientModel.Generate(ctx, req)
package retry

import (
	"context"
	"errors"
	"iter"
	"log/slog"
	"math"
	"math/rand/v2"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Compile-time check that Interceptor implements ModelInterceptor.
var _ agent.ModelInterceptor = (*Interceptor)(nil)

// Interceptor implements [agent.ModelInterceptor] to add retry logic
// to model generation calls within the agent framework.
type Interceptor struct {
	cfg config
}

// New creates a new retry interceptor with the given options.
func New(opts ...Option) *Interceptor {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	return &Interceptor{cfg: cfg}
}

// InterceptModel wraps model generation with retry logic.
func (r *Interceptor) InterceptModel(
	_ context.Context,
	_ *agent.ModelCallInfo,
	next agent.ModelCallHandler,
) agent.ModelCallHandler {
	return &retryHandler{
		cfg:  r.cfg,
		next: next,
	}
}

// WrapModel wraps any [llm.Model] with retry logic, usable without the agent framework.
//
// The returned model retries both Generate and GenerateEvents calls on retryable errors.
func WrapModel(model llm.Model, opts ...Option) llm.Model {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	return &wrappedModel{
		ModelInfo: model,
		handler: &retryHandler{
			cfg:  cfg,
			next: model,
		},
	}
}

// wrappedModel combines the base model's identity with the retry handler's behavior.
type wrappedModel struct {
	llm.ModelInfo

	handler *retryHandler
}

func (m *wrappedModel) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	return m.handler.Generate(ctx, req)
}

func (m *wrappedModel) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return m.handler.GenerateEvents(ctx, req)
}

// retryHandler implements the retry logic for both Generate and GenerateEvents.
type retryHandler struct {
	cfg  config
	next agent.ModelCallHandler
}

// Generate retries synchronous generation on retryable errors.
func (h *retryHandler) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	var lastErr error

	for attempt := range h.cfg.maxRetries + 1 {
		if attempt > 0 {
			if err := h.backoff(ctx, attempt); err != nil {
				return nil, err
			}

			h.logRetry("Generate", attempt, lastErr)
		}

		resp, err := h.next.Generate(ctx, req)
		if err == nil {
			return resp, nil
		}

		lastErr = err

		if !h.shouldRetry(err, attempt+1) {
			return nil, err
		}
	}

	return nil, lastErr
}

// GenerateEvents retries streaming generation on retryable errors.
// On retry, it emits StreamResetEvent to signal consumers to discard partial content.
func (h *retryHandler) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		var lastErr error

		for attempt := range h.cfg.maxRetries + 1 {
			if attempt > 0 {
				if err := h.backoff(ctx, attempt); err != nil {
					yield(nil, err)
					return
				}

				h.logRetry("GenerateEvents", attempt, lastErr)

				// Signal consumers to discard partial content
				if !yield(llm.StreamResetEvent{
					Attempt: attempt,
					Reason:  lastErr.Error(),
				}, nil) {
					return
				}
			}

			failed := false

			for event, err := range h.next.GenerateEvents(ctx, req) {
				if err != nil {
					if h.shouldRetry(err, attempt+1) {
						lastErr = err
						failed = true

						break
					}

					// Not retryable — propagate immediately
					yield(nil, err)

					return
				}

				if !yield(event, nil) {
					return
				}
			}

			if !failed {
				return // success
			}
		}

		// Exhausted retries
		yield(nil, lastErr)
	}
}

// shouldRetry checks if an error should be retried.
// It never retries context cancellation or deadline exceeded.
func (h *retryHandler) shouldRetry(err error, attempt int) bool {
	if err == nil {
		return false
	}

	// Never retry context errors
	if isContextError(err) {
		return false
	}

	// Check attempt limit
	if attempt > h.cfg.maxRetries {
		return false
	}

	// Custom decision function
	if h.cfg.retryDecision != nil {
		return h.cfg.retryDecision(err, attempt)
	}

	// Default: use llm.IsRetryable
	return llm.IsRetryable(err)
}

// backoff waits with exponential backoff and jitter. Returns an error if the context
// is cancelled during the wait.
func (h *retryHandler) backoff(ctx context.Context, attempt int) error {
	delay := h.calculateDelay(attempt)

	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// calculateDelay computes exponential backoff with ±25% jitter.
func (h *retryHandler) calculateDelay(attempt int) time.Duration {
	// Exponential: initialDelay * 2^(attempt-1)
	delay := float64(h.cfg.initialDelay) * math.Pow(2, float64(attempt-1))

	// Cap at maxDelay
	if delay > float64(h.cfg.maxDelay) {
		delay = float64(h.cfg.maxDelay)
	}

	// Apply ±25% jitter
	jitter := delay * 0.25 * (2*rand.Float64() - 1) //nolint:gosec // jitter doesn't need crypto
	delay += jitter

	if delay < 0 {
		delay = 0
	}

	return time.Duration(delay)
}

// logRetry logs a retry attempt if a logger is configured.
func (h *retryHandler) logRetry(method string, attempt int, err error) {
	if h.cfg.logger != nil {
		h.cfg.logger.Warn("retrying model call",
			slog.String("method", method),
			slog.Int("attempt", attempt),
			slog.String("error", err.Error()),
		)
	}
}

// isContextError checks if the error is a context cancellation or deadline exceeded.
func isContextError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}
