package fakellm

import (
	"context"
	"fmt"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Error helper functions for testing error handling and retry logic.
// These methods inject errors from the llm package to simulate various failure scenarios.
// The fakellm package does not define its own error types - it uses errors from the
// llm package that real providers would return.

// RateLimitOnce configures the model to return an API call error on the first call,
// then succeed on subsequent calls. Use this to simulate rate limiting behavior.
//
// Example:
//
//	model := fakellm.NewFakeModel().
//	    RateLimitOnce().
//	    When(fakellm.Any()).
//	    ThenRespondText("Success!")
//
//	// First call fails
//	_, err := model.Generate(ctx, req)
//	// errors.Is(err, llm.ErrAPICall) == true
//
//	// Second call succeeds
//	resp, err := model.Generate(ctx, req)
//	// err == nil
func (m *FakeModel) RateLimitOnce() *FakeModel {
	return m.When(Any()).Named("rate-limit-once").Times(1).ThenError(llm.ErrAPICall)
}

// RateLimitNTimes configures the model to return API call errors for the first n calls.
// Use this to simulate repeated rate limiting that requires multiple retries.
//
// Example:
//
//	model := fakellm.NewFakeModel().
//	    RateLimitNTimes(3).
//	    When(fakellm.Any()).
//	    ThenRespondText("Success after 3 retries")
func (m *FakeModel) RateLimitNTimes(n int) *FakeModel {
	return m.When(Any()).Named(fmt.Sprintf("rate-limit-%d-times", n)).Times(n).ThenError(llm.ErrAPICall)
}

// TimeoutOnce configures the model to return a timeout error on the first call.
//
// Example:
//
//	model := fakellm.NewFakeModel().
//	    TimeoutOnce().
//	    When(fakellm.Any()).
//	    ThenRespondText("Success!")
func (m *FakeModel) TimeoutOnce() *FakeModel {
	return m.When(Any()).Named("timeout-once").Times(1).ThenError(context.DeadlineExceeded)
}

// APIErrorOnce configures the model to return an API error on the first call.
//
// Example:
//
//	model := fakellm.NewFakeModel().
//	    APIErrorOnce().
//	    When(fakellm.Any()).
//	    ThenRespondText("Success!")
func (m *FakeModel) APIErrorOnce() *FakeModel {
	return m.When(Any()).Named("api-error-once").Times(1).ThenError(llm.ErrAPICall)
}

// ErrorAfterNCalls configures the model to return an error after n successful calls.
// This is useful for testing error handling in long-running operations.
//
// Example:
//
//	model := fakellm.NewFakeModel().
//	    When(fakellm.Any()).
//	    ThenRespondText("Success").
//	    ErrorAfterNCalls(5, errors.New("unexpected error"))
//
//	// First 5 calls succeed, 6th call fails
func (m *FakeModel) ErrorAfterNCalls(n int, err error) *FakeModel {
	return m.When(CallNumber(n + 1)).Named(fmt.Sprintf("error-after-%d", n)).ThenError(err)
}

// MidStreamError configures streaming to fail after emitting some chunks.
// This simulates connection drops or service interruptions during streaming.
//
// Example:
//
//	model := fakellm.NewFakeModel().
//	    When(fakellm.Any()).
//	    ThenStreamText("This will be interrupted...", fakellm.StreamConfig{
//	        ErrorAfterChunks: 3,
//	        MidStreamError:   llm.ErrStreamClosed,
//	    })
func (m *FakeModel) MidStreamError(afterChunks int, err error) *FakeModel {
	return m.When(Any()).
		Named(fmt.Sprintf("midstream-error-after-%d", afterChunks)).
		ThenStreamText("This will be interrupted", StreamConfig{
			ErrorAfterChunks: afterChunks,
			MidStreamError:   err,
		})
}

// SimulateTimeout simulates a timeout by using a slow response and a short context deadline.
// This is useful for testing context cancellation handling.
//
// Example:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
//	defer cancel()
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithLatency(fakellm.LatencyProfile{
//	        Base: 200 * time.Millisecond,
//	    }),
//	)
//
//	_, err := model.Generate(ctx, req)
//	// err == context.DeadlineExceeded
func SimulateTimeout(baseLatency time.Duration) Option {
	return WithLatency(LatencyProfile{
		Base: baseLatency,
	})
}

// ErrorPattern creates a pattern of successes and failures.
// The pattern string uses 'S' for success and 'E' for error.
//
// Example:
//
//	model := fakellm.NewFakeModel().
//	    ErrorPattern("EEESSS", llm.ErrAPICall).
//	    When(fakellm.Any()).
//	    ThenRespondText("Success")
//
//	// First 3 calls fail, next 3 succeed, then repeats
func (m *FakeModel) ErrorPattern(pattern string, err error) *FakeModel {
	for i, char := range pattern {
		if char == 'E' || char == 'e' {
			m.When(CallNumber(i + 1)).
				Named(fmt.Sprintf("pattern-error-%d", i+1)).
				ThenError(err)
		}
	}

	return m
}

// CancelAfterDelay returns a context that will be cancelled after the specified delay.
// This is a helper for testing context cancellation.
//
// Example:
//
//	ctx := fakellm.CancelAfterDelay(context.Background(), 100*time.Millisecond)
//	_, err := model.Generate(ctx, req)
//	// err == context.Canceled after 100ms
func CancelAfterDelay(parent context.Context, delay time.Duration) context.Context {
	ctx, cancel := context.WithCancel(parent)
	time.AfterFunc(delay, cancel)

	return ctx
}

// AlwaysFail creates a model that always returns the specified error.
// This is useful for testing error handling paths.
//
// Example:
//
//	model := fakellm.AlwaysFail(llm.ErrInvalidConfig)
//	_, err := model.Generate(ctx, req)
//	// errors.Is(err, llm.ErrInvalidConfig) == true
func AlwaysFail(err error) *FakeModel {
	return NewFakeModel().When(Any()).ThenError(err)
}
