package fakellm

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestErrorHelperMethods verifies that helper methods inject the correct llm package errors.
func TestErrorHelperMethods(t *testing.T) {
	t.Parallel()

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	t.Run("RateLimitOnce returns llm.ErrAPICall", func(t *testing.T) {
		t.Parallel()

		model := NewFakeModel().RateLimitOnce()

		// First call should fail with llm.ErrAPICall
		_, err := model.Generate(context.Background(), req)
		require.Error(t, err)
		require.ErrorIs(t, err, llm.ErrAPICall, "Should return llm.ErrAPICall")

		// Second call should succeed
		model.When(Any()).ThenRespondText("Success")
		_, err = model.Generate(context.Background(), req)
		require.NoError(t, err)
	})

	t.Run("RateLimitNTimes returns llm.ErrAPICall N times", func(t *testing.T) {
		t.Parallel()

		model := NewFakeModel().
			RateLimitNTimes(3).
			When(Any()).
			ThenRespondText("Success")

		// First 3 calls should fail
		for i := range 3 {
			_, err := model.Generate(context.Background(), req)
			require.Error(t, err, "Call %d should fail", i+1)
			require.ErrorIs(t, err, llm.ErrAPICall, "Call %d should return llm.ErrAPICall", i+1)
		}

		// Fourth call should succeed
		_, err := model.Generate(context.Background(), req)
		require.NoError(t, err)
	})

	t.Run("APIErrorOnce returns llm.ErrAPICall", func(t *testing.T) {
		t.Parallel()

		model := NewFakeModel().
			APIErrorOnce().
			When(Any()).
			ThenRespondText("Success")

		// First call should fail
		_, err := model.Generate(context.Background(), req)
		require.Error(t, err)
		require.ErrorIs(t, err, llm.ErrAPICall, "Should return llm.ErrAPICall")

		// Second call should succeed
		_, err = model.Generate(context.Background(), req)
		require.NoError(t, err)
	})

	t.Run("TimeoutOnce returns context.DeadlineExceeded", func(t *testing.T) {
		t.Parallel()

		model := NewFakeModel().
			TimeoutOnce().
			When(Any()).
			ThenRespondText("Success")

		// First call should fail with timeout
		_, err := model.Generate(context.Background(), req)
		require.Error(t, err)
		require.ErrorIs(t, err, context.DeadlineExceeded, "Should return context.DeadlineExceeded")

		// Second call should succeed
		_, err = model.Generate(context.Background(), req)
		require.NoError(t, err)
	})

	t.Run("MidStreamError with llm.ErrStreamClosed", func(t *testing.T) {
		t.Parallel()

		model := NewFakeModel().MidStreamError(2, llm.ErrStreamClosed)

		stream, err := model.GenerateStream(context.Background(), req)
		require.NoError(t, err)

		defer func() { _ = stream.Close() }()

		// Read a couple chunks successfully
		_, err = stream.Recv()
		require.NoError(t, err)
		_, err = stream.Recv()
		require.NoError(t, err)

		// Next recv should hit the error
		_, err = stream.Recv()
		require.Error(t, err)
		require.ErrorIs(t, err, llm.ErrStreamClosed, "Should return llm.ErrStreamClosed")
	})

	t.Run("ErrorAfterNCalls", func(t *testing.T) {
		t.Parallel()

		customErr := errors.New("custom error after 2 calls")
		model := NewFakeModel().
			ErrorAfterNCalls(2, customErr).
			When(Any()).
			ThenRespondText("Success")

		// First 2 calls should succeed
		_, err := model.Generate(context.Background(), req)
		require.NoError(t, err)
		_, err = model.Generate(context.Background(), req)
		require.NoError(t, err)

		// Third call should fail with custom error
		_, err = model.Generate(context.Background(), req)
		require.Error(t, err)
		require.ErrorIs(t, err, customErr, "Should return the custom error")
	})

	t.Run("ErrorPattern", func(t *testing.T) {
		t.Parallel()

		model := NewFakeModel().
			ErrorPattern("EES", llm.ErrAPICall).
			When(Any()).
			ThenRespondText("Success")

		// Call 1: Error
		_, err := model.Generate(context.Background(), req)
		require.Error(t, err)
		require.ErrorIs(t, err, llm.ErrAPICall)

		// Call 2: Error
		_, err = model.Generate(context.Background(), req)
		require.Error(t, err)
		require.ErrorIs(t, err, llm.ErrAPICall)

		// Call 3: Success
		_, err = model.Generate(context.Background(), req)
		require.NoError(t, err)
	})

	t.Run("AlwaysFail", func(t *testing.T) {
		t.Parallel()

		model := AlwaysFail(llm.ErrInvalidConfig)

		// All calls should fail
		for i := range 3 {
			_, err := model.Generate(context.Background(), req)
			require.Error(t, err)
			require.ErrorIs(t, err, llm.ErrInvalidConfig, "Call %d should return llm.ErrInvalidConfig", i+1)
		}
	})
}

// TestCancelAfterDelay verifies the context cancellation helper.
func TestCancelAfterDelay(t *testing.T) {
	t.Parallel()

	ctx := CancelAfterDelay(context.Background(), 1) // 1 nanosecond

	// Wait a bit to ensure cancellation happens
	<-ctx.Done()

	err := ctx.Err()
	assert.ErrorIs(t, err, context.Canceled, "Context should be canceled")
}
