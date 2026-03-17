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

		eventCount := 0

		var lastErr error

		// Read events using range loop
		for _, err := range model.GenerateEvents(context.Background(), req) {
			if err != nil {
				lastErr = err
				break
			}

			eventCount++
		}

		// Should have read 2 events before the error
		require.Equal(t, 2, eventCount, "Should receive 2 events before error")
		require.Error(t, lastErr)
		require.ErrorIs(t, lastErr, llm.ErrStreamClosed, "Should return llm.ErrStreamClosed")
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
