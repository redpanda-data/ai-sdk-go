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

package google

import (
	"errors"
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestClassifyError_Nil(t *testing.T) {
	t.Parallel()

	assert.NoError(t, classifyError(nil))
}

func TestClassifyError_UnknownError(t *testing.T) {
	t.Parallel()

	err := errors.New("something unexpected")
	result := classifyError(err)
	assert.Equal(t, err, result)
}

func TestClassifyAPIError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		code      int
		status    string
		message   string
		wantBase  error
		wantRetry bool
	}{
		{"rate limit", 429, "RESOURCE_EXHAUSTED", "Rate limit exceeded", llm.ErrRateLimitExceeded, true},
		{"server error 500", 500, "INTERNAL", "Internal server error", llm.ErrServerError, true},
		{"bad gateway 502", 502, "UNAVAILABLE", "Bad gateway", llm.ErrServerError, true},
		{"service unavailable 503", 503, "UNAVAILABLE", "Service unavailable", llm.ErrServerError, true},
		{"bad request 400", 400, "INVALID_ARGUMENT", "Invalid request", llm.ErrInvalidInput, false},
		{
			// Recorded from the live Gemini API, 2026-08-21.
			"context overflow current wording", 400, "INVALID_ARGUMENT",
			"The input token count exceeds the maximum number of tokens allowed (1048576).",
			llm.ErrContextOverflow, false,
		},
		{
			"context overflow legacy vertex wording", 400, "INVALID_ARGUMENT",
			"Unable to submit request because the input token count is 40000 but model only supports up to 32768. Reduce the input token count and try again.",
			llm.ErrContextOverflow, false,
		},
		{
			"max output tokens out of range", 400, "INVALID_ARGUMENT",
			"Unable to submit request because it has a maxOutputTokens value of 100117 but the supported range is from 1 (inclusive) to 65537 (exclusive). Update the value and try again.",
			llm.ErrInvalidInput, false,
		},
		{
			"bad temperature stays invalid input", 400, "INVALID_ARGUMENT",
			"* GenerateContentRequest.generation_config.temperature: temperature must be in the range [0.0, 2.0].",
			llm.ErrInvalidInput, false,
		},
		{"unauthorized 401", 401, "UNAUTHENTICATED", "Unauthorized", llm.ErrAPICall, false},
		{"forbidden 403", 403, "PERMISSION_DENIED", "Forbidden", llm.ErrAPICall, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			apiErr := genai.APIError{
				Code:    tt.code,
				Status:  tt.status,
				Message: tt.message,
			}

			result := classifyError(apiErr)
			require.Error(t, result)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.Equal(t, tt.wantRetry, pe.Retryable)
			assert.Equal(t, tt.status, pe.Code)
			assert.Equal(t, tt.message, pe.Message)

			if errors.Is(tt.wantBase, llm.ErrInvalidInput) && !errors.Is(tt.wantBase, llm.ErrContextOverflow) {
				assert.NotErrorIs(t, pe, llm.ErrContextOverflow)
			}
		})
	}
}

// timeoutError is a test helper that implements net.Error with Timeout()=true,
// mimicking Go's http.httpError produced on Client.Timeout exceeded.
type timeoutError struct{ msg string }

func (e *timeoutError) Error() string   { return e.msg }
func (e *timeoutError) Timeout() bool   { return true }
func (e *timeoutError) Temporary() bool { return true }

func TestClassifyError_NetworkTimeout(t *testing.T) {
	t.Parallel()

	// Simulate the error chain produced by Go's HTTP client on Client.Timeout:
	// url.Error -> httpError (timeout=true)
	timeoutErr := &url.Error{
		Op:  "Post",
		URL: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
		Err: &timeoutError{msg: "context deadline exceeded (Client.Timeout exceeded while awaiting headers)"},
	}

	result := classifyError(timeoutErr)
	require.Error(t, result)

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrServerError)
	assert.True(t, pe.Retryable)
	assert.Equal(t, "TIMEOUT", pe.Code)
}
