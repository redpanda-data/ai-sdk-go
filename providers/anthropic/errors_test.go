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

package anthropic

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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

func TestClassifyHTTPError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		wantBase   error
		wantRetry  bool
		wantCode   string
	}{
		{"rate limit", 429, llm.ErrRateLimitExceeded, true, "rate_limit_exceeded"},
		{"server error 500", 500, llm.ErrServerError, true, "internal_server_error"},
		{"bad gateway 502", 502, llm.ErrServerError, true, "bad_gateway"},
		{"service unavailable 503", 503, llm.ErrServerError, true, "service_unavailable"},
		{"overloaded 529", 529, llm.ErrServerError, true, "overloaded"},
		{"bad request 400", 400, llm.ErrInvalidInput, false, "bad_request"},
		{"unauthorized 401", 401, llm.ErrAPICall, false, "unauthorized"},
		{"forbidden 403", 403, llm.ErrAPICall, false, "forbidden"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			reqURL, _ := url.Parse("https://api.anthropic.com/v1/messages")
			apiErr := &anthropic.Error{
				StatusCode: tt.statusCode,
				Request:    &http.Request{Method: http.MethodPost, URL: reqURL},
				Response:   &http.Response{StatusCode: tt.statusCode},
			}

			result := classifyError(apiErr)
			require.Error(t, result)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.Equal(t, tt.wantRetry, pe.Retryable)
			assert.Equal(t, tt.wantCode, pe.Code)
		})
	}
}

func TestClassifySSEError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		errType   string
		errMsg    string
		wantBase  error
		wantRetry bool
	}{
		{"api_error", "api_error", "Internal server error", llm.ErrServerError, true},
		{"overloaded_error", "overloaded_error", "Overloaded", llm.ErrServerError, true},
		{"rate_limit_error", "rate_limit_error", "Rate limit exceeded", llm.ErrRateLimitExceeded, true},
		{"invalid_request_error", "invalid_request_error", "Invalid request", llm.ErrInvalidInput, false},
		{"authentication_error", "authentication_error", "Invalid API key", llm.ErrAPICall, false},
		{"permission_error", "permission_error", "Permission denied", llm.ErrAPICall, false},
		{"not_found_error", "not_found_error", "Not found", llm.ErrAPICall, false},
		{"unknown_error_type", "something_new", "New error", llm.ErrServerError, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			jsonPayload := fmt.Sprintf(
				`{"type":"error","error":{"type":"%s","message":"%s"}}`,
				tt.errType, tt.errMsg,
			)
			err := fmt.Errorf("received error while streaming: %s", jsonPayload)

			result := classifyError(err)
			require.Error(t, result)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.Equal(t, tt.wantRetry, pe.Retryable)
			assert.Equal(t, tt.errType, pe.Code)
			assert.Equal(t, tt.errMsg, pe.Message)
		})
	}
}

func TestClassifySSEError_UnparseableJSON(t *testing.T) {
	t.Parallel()

	err := errors.New("received error while streaming: {invalid json")
	result := classifyError(err)

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrServerError)
	assert.True(t, pe.Retryable)
	assert.Equal(t, "unparseable_sse_error", pe.Code)
}

func TestClassifySSEError_NonJSONSuffix(t *testing.T) {
	t.Parallel()

	err := errors.New("received error while streaming: not json at all")
	result := classifySSEError(err)
	assert.Nil(t, result)
}

func TestClassifySSEError_EmptyErrorType(t *testing.T) {
	t.Parallel()

	err := errors.New(`received error while streaming: {"type":"error","error":{"type":"","message":"something broke"}}`)
	result := classifyError(err)

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrAPICall)
	assert.False(t, pe.Retryable)
	assert.Equal(t, "unknown_sse_error", pe.Code)
}

func TestClassifySSEError_NotSSE(t *testing.T) {
	t.Parallel()

	err := errors.New("some other error")
	result := classifySSEError(err)
	assert.Nil(t, result)
}

func TestIsRetryable_ProviderError(t *testing.T) {
	t.Parallel()

	retryable := &llm.ProviderError{
		Base:      llm.ErrServerError,
		Retryable: true,
	}
	assert.True(t, llm.IsRetryable(retryable))

	notRetryable := &llm.ProviderError{
		Base:      llm.ErrInvalidInput,
		Retryable: false,
	}
	assert.False(t, llm.IsRetryable(notRetryable))
}

func TestIsRetryable_Wrapped(t *testing.T) {
	t.Parallel()

	wrapped := fmt.Errorf("wrapped: %w", &llm.ProviderError{
		Base:      llm.ErrRateLimitExceeded,
		Retryable: true,
	})
	assert.True(t, llm.IsRetryable(wrapped))
}

func TestIsRetryable_SentinelFallback(t *testing.T) {
	t.Parallel()

	assert.True(t, llm.IsRetryable(fmt.Errorf("wrap: %w", llm.ErrServerError)))
	assert.True(t, llm.IsRetryable(fmt.Errorf("wrap: %w", llm.ErrRateLimitExceeded)))
	assert.False(t, llm.IsRetryable(fmt.Errorf("wrap: %w", llm.ErrInvalidInput)))
	assert.False(t, llm.IsRetryable(errors.New("plain error")))
}
