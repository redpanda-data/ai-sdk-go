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

// badRequest400 builds an *anthropic.Error for a 400 response whose body is the
// given provider JSON, mirroring how the SDK populates it from a real response.
func badRequest400(t *testing.T, body string) *anthropic.Error {
	t.Helper()

	reqURL, _ := url.Parse("https://api.anthropic.com/v1/messages")
	apiErr := &anthropic.Error{}
	require.NoError(t, apiErr.UnmarshalJSON([]byte(body)))
	apiErr.StatusCode = 400
	apiErr.Request = &http.Request{Method: http.MethodPost, URL: reqURL}
	apiErr.Response = &http.Response{StatusCode: http.StatusBadRequest}

	return apiErr
}

func TestClassifyError_ContextWindowExceeded(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		body string
	}{
		{
			name: "input plus max_tokens exceed the window",
			body: `{"type":"error","error":{"type":"invalid_request_error","message":"input length and max_tokens exceed context limit: 188240 + 21333 > 200000, decrease input length or max_tokens and try again"}}`,
		},
		{
			name: "prompt alone is too long",
			body: `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := classifyError(badRequest400(t, tt.body))

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, llm.ErrContextWindowExceeded)
			require.ErrorIs(t, pe, llm.ErrInvalidInput,
				"the overflow sentinel must stay a specific case of ErrInvalidInput so existing callers still match")
			assert.False(t, pe.Retryable, "context-window overflow is not retryable as-is")
		})
	}
}

// TestClassifyError_OverflowPhraseOnlyInEchoedInput guards against matching the
// whole response body: Anthropic 400s can echo request content (tool schemas,
// field values), so the phrase must only count when it is the error message.
func TestClassifyError_OverflowPhraseOnlyInEchoedInput(t *testing.T) {
	t.Parallel()

	body := `{"type":"error","error":{"type":"invalid_request_error","message":"tools.0.custom.input_schema: Extra inputs are not permitted"},"request":{"tools":[{"name":"check","description":"Reports whether the prompt is too long"}]}}`
	result := classifyError(badRequest400(t, body))

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.NotErrorIs(t, pe, llm.ErrContextWindowExceeded)
	assert.Equal(t, "bad_request", pe.Code)
}

// TestClassifyError_OrdinaryBadRequest guards the detection from over-matching:
// a plain invalid_request_error must stay ErrInvalidInput, not context overflow.
func TestClassifyError_OrdinaryBadRequest(t *testing.T) {
	t.Parallel()

	body := `{"type":"error","error":{"type":"invalid_request_error","message":"messages: at least one message is required"}}`
	result := classifyError(badRequest400(t, body))

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrInvalidInput)
	assert.NotErrorIs(t, pe, llm.ErrContextWindowExceeded)
}

// TestClassifySSEError_ContextWindowExceeded covers the same overflow surfacing
// through the streaming error path.
func TestClassifySSEError_ContextWindowExceeded(t *testing.T) {
	t.Parallel()

	payload := `{"type":"error","error":{"type":"invalid_request_error","message":"input length and max_tokens exceed context limit: 188240 + 21333 > 200000"}}`
	err := fmt.Errorf("received error while streaming: %s", payload)

	result := classifyError(err)

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrContextWindowExceeded)
	assert.False(t, pe.Retryable)
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
