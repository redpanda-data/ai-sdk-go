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

package openai

import (
	"errors"
	"net/http"
	"net/url"
	"testing"

	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
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

// Payloads recorded from the live API (2026-08-21) plus documented variants.
func TestClassifyHTTPError_Fields(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		code       string
		errType    string
		message    string
		wantBase   error
		wantRetry  bool
	}{
		{
			name:       "chat completions context overflow",
			statusCode: 400, code: "context_length_exceeded", errType: "invalid_request_error",
			message:  "This model's maximum context length is 128000 tokens. However, your messages resulted in 194296 tokens. Please reduce the length of the messages.",
			wantBase: llm.ErrContextOverflow, wantRetry: false,
		},
		{
			name:       "responses API context overflow",
			statusCode: 400, code: "context_length_exceeded", errType: "invalid_request_error",
			message:  "Your input exceeds the context window of this model. Please adjust your input and try again.",
			wantBase: llm.ErrContextOverflow, wantRetry: false,
		},
		{
			name:       "gpt-5 input cap without the code",
			statusCode: 400, code: "", errType: "invalid_request_error",
			message:  "Input tokens exceed the configured limit of 272,000 tokens. Your messages resulted in 297,006 tokens.",
			wantBase: llm.ErrContextOverflow, wantRetry: false,
		},
		{
			name:       "max_tokens over output cap stays invalid input",
			statusCode: 400, code: "invalid_value", errType: "invalid_request_error",
			message:  "max_tokens is too large: 10000000. This model supports at most 16384 completion tokens, whereas you provided 10000000.",
			wantBase: llm.ErrInvalidInput, wantRetry: false,
		},
		{
			name:       "per-string length cap stays invalid input",
			statusCode: 400, code: "string_above_max_length", errType: "invalid_request_error",
			message:  "Invalid 'input[0].content': string too long. Expected a string with maximum length 10485760, but got a string with length 10500000 instead.",
			wantBase: llm.ErrInvalidInput, wantRetry: false,
		},
		{
			name:       "unrelated 400 stays invalid input",
			statusCode: 400, code: "decimal_above_max_value", errType: "invalid_request_error",
			message:  "Invalid 'temperature': decimal above maximum value. Expected a value <= 2, but got 9.5 instead.",
			wantBase: llm.ErrInvalidInput, wantRetry: false,
		},
		{
			name:       "TPM request too large is a non-retryable rate limit",
			statusCode: 429, code: "rate_limit_exceeded", errType: "tokens",
			message:  "Request too large for gpt-4.1 in organization org-x on tokens per min (TPM): Limit 30000, Requested 42638. The input or output tokens must be reduced in order to run successfully.",
			wantBase: llm.ErrRateLimitExceeded, wantRetry: false,
		},
		{
			name:       "transient TPM exhaustion stays retryable",
			statusCode: 429, code: "rate_limit_exceeded", errType: "tokens",
			message:  "Rate limit reached for gpt-4o-mini in organization org-x on tokens per min (TPM): Limit 30000, Used 29000, Requested 2000.",
			wantBase: llm.ErrRateLimitExceeded, wantRetry: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			reqURL, _ := url.Parse("https://api.openai.com/v1/responses")
			apiErr := &oai.Error{
				StatusCode: tt.statusCode,
				Code:       tt.code,
				Type:       tt.errType,
				Message:    tt.message,
				Request:    &http.Request{Method: http.MethodPost, URL: reqURL},
				Response:   &http.Response{StatusCode: tt.statusCode},
			}

			result := classifyError(apiErr)
			require.Error(t, result)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.Equal(t, tt.wantRetry, pe.Retryable)
			assert.Equal(t, tt.code, pe.Code)
			assert.Equal(t, tt.message, pe.Message)

			if errors.Is(tt.wantBase, llm.ErrInvalidInput) && !errors.Is(tt.wantBase, llm.ErrContextOverflow) {
				assert.NotErrorIs(t, pe, llm.ErrContextOverflow)
			}
		})
	}
}

// Flat Responses API payloads (recorded live 2026-08-21) and nested Chat
// Completions payloads.
func TestClassifyStreamError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		payload   string
		wantBase  error
		wantCode  string
		wantRetry bool
	}{
		{
			name:      "responses streaming context overflow (flat payload)",
			payload:   `received error while streaming: {"type":"invalid_request_error","code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again.","param":"input"}`,
			wantBase:  llm.ErrContextOverflow,
			wantCode:  "context_length_exceeded",
			wantRetry: false,
		},
		{
			name:      "chat completions mid-stream overflow (nested payload)",
			payload:   `received error while streaming: {"error":{"message":"This model's maximum context length is 128000 tokens.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}`,
			wantBase:  llm.ErrContextOverflow,
			wantCode:  "context_length_exceeded",
			wantRetry: false,
		},
		{
			name:      "mid-stream invalid request stays invalid input",
			payload:   `received error while streaming: {"type":"invalid_request_error","code":"invalid_value","message":"bad param","param":"tools"}`,
			wantBase:  llm.ErrInvalidInput,
			wantCode:  "invalid_value",
			wantRetry: false,
		},
		{
			name:      "mid-stream server error stays retryable",
			payload:   `received error while streaming: {"error":{"message":"The server had an error processing your request.","type":"server_error","code":"server_error"}}`,
			wantBase:  llm.ErrServerError,
			wantCode:  "server_error",
			wantRetry: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := classifyError(&ssestream.StreamError{Message: tt.payload})
			require.Error(t, result)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.Equal(t, tt.wantRetry, pe.Retryable)
			assert.Equal(t, tt.wantCode, pe.Code)
		})
	}
}

func TestClassifyStreamError_EmptyPayload(t *testing.T) {
	t.Parallel()

	// An empty terminal error event must still surface, unclassified.
	streamErr := &ssestream.StreamError{Message: "received error while streaming: {}"}
	result := classifyError(streamErr)
	assert.Equal(t, error(streamErr), result)
}

func TestClassifyHTTPError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		wantBase   error
		wantRetry  bool
	}{
		{"rate limit", 429, llm.ErrRateLimitExceeded, true},
		{"server error 500", 500, llm.ErrServerError, true},
		{"bad gateway 502", 502, llm.ErrServerError, true},
		{"service unavailable 503", 503, llm.ErrServerError, true},
		{"overloaded 529", 529, llm.ErrServerError, true},
		{"bad request 400", 400, llm.ErrInvalidInput, false},
		{"unauthorized 401", 401, llm.ErrAPICall, false},
		{"forbidden 403", 403, llm.ErrAPICall, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			reqURL, _ := url.Parse("https://api.openai.com/v1/responses")
			apiErr := &oai.Error{
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
		})
	}
}

func TestClassifyHTTPError_ProviderErrorFields(t *testing.T) {
	t.Parallel()

	reqURL, _ := url.Parse("https://api.openai.com/v1/responses")
	apiErr := &oai.Error{
		StatusCode: http.StatusTooManyRequests,
		Code:       "rate_limit_exceeded",
		Message:    "Rate limit exceeded",
		Request:    &http.Request{Method: http.MethodPost, URL: reqURL},
		Response:   &http.Response{StatusCode: http.StatusTooManyRequests},
	}

	result := classifyError(apiErr)

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	assert.Equal(t, "rate_limit_exceeded", pe.Code)
	assert.Equal(t, "Rate limit exceeded", pe.Message)
	assert.True(t, pe.Retryable)
}
