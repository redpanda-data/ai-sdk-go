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

package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestClassifyError_Nil(t *testing.T) {
	t.Parallel()

	assert.NoError(t, classifyError(nil))
}

func TestClassifyError_UnrecognisedIsRetryableTransport(t *testing.T) {
	t.Parallel()

	result := classifyError(errors.New("something unexpected"))

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrServerError)
	assert.Equal(t, codeTransport, pe.Code)
	assert.True(t, pe.Retryable)
	assert.True(t, llm.IsRetryable(result))
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
		{"bad request 400", 400, llm.ErrInvalidInput, false},
		{"unauthorized 401", 401, llm.ErrAPICall, false},
		{"forbidden 403", 403, llm.ErrAPICall, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			reqURL, _ := url.Parse("https://api.openai.com/v1/chat/completions")
			apiErr := &openai.Error{
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

	reqURL, _ := url.Parse("https://api.openai.com/v1/chat/completions")
	apiErr := &openai.Error{
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

func TestClassifyError_DecodeFailure(t *testing.T) {
	t.Parallel()

	// The shape openai-go's SSE decoder produced on an empty data buffer.
	err := json.Unmarshal([]byte{}, &struct{}{})
	require.Error(t, err)

	result := classifyError(err)

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrServerError)
	assert.Equal(t, codeResponseDecode, pe.Code)
	assert.True(t, pe.Retryable)
	assert.True(t, llm.IsRetryable(result))
	assert.Contains(t, pe.Message, "unexpected end of JSON input")
}

func TestClassifyError_DecodeTypeMismatchIsNotRetryable(t *testing.T) {
	t.Parallel()

	var target struct {
		N int `json:"n"`
	}

	err := json.Unmarshal([]byte(`{"n":"not-a-number"}`), &target)
	require.Error(t, err)

	result := classifyError(err)

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrResponseMapping)
	assert.Equal(t, codeResponseDecode, pe.Code)
	assert.False(t, pe.Retryable, "a type mismatch replays identically")
	assert.False(t, llm.IsRetryable(result))
}

func TestClassifyError_StreamError(t *testing.T) {
	t.Parallel()

	result := classifyError(&ssestream.StreamError{Message: "received error while streaming: overloaded"})

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrAPICall)
	assert.Equal(t, codeStreamError, pe.Code)
	assert.False(t, pe.Retryable)
	assert.Contains(t, pe.Message, "overloaded")
}

func TestClassifyError_Transport(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  error
	}{
		{"net error", &net.OpError{Op: "dial", Err: errors.New("connection refused")}},
		{"url error", &url.Error{Op: "Post", URL: "https://openrouter.ai", Err: errors.New("EOF")}},
		{"truncated body", fmt.Errorf("reading body: %w", io.ErrUnexpectedEOF)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := classifyError(tt.err)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, llm.ErrServerError)
			assert.Equal(t, codeTransport, pe.Code)
			assert.True(t, pe.Retryable)
		})
	}
}

// Cancellation and deadlines are the caller's, and callers match them with
// errors.Is. ProviderError.Unwrap returns only Base, so wrapping them here
// would sever that chain.
func TestClassifyError_ContextErrorsPassThrough(t *testing.T) {
	t.Parallel()

	for _, sentinel := range []error{context.Canceled, context.DeadlineExceeded} {
		t.Run(sentinel.Error(), func(t *testing.T) {
			t.Parallel()

			wrapped := &url.Error{Op: "Post", URL: "https://openrouter.ai", Err: sentinel}

			result := classifyError(wrapped)
			require.ErrorIs(t, result, sentinel)

			var pe *llm.ProviderError
			assert.NotErrorAs(t, result, &pe)
		})
	}
}

// TestStreamDecodeErrorIsClassified exercises the real path: a malformed SSE
// frame from the upstream, through model.GenerateEvents, out to the caller.
// Before decode errors were classified, this arrived as a bare "unexpected end
// of JSON input" with no category, no code, and no status.
func TestStreamDecodeErrorIsClassified(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"gen-1\",\"choices\":[\n\n"))
	}))
	t.Cleanup(server.Close)

	provider, err := NewProvider("sk-test-key", WithBaseURL(server.URL))
	require.NoError(t, err)

	model, err := provider.NewModel("m")
	require.NoError(t, err)

	var streamErr error

	for _, err := range model.GenerateEvents(t.Context(), &llm.Request{
		Messages: []llm.Message{{
			Role:    llm.RoleUser,
			Content: []llm.Part{llm.NewTextPart("hi")},
		}},
	}) {
		if err != nil {
			streamErr = err
			break
		}
	}

	require.Error(t, streamErr)
	require.ErrorIs(t, streamErr, llm.ErrAPICall)
	require.ErrorIs(t, streamErr, llm.ErrServerError)

	var pe *llm.ProviderError
	require.ErrorAs(t, streamErr, &pe)
	assert.Equal(t, codeResponseDecode, pe.Code)
	assert.True(t, llm.IsRetryable(streamErr))
	assert.Equal(t,
		"API call failed: server error: [response_decode_error] "+
			"could not decode provider response: unexpected end of JSON input",
		streamErr.Error())
}

// Guards the classifier's transport default against being narrowed; see the
// comment on that arm in errors.go.
func TestHTTP2StreamResetIsRetryable(t *testing.T) {
	t.Parallel()

	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"gen-1","object":"chat.completion","choices":[`))

		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}

		// A short Content-Length yields a plain EOF and misses this path.
		panic(http.ErrAbortHandler) //nolint:forbidigo // aborts h2 with RST_STREAM
	}))
	server.EnableHTTP2 = true
	server.StartTLS()
	t.Cleanup(server.Close)

	provider, err := NewProvider("sk-test-key",
		WithBaseURL(server.URL+"/v1"), WithHTTPClient(server.Client()))
	require.NoError(t, err)

	model, err := provider.NewModel("m")
	require.NoError(t, err)

	_, err = model.Generate(t.Context(), &llm.Request{
		Messages: []llm.Message{{
			Role:    llm.RoleUser,
			Content: []llm.Part{llm.NewTextPart("hi")},
		}},
	})
	require.Error(t, err)
	require.ErrorContains(t, err, "INTERNAL_ERROR", "expected an h2 stream reset")

	var pe *llm.ProviderError
	require.ErrorAs(t, err, &pe)
	assert.Equal(t, codeTransport, pe.Code)
	assert.True(t, pe.Retryable)
	assert.True(t, llm.IsRetryable(err))
}

// Overflow wordings of OpenAI-compatible backends (DeepSeek, vLLM, llama.cpp, TGI).
func TestClassifyHTTPError_ContextOverflow(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		code     string
		message  string
		wantBase error
	}{
		{
			name:     "openai style code",
			code:     "context_length_exceeded",
			message:  "This model's maximum context length is 65536 tokens.",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "deepseek message without code",
			code:     "invalid_request_error",
			message:  "This model's maximum context length is 65536 tokens. However, you requested 81920 tokens.",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "vllm style",
			code:     "",
			message:  "This model's maximum context length is 32768 tokens. However, you requested 33000 tokens in the messages, Please reduce the length of the messages.",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "llama.cpp style",
			code:     "",
			message:  "the request exceeds the available context size. try increasing the context size or enable context shift",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "tgi style",
			code:     "",
			message:  "`inputs` tokens + `max_new_tokens` must be <= 4096",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "per-string cap stays invalid input",
			code:     "string_above_max_length",
			message:  "Invalid 'input': string too long. Expected a string with maximum length 10485760.",
			wantBase: llm.ErrInvalidInput,
		},
		{
			name:     "unrelated 400 stays invalid input",
			code:     "invalid_value",
			message:  "Invalid 'temperature': decimal above maximum value.",
			wantBase: llm.ErrInvalidInput,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			reqURL, _ := url.Parse("https://api.deepseek.com/v1/chat/completions")
			apiErr := &openai.Error{
				StatusCode: 400,
				Code:       tt.code,
				Message:    tt.message,
				Request:    &http.Request{Method: http.MethodPost, URL: reqURL},
				Response:   &http.Response{StatusCode: http.StatusBadRequest},
			}

			result := classifyError(apiErr)
			require.Error(t, result)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.False(t, pe.Retryable)

			if errors.Is(tt.wantBase, llm.ErrInvalidInput) && !errors.Is(tt.wantBase, llm.ErrContextOverflow) {
				assert.NotErrorIs(t, pe, llm.ErrContextOverflow)
			}
		})
	}
}

func TestClassifyStreamError_ContextOverflow(t *testing.T) {
	t.Parallel()

	streamErr := &ssestream.StreamError{
		Message: `received error while streaming: {"error":{"message":"This model's maximum context length is 65536 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}`,
	}

	result := classifyError(streamErr)

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	require.ErrorIs(t, pe, llm.ErrContextOverflow)
	assert.False(t, pe.Retryable)
}
