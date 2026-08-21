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
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	codeResponseDecode = "response_decode_error"
	codeTransport      = "transport_error"
	codeStreamError    = "stream_error"
)

// classifyError maps SDK errors to *llm.ProviderError.
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return classifyHTTPError(apiErr)
	}

	// A structured SSE error event (context overflow, mid-stream invalid
	// request) classifies precisely; anything else falls through to the
	// generic non-HTTP classification below.
	if streamErr, ok := errors.AsType[*ssestream.StreamError](err); ok {
		if pe := classifyStreamError(streamErr); pe != nil {
			return pe
		}
	}

	return classifyNonHTTPError(err)
}

func classifyNonHTTPError(err error) error {
	// Returned untouched: ProviderError.Unwrap yields only Base, so wrapping
	// these would break errors.Is against them at the call site.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}

	// Not retryable: the frame's payload is opaque here, and a mid-stream
	// failure is as likely to be terminal as transient.
	var streamErr *ssestream.StreamError
	if errors.As(err, &streamErr) {
		return &llm.ProviderError{
			Base:    llm.ErrAPICall,
			Code:    codeStreamError,
			Message: streamErr.Error(),
		}
	}

	// Well-formed JSON disagreeing with our structs replays identically.
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) {
		return &llm.ProviderError{
			Base:    llm.ErrResponseMapping,
			Code:    codeResponseDecode,
			Message: fmt.Sprintf("could not decode provider response: %s", err),
		}
	}

	// A mangled stream; a second attempt can land on a clean response.
	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      codeResponseDecode,
			Message:   fmt.Sprintf("could not decode provider response: %s", err),
			Retryable: true,
		}
	}

	// Do not narrow this to net.Error or an errno set. An h2 stream reset is
	// net/http's unexported http2StreamError: not a net.Error, unreachable by
	// errors.As, and h2 is the default for https upstreams. Whatever such a
	// check missed would silently become permanent, which costs the caller's
	// whole task where a needless retry costs three short attempts.
	return &llm.ProviderError{
		Base:      llm.ErrServerError,
		Code:      codeTransport,
		Message:   fmt.Sprintf("transport failure: %s", err),
		Retryable: true,
	}
}

// classifyHTTPError maps an OpenAI HTTP API error to a *llm.ProviderError.
func classifyHTTPError(apiErr *openai.Error) *llm.ProviderError {
	retryable, base := classifyStatusCode(apiErr.StatusCode)

	if errors.Is(base, llm.ErrInvalidInput) && isContextOverflow(apiErr.Code, apiErr.Message) {
		base = llm.ErrContextOverflow
	}

	return &llm.ProviderError{
		Base:      base,
		Code:      apiErr.Code,
		Message:   apiErr.Message,
		Retryable: retryable,
	}
}

// classifyStatusCode maps HTTP status codes to sentinel errors and retryability.
func classifyStatusCode(code int) (bool, error) {
	switch code {
	case 429:
		return true, llm.ErrRateLimitExceeded
	case 500, 502, 503, 529:
		return true, llm.ErrServerError
	case 400:
		return false, llm.ErrInvalidInput
	case 401, 403:
		return false, llm.ErrAPICall
	default:
		if code >= 500 {
			return true, llm.ErrServerError
		}

		return false, llm.ErrAPICall
	}
}

// streamErrorPayload covers both JSON shapes an SSE error event takes: flat
// {"type","code","message"} events and the same fields nested under "error".
type streamErrorPayload struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Error   struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// classifyStreamError parses an SSE error ("received error while streaming:
// {json}") and classifies its payload. Returns nil when the payload carries no
// recognizable fields, letting the error fall through to classifyNonHTTPError.
func classifyStreamError(streamErr *ssestream.StreamError) *llm.ProviderError {
	msg := streamErr.Error()

	start := strings.IndexByte(msg, '{')
	if start < 0 {
		return nil
	}

	var payload streamErrorPayload
	if err := json.Unmarshal([]byte(msg[start:]), &payload); err != nil {
		return nil //nolint:nilerr // unparseable payload falls through unclassified
	}

	errType, code, message := payload.Type, payload.Code, payload.Message
	if payload.Error.Code != "" || payload.Error.Message != "" {
		errType, code, message = payload.Error.Type, payload.Error.Code, payload.Error.Message
	}

	if code == "" && message == "" {
		return nil
	}

	base, retryable := llm.ErrServerError, true

	switch {
	case isContextOverflow(code, message):
		base, retryable = llm.ErrContextOverflow, false
	case errType == "invalid_request_error":
		base, retryable = llm.ErrInvalidInput, false
	case code == "rate_limit_exceeded" || errType == "rate_limit_error":
		base, retryable = llm.ErrRateLimitExceeded, true
	}

	return &llm.ProviderError{
		Base:      base,
		Code:      code,
		Message:   message,
		Retryable: retryable,
	}
}

// isContextOverflow reports whether an error code or message describes the
// request exceeding the model's context window. Many compatible backends emit
// only a message, so the pattern list mirrors LiteLLM's cross-provider set.
func isContextOverflow(code, message string) bool {
	if code == "context_length_exceeded" {
		return true
	}

	// A per-string char cap, not a conversation overflow.
	if code == "string_above_max_length" {
		return false
	}

	msg := strings.ToLower(message)

	for _, pattern := range []string{
		"context_length_exceeded",
		"context length exceeded",
		"exceeds the context window",
		"maximum context length",
		"model's maximum context limit",
		"exceed context limit",
		"is longer than the model's context length",
		"input tokens exceed the configured limit",
		"exceeds the available context size",
		"exceeds the maximum number of tokens allowed",
		"`inputs` tokens + `max_new_tokens` must be",
	} {
		if strings.Contains(msg, pattern) {
			return true
		}
	}

	return false
}
