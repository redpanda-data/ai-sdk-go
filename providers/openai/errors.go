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
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// contextLengthExceededCode is OpenAI's error code for a request that does
// not fit the model's context window. The SDK ships no constant for it.
const contextLengthExceededCode = "context_length_exceeded"

// classifyError maps OpenAI SDK errors to *llm.ProviderError with the
// appropriate sentinel base and Retryable flag. It handles HTTP errors
// (oai.Error, also surfaced via stream.Err() for rejected streaming requests)
// and SSE error events (ssestream.StreamError).
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	if apiErr, ok := errors.AsType[*oai.Error](err); ok {
		return classifyHTTPError(apiErr)
	}

	if streamErr, ok := errors.AsType[*ssestream.StreamError](err); ok {
		if pe := classifyStreamError(streamErr); pe != nil {
			return pe
		}
	}

	return err
}

// classifyHTTPError maps an OpenAI HTTP API error to a *llm.ProviderError.
func classifyHTTPError(apiErr *oai.Error) *llm.ProviderError {
	retryable, base := classifyStatusCode(apiErr.StatusCode)

	switch {
	case isContextOverflow(apiErr.Code, apiErr.Message):
		base = llm.ErrContextOverflow
		retryable = false

	case apiErr.StatusCode == http.StatusTooManyRequests &&
		strings.HasPrefix(apiErr.Message, "Request too large"):
		// A single request over the TPM ceiling never succeeds on retry.
		// Type "tokens" alone is not enough: ordinary transient TPM
		// exhaustion ("Rate limit reached ...") carries it too.
		retryable = false
	}

	return &llm.ProviderError{
		Base:      base,
		Code:      apiErr.Code,
		Message:   apiErr.Message,
		Retryable: retryable,
	}
}

// streamErrorPayload covers both JSON shapes an SSE error event takes: the
// Responses API emits flat {"type","code","message","param"} events, while
// Chat Completions wraps the same fields under a top-level "error" key.
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
// recognizable fields, letting the error fall through unclassified.
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
// request exceeding the model's context window. The code is the stable
// signal; the message patterns catch variants that drop it. Per-string caps
// ("string_above_max_length") and oversized max_tokens deliberately do not
// match — they are config errors, not overflows.
func isContextOverflow(code, message string) bool {
	if code == contextLengthExceededCode {
		return true
	}

	msg := strings.ToLower(message)

	return strings.Contains(msg, "input tokens exceed the configured limit") || // gpt-5 input cap
		strings.Contains(msg, "exceeds the context window") || // Responses API
		strings.Contains(msg, "maximum context length") // Chat Completions / Azure
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
