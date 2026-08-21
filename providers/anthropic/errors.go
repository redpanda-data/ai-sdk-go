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
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// classifyError maps Anthropic SDK errors to *llm.ProviderError with the
// appropriate sentinel base and Retryable flag. It handles HTTP errors
// (anthropic.Error, also surfaced via stream.Err() for rejected streaming
// requests) and the SDK's stringly SSE streaming errors.
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	// Try HTTP API error first
	if apiErr, ok := errors.AsType[*anthropic.Error](err); ok {
		return classifyHTTPError(apiErr)
	}

	// Try SSE streaming error
	if pe := classifySSEError(err); pe != nil {
		return pe
	}

	// Unknown error type — return as-is
	return err
}

// classifyHTTPError maps an Anthropic HTTP API error to a *llm.ProviderError.
func classifyHTTPError(apiErr *anthropic.Error) *llm.ProviderError {
	retryable, base := classifyStatusCode(apiErr.StatusCode)
	code := statusCodeToString(apiErr.StatusCode)
	message := apiErr.Error()

	// anthropic.Error exposes no typed error fields; parse the raw body for
	// the provider's error type and bare message.
	var payload sseErrorPayload
	if jsonErr := json.Unmarshal([]byte(apiErr.RawJSON()), &payload); jsonErr == nil && payload.Error.Type != "" {
		code = payload.Error.Type
		message = payload.Error.Message

		if apiErr.StatusCode == http.StatusBadRequest && payload.Error.Type == "invalid_request_error" &&
			isContextOverflowMessage(payload.Error.Message) {
			base = llm.ErrContextOverflow
		}
	}

	return &llm.ProviderError{
		Base:      base,
		Code:      code,
		Message:   message,
		Retryable: retryable,
	}
}

// isContextOverflowMessage reports whether a message describes the request
// exceeding the model's context window. Anthropic has no dedicated error type
// for this, so message text is the only signal. Oversized max_tokens is a
// config error, not an overflow, and deliberately does not match.
func isContextOverflowMessage(msg string) bool {
	msg = strings.ToLower(msg)

	return strings.Contains(msg, "prompt is too long") || // input alone too big, every model
		strings.Contains(msg, "exceed context limit") || // input + max_tokens, pre-4.5 models
		strings.Contains(msg, "prompt: length") // Claude-2-era wording
}

// classifySSEError parses the SDK's SSE streaming error format and classifies it.
// The format is: "received error while streaming: {json}"
// where JSON is: {"type":"error","error":{"type":"<error_type>","message":"<msg>"}}.
//
// String matching is required because the Anthropic SDK (v1.22.1) does not expose
// typed errors for SSE streams — ssestream.Stream returns a plain error via
// stream.Err(), constructed as fmt.Errorf("received error while streaming: %s", data).
// Returns nil when the error doesn't match, letting it fall through unclassified.
func classifySSEError(err error) *llm.ProviderError {
	msg := err.Error()

	const prefix = "received error while streaming: "
	if !strings.HasPrefix(msg, prefix) {
		return nil
	}

	jsonStr := msg[len(prefix):]

	// Verify it looks like JSON before parsing
	if !strings.HasPrefix(jsonStr, "{") {
		return nil
	}

	var sseErr sseErrorPayload
	if jsonErr := json.Unmarshal([]byte(jsonStr), &sseErr); jsonErr != nil {
		// Can't parse — treat as server error (retryable)
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      "unparseable_sse_error",
			Message:   msg,
			Retryable: true,
		}
	}

	// Guard against empty error type
	if sseErr.Error.Type == "" {
		return &llm.ProviderError{
			Base:      llm.ErrAPICall,
			Code:      "unknown_sse_error",
			Message:   msg,
			Retryable: false,
		}
	}

	retryable, base := classifySSEErrorType(sseErr.Error.Type)

	if errors.Is(base, llm.ErrInvalidInput) && isContextOverflowMessage(sseErr.Error.Message) {
		base = llm.ErrContextOverflow
	}

	return &llm.ProviderError{
		Base:      base,
		Code:      sseErr.Error.Type,
		Message:   sseErr.Error.Message,
		Retryable: retryable,
	}
}

// sseErrorPayload represents the JSON payload of an SSE error event. The HTTP
// error body uses the identical envelope.
type sseErrorPayload struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// classifySSEErrorType maps Anthropic SSE error type strings to sentinel errors.
func classifySSEErrorType(errType string) (bool, error) {
	switch errType {
	case "api_error", "overloaded_error":
		return true, llm.ErrServerError
	case "rate_limit_error":
		return true, llm.ErrRateLimitExceeded
	case "invalid_request_error":
		return false, llm.ErrInvalidInput
	case "authentication_error", "permission_error":
		return false, llm.ErrAPICall
	case "not_found_error":
		return false, llm.ErrAPICall
	default:
		// Unknown SSE error type — default to server error (retryable)
		return true, llm.ErrServerError
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

// statusCodeToString returns a human-readable string for an HTTP status code.
func statusCodeToString(code int) string {
	switch code {
	case 400:
		return "bad_request"
	case 401:
		return "unauthorized"
	case 403:
		return "forbidden"
	case 429:
		return "rate_limit_exceeded"
	case 500:
		return "internal_server_error"
	case 502:
		return "bad_gateway"
	case 503:
		return "service_unavailable"
	case 529:
		return "overloaded"
	default:
		return "http_error"
	}
}
