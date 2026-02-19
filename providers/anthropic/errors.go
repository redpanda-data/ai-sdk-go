package anthropic

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// classifyError maps Anthropic SDK errors to *llm.ProviderError with the
// appropriate sentinel base and Retryable flag.
//
// Two error sources are handled:
//   - HTTP errors: anthropic.Error with StatusCode
//   - SSE streaming errors: fmt.Errorf("received error while streaming: %s", json)
//     from the SDK's ssestream package
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	// Try HTTP API error first
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
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

	return &llm.ProviderError{
		Base:      base,
		Code:      statusCodeToString(apiErr.StatusCode),
		Message:   apiErr.Error(),
		Retryable: retryable,
	}
}

// classifySSEError parses the SDK's SSE streaming error format and classifies it.
// The format is: "received error while streaming: {json}"
// where JSON is: {"type":"error","error":{"type":"<error_type>","message":"<msg>"}}.
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

	return &llm.ProviderError{
		Base:      base,
		Code:      sseErr.Error.Type,
		Message:   sseErr.Error.Message,
		Retryable: retryable,
	}
}

// sseErrorPayload represents the JSON payload of an SSE error event.
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
