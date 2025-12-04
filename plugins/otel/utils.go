package otel

import (
	"encoding/json"
	"fmt"

	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// extractSystemPrompt finds the first system role message in the message history.
// Returns empty string if no system message is found.
func extractSystemPrompt(messages []llm.Message) string {
	for _, msg := range messages {
		if msg.Role == llm.RoleSystem {
			return msg.TextContent()
		}
	}

	return ""
}

// isValidStructuredJSON checks if the given bytes contain valid JSON that represents
// a structured object (object or array), not a primitive value.
// Returns true if the JSON is valid and structured, false otherwise.
func isValidStructuredJSON(data []byte) bool {
	if len(data) == 0 {
		return false
	}

	// Fast validation without full parse
	if !json.Valid(data) {
		return false
	}

	// Check first non-whitespace character to determine if it's structured
	for _, b := range data {
		switch b {
		case ' ', '\t', '\n', '\r':
			continue
		case '{', '[':
			return true
		default:
			return false
		}
	}

	return false
}

// setSpanError records an error on a span with concrete error type information.
func setSpanError(span trace.Span, err error) {
	if err == nil {
		return
	}

	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
	// Use concrete error type for better debugging (e.g., "*errors.errorString")
	span.SetAttributes(errorType(fmt.Sprintf("%T", err)))
}
