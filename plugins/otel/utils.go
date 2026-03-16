package otel

import (
	"encoding/json"
	"fmt"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

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

// setToolError records a tool-level error on a span.
// This is for tools that return error content (analogous to MCP isError=true),
// as opposed to Go errors from infrastructure failures.
// Per OTel MCP semconv, error.type SHOULD be "tool_error".
//
// Unlike setSpanError, this does not call span.RecordError() because there is
// no Go error to record — the error is a string from the tool's response payload,
// not an exception. The status description carries the error message.
func setToolError(span trace.Span, errMsg string) {
	span.SetStatus(codes.Error, errMsg)
	span.SetAttributes(errorType(errorTypeToolError))
}

// setUsageAttributes sets token usage attributes on a span.
// Conditionally includes cache_read tokens when present.
func setUsageAttributes(span trace.Span, usage *llm.TokenUsage) {
	if usage == nil {
		return
	}

	attrs := []attribute.KeyValue{
		genAIUsageInputTokens(usage.InputTokens),
		genAIUsageOutputTokens(usage.OutputTokens),
	}
	if usage.CachedTokens > 0 {
		attrs = append(attrs, genAIUsageCacheReadInputTokens(usage.CachedTokens))
	}

	span.SetAttributes(attrs...)
}
