package otel

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// InterceptToolExecution creates a "gen_ai.tool" span wrapping tool calls.
func (t *TracingInterceptor) InterceptToolExecution(
	ctx context.Context,
	info *agent.ToolCallInfo,
	next agent.ToolExecutionNext,
) (*llm.ToolResponse, error) {
	req := info.Req

	// Build span name following OTel convention: "execute_tool {gen_ai.tool.name}"
	spanName := "execute_tool " + req.Name

	// Build base attributes
	attrs := []attribute.KeyValue{
		GenAIOperationName(OperationToolCall),
		GenAIToolName(req.Name),
		GenAIToolCallID(req.ID),
	}

	// Add conversation ID if session is available
	if session := info.Inv.Session(); session != nil && session.ID != "" {
		attrs = append(attrs, GenAIConversationID(session.ID))
	}

	// Call attribute injector if configured (before span creation for sampling)
	if t.cfg.attributeInjector != nil {
		sessionID := ""
		if session := info.Inv.Session(); session != nil {
			sessionID = session.ID
		}

		spanCtx := SpanContext{
			SpanType:  SpanTypeTool,
			SpanName:  spanName,
			SessionID: sessionID,
			Inv:       info.Inv,
		}
		if customAttrs := t.cfg.attributeInjector(ctx, spanCtx); len(customAttrs) > 0 {
			attrs = append(attrs, customAttrs...)
		}
	}

	// Start tool span as child of current context (invocation span)
	ctx, span := t.tracer.Start(ctx, spanName,
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attrs...),
	)
	defer span.End()

	// Measure and record argument size
	if len(req.Arguments) > 0 {
		span.SetAttributes(ToolArgumentsSize(len(req.Arguments)))
	}

	// Optionally record tool arguments as span attribute (opt-in - may contain PII)
	if t.cfg.recordInputs && len(req.Arguments) > 0 {
		// Validate JSON is structured object
		if isValidStructuredJSON(req.Arguments) {
			span.SetAttributes(GenAIToolCallArguments(string(req.Arguments)))
		}
	}

	// Track execution start time
	startTime := time.Now()

	// Execute tool
	resp, err := next(ctx, info)

	// Calculate and record execution duration (metadata - no PII)
	duration := time.Since(startTime)
	span.SetAttributes(ToolExecutionDuration(duration.Milliseconds()))

	// Record errors
	//nolint:nestif // Complex result processing logic
	if err != nil {
		setSpanError(span, err)
		span.SetAttributes(ToolResultAvailable(false))
	} else {
		// Record result availability and size (metadata - no PII)
		resultAvailable := resp != nil && resp.Result != nil
		span.SetAttributes(ToolResultAvailable(resultAvailable))

		if resultAvailable {
			span.SetAttributes(ToolResultSize(len(resp.Result)))

			// Optionally record tool output as span attribute (opt-in - may contain PII)
			if t.cfg.recordOutputs {
				// Validate JSON is structured object
				if isValidStructuredJSON(resp.Result) {
					span.SetAttributes(GenAIToolCallResult(string(resp.Result)))
				}
			}
		}
	}

	return resp, err
}
