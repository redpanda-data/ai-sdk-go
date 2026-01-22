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
		genAIOperationName(operationToolCall),
		genAIToolName(req.Name),
		genAIToolCallID(req.ID),
	}

	// Add conversation ID if session is available
	if session := info.Inv.Session(); session != nil && session.ID != "" {
		attrs = append(attrs, genAIConversationID(session.ID))
	}

	// Add tool type and description if definition is available
	if info.Definition != nil {
		if info.Definition.Description != "" {
			attrs = append(attrs, genAIToolDescription(info.Definition.Description))
		}

		toolType := info.Definition.Type
		switch toolType {
		case "", llm.ToolTypeFunction:
			toolType = llm.ToolTypeFunction
		case llm.ToolTypeExtension, llm.ToolTypeDatastore:
			// Valid types - use as-is
		default:
			// Invalid type - default to function for OTel compliance
			toolType = llm.ToolTypeFunction
		}

		attrs = append(attrs, genAIToolType(toolType))
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
		span.SetAttributes(toolArgumentsSize(len(req.Arguments)))
	}

	// Optionally record tool arguments as span attribute (opt-in - may contain PII)
	if t.cfg.recordInputs && len(req.Arguments) > 0 {
		// Validate JSON is structured object
		if isValidStructuredJSON(req.Arguments) {
			span.SetAttributes(genAIToolCallArguments(string(req.Arguments)))
		}
	}

	// Track execution start time
	startTime := time.Now()

	// Execute tool
	resp, err := next(ctx, info)

	// Calculate and record execution duration (metadata - no PII)
	duration := time.Since(startTime)
	span.SetAttributes(toolExecutionDuration(duration.Milliseconds()))

	// Record errors
	//nolint:nestif // Complex result processing logic
	if err != nil {
		setSpanError(span, err)
		span.SetAttributes(toolResultAvailable(false))
	} else {
		// Record result availability and size (metadata - no PII)
		resultAvailable := resp != nil && resp.Result != nil
		span.SetAttributes(toolResultAvailable(resultAvailable))

		if resultAvailable {
			span.SetAttributes(toolResultSize(len(resp.Result)))

			// Optionally record tool output as span attribute (opt-in - may contain PII)
			if t.cfg.recordOutputs {
				// Validate JSON is structured object
				if isValidStructuredJSON(resp.Result) {
					span.SetAttributes(genAIToolCallResult(string(resp.Result)))
				}
			}
		}
	}

	return resp, err
}
