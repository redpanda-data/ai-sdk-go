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

	// Record errors and results.
	//
	// Three outcomes:
	//   1. Go error (err != nil): infrastructure failure — set span error from the Go error
	//   2. Tool error (resp.Error != ""): tool returned error content — set error.type = "tool_error"
	//      per MCP semconv (isError=true). Do NOT record gen_ai.tool.call.result (spec: "if successful").
	//   3. Success: record result availability/size, optionally record gen_ai.tool.call.result
	//

	switch {
	case err != nil:
		// Case 1: Go error — infrastructure/transport failure
		setSpanError(span, err)
		span.SetAttributes(toolResultAvailable(false))
	case resp != nil && resp.Error != "":
		// Case 2: Tool returned error content (analogous to MCP isError=true).
		// Per OTel MCP semconv: error.type SHOULD be "tool_error" and span status SHOULD be Error.
		// gen_ai.tool.call.result is NOT recorded (spec: "if execution was successful").
		setToolError(span, resp.Error)
		span.SetAttributes(toolResultAvailable(false))
	default:
		// Case 3: Success
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
