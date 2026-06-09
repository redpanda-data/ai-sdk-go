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
	"github.com/redpanda-data/ai-sdk-go/plugins/otel/genai"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// InterceptToolExecution creates a "gen_ai.tool" span wrapping tool calls.
func (t *TracingInterceptor) InterceptToolExecution(
	ctx context.Context,
	info *agent.ToolCallInfo,
	next agent.ToolExecutionNext,
) (tool.Execution, error) {
	req := info.Req

	// Build span name following OTel convention: "execute_tool {gen_ai.tool.name}"
	spanName := "execute_tool " + req.Name

	// Build base attributes
	attrs := []attribute.KeyValue{
		genAIOperationName(genai.OperationToolCall),
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
	exec, err := next(ctx, info)

	// Calculate and record execution duration (metadata - no PII)
	duration := time.Since(startTime)
	span.SetAttributes(toolExecutionDuration(duration.Milliseconds()))

	// Record errors and results.
	//
	// Four outcomes:
	//   1. Go error: infrastructure failure — set span error from the Go error.
	//   2. Tool pause (exec.Await != nil): non-terminal — record placeholder
	//      output info and emit an "await" event; do not finalize a result.
	//   3. Empty output (no error, no pause): success but no payload.
	//   4. Output present: success — record size and optionally the result.
	switch {
	case err != nil:
		// Tool errors from the inner executor are model-visible (the
		// registry encodes them as IsError responses), not infrastructure
		// failures, so use the MCP-compatible "tool_error" type rather
		// than the Go concrete error type.
		setToolError(span, err.Error())
		span.SetAttributes(toolResultAvailable(false))
	case exec.Await != nil:
		span.SetAttributes(
			toolResultAvailable(false),
			genAIToolCallResult("<paused: "+string(exec.Await.Reason)+">"),
		)
	default:
		resultAvailable := len(exec.Output) > 0
		span.SetAttributes(toolResultAvailable(resultAvailable))

		if resultAvailable {
			span.SetAttributes(toolResultSize(len(exec.Output)))

			if t.cfg.recordOutputs && isValidStructuredJSON(exec.Output) {
				span.SetAttributes(genAIToolCallResult(string(exec.Output)))
			}
		}
	}

	return exec, err
}
