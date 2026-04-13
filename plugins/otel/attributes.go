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

// This file defines OpenTelemetry attribute keys and helper functions following
// the Gen AI semantic conventions.
//
// Shared gen_ai.* constants live in plugins/otel/genai. This file adds
// Redpanda-specific attributes and convenience helpers for the interceptor.
//
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/

import (
	"context"

	"go.opentelemetry.io/otel/attribute"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/plugins/otel/genai"
)

// Redpanda-specific and error attributes (not part of OTel Gen AI semconv).
const (
	attrToolArgumentsSize     = "redpanda.tool.arguments.size"
	attrToolResultSize        = "redpanda.tool.result.size"
	attrToolExecutionDuration = "redpanda.tool.execution.duration"
	attrToolResultAvailable   = "redpanda.tool.result.available"
	attrErrorType             = "error.type"
)

// errorTypeToolError is the error.type value for tool-level errors
// (analogous to MCP isError=true). Per OTel MCP semconv.
const errorTypeToolError = "tool_error"

// Metadata key for span propagation via InvocationMetadata.
// This is an internal constant - users should never access this directly.
const metadataKeyInvocationSpan = "otel.invocation.span"

// SpanType identifies the category of the operation being traced.
type SpanType string

const (
	// SpanTypeInvocation represents an agent invocation span (root).
	SpanTypeInvocation SpanType = "invocation"
	// SpanTypeModel represents a model generation span (LLM call).
	SpanTypeModel SpanType = "model"
	// SpanTypeTool represents a tool execution span.
	SpanTypeTool SpanType = "tool"
)

// SpanContext provides span-specific context for attribute injection.
//
// This contains the minimal information needed to add custom attributes
// to OpenTelemetry spans before creation (important for sampling decisions).
//
// Note: This is distinct from trace.SpanContext (from the OTel SDK). This struct
// provides high-level span information for attribute injection, while trace.SpanContext
// is a low-level OTel primitive containing trace/span IDs.
//
// Example usage for Langfuse:
//
//	func(ctx context.Context, spanCtx SpanContext) []attribute.KeyValue {
//	    if spanCtx.SpanType == SpanTypeInvocation {
//	        return []attribute.KeyValue{
//	            attribute.String("langfuse.trace.name", spanCtx.SpanName),
//	            attribute.String("langfuse.session.id", spanCtx.SessionID),
//	        }
//	    }
//	    return nil
//	}
type SpanContext struct {
	// SpanType identifies the span category (invocation/model/tool).
	SpanType SpanType

	// SpanName is the computed span name (e.g., "invoke_agent my-assistant").
	SpanName string

	// SessionID is the session/conversation identifier.
	// Empty string if no session is available.
	SessionID string

	// Inv provides full invocation metadata (session, turn, usage, custom metadata).
	// This is the SDK's primary state carrier.
	//
	// May be nil for some span types - always check before accessing.
	Inv *agent.InvocationMetadata
}

// AttributeInjector is a callback that adds custom attributes to spans.
//
// Called before span creation for all span types (invocation/model/tool),
// ensuring attributes are available for sampling decisions.
//
// Must be thread-safe - tools may execute concurrently.
//
// Parameters:
//   - ctx: Go context for request-scoped values (user ID, tenant ID, etc.)
//   - spanCtx: Span-specific context (type, name, session, invocation metadata)
//
// Return nil or empty slice if no attributes should be added.
//
// Example:
//
//	func(ctx context.Context, spanCtx otel.SpanContext) []attribute.KeyValue {
//	    if spanCtx.SpanType == otel.SpanTypeInvocation {
//	        return []attribute.KeyValue{
//	            attribute.String("customer.id", getCustomerID(ctx)),
//	            attribute.String("langfuse.trace.name", spanCtx.SpanName),
//	        }
//	    }
//	    return nil
//	}
type AttributeInjector func(ctx context.Context, spanCtx SpanContext) []attribute.KeyValue

// Helper functions create typed attributes for internal use.
// These ensure correct attribute types (String vs Int vs Slice) according to OTel conventions.

func genAIOperationName(op string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIOperationName, op)
}

func genAIProviderName(name string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIProviderName, name)
}

func genAIAgentName(name string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIAgentName, name)
}

func genAIAgentDescription(description string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIAgentDescription, description)
}

func genAIAgentID(id string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIAgentID, id)
}

func genAIAgentVersion(version string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIAgentVersion, version)
}

func genAISystemInstructions(instructions string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAISystemInstructions, instructions)
}

func genAIConversationID(id string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIConversationID, id)
}

func genAIRequestModel(model string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIRequestModel, model)
}

func genAIResponseID(id string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIResponseID, id)
}

func genAIResponseFinishReasons(reasons ...string) attribute.KeyValue {
	return attribute.StringSlice(genai.AttrGenAIResponseFinishReasons, reasons)
}

func genAIUsageInputTokens(tokens int) attribute.KeyValue {
	return attribute.Int(genai.AttrGenAIUsageInputTokens, tokens)
}

func genAIUsageOutputTokens(tokens int) attribute.KeyValue {
	return attribute.Int(genai.AttrGenAIUsageOutputTokens, tokens)
}

func genAIUsageCacheReadInputTokens(tokens int) attribute.KeyValue {
	return attribute.Int(genai.AttrGenAIUsageCacheReadInputTokens, tokens)
}

func genAIToolName(name string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIToolName, name)
}

func genAIToolCallID(id string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIToolCallID, id)
}

func genAIToolCallArguments(args string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIToolCallArguments, args)
}

func genAIToolCallResult(result string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIToolCallResult, result)
}

func genAIToolType(toolType string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIToolType, toolType)
}

func genAIToolDescription(description string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIToolDescription, description)
}

func genAIInputMessages(json string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIInputMessages, json)
}

func genAIOutputMessages(json string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIOutputMessages, json)
}

func genAIToolDefinitions(json string) attribute.KeyValue {
	return attribute.String(genai.AttrGenAIToolDefinitions, json)
}

func errorType(errType string) attribute.KeyValue {
	return attribute.String(attrErrorType, errType)
}

func toolArgumentsSize(size int) attribute.KeyValue {
	return attribute.Int(attrToolArgumentsSize, size)
}

func toolResultSize(size int) attribute.KeyValue {
	return attribute.Int(attrToolResultSize, size)
}

func toolExecutionDuration(durationMs int64) attribute.KeyValue {
	return attribute.Int64(attrToolExecutionDuration, durationMs)
}

func toolResultAvailable(available bool) attribute.KeyValue {
	return attribute.Bool(attrToolResultAvailable, available)
}
