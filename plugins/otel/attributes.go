package otel

// This file defines OpenTelemetry attribute keys and helper functions following
// the Gen AI semantic conventions.
//
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/

import (
	"context"

	"go.opentelemetry.io/otel/attribute"

	"github.com/redpanda-data/ai-sdk-go/agent"
)

// Attribute keys following OpenTelemetry Gen AI semantic conventions.
// These constants are used internally by the plugin to populate standard attributes.
// External users should use AttributeInjector for custom attributes only.
//
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
const (
	attrGenAIOperationName         = "gen_ai.operation.name"
	attrGenAIProviderName          = "gen_ai.provider.name"
	attrGenAIAgentName             = "gen_ai.agent.name"
	attrGenAIAgentDescription      = "gen_ai.agent.description"
	attrGenAIConversationID        = "gen_ai.conversation.id"
	attrGenAISystemInstructions    = "gen_ai.system_instructions"
	attrGenAIRequestModel          = "gen_ai.request.model"
	attrGenAIResponseID            = "gen_ai.response.id"
	attrGenAIResponseFinishReasons = "gen_ai.response.finish_reasons"
	attrGenAIUsageInputTokens      = "gen_ai.usage.input_tokens"  //nolint:gosec // Not a credential
	attrGenAIUsageOutputTokens     = "gen_ai.usage.output_tokens" //nolint:gosec // Not a credential
	attrGenAIInputMessages         = "gen_ai.input.messages"
	attrGenAIOutputMessages        = "gen_ai.output.messages"
	attrGenAIToolDefinitions       = "gen_ai.tool.definitions"
	attrGenAIToolName              = "gen_ai.tool.name"
	attrGenAIToolCallID            = "gen_ai.tool.call.id"
	attrGenAIToolCallArguments     = "gen_ai.tool.call.arguments"
	attrGenAIToolCallResult        = "gen_ai.tool.call.result"
	attrToolArgumentsSize          = "redpanda.tool.arguments.size"
	attrToolResultSize             = "redpanda.tool.result.size"
	attrToolExecutionDuration      = "redpanda.tool.execution.duration"
	attrToolResultAvailable        = "redpanda.tool.result.available"
	attrErrorType                  = "error.type"
)

// Operation names for gen_ai.operation.name attribute.
// These are internal constants used by the plugin.
const (
	operationInvokeAgent = "invoke_agent"
	operationChat        = "chat"
	operationToolCall    = "execute_tool"
)

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
	return attribute.String(attrGenAIOperationName, op)
}

func genAIProviderName(name string) attribute.KeyValue {
	return attribute.String(attrGenAIProviderName, name)
}

func genAIAgentName(name string) attribute.KeyValue {
	return attribute.String(attrGenAIAgentName, name)
}

func genAIAgentDescription(description string) attribute.KeyValue {
	return attribute.String(attrGenAIAgentDescription, description)
}

func genAISystemInstructions(instructions string) attribute.KeyValue {
	return attribute.String(attrGenAISystemInstructions, instructions)
}

func genAIConversationID(id string) attribute.KeyValue {
	return attribute.String(attrGenAIConversationID, id)
}

func genAIRequestModel(model string) attribute.KeyValue {
	return attribute.String(attrGenAIRequestModel, model)
}

func genAIResponseID(id string) attribute.KeyValue {
	return attribute.String(attrGenAIResponseID, id)
}

func genAIResponseFinishReasons(reasons ...string) attribute.KeyValue {
	return attribute.StringSlice(attrGenAIResponseFinishReasons, reasons)
}

func genAIUsageInputTokens(tokens int) attribute.KeyValue {
	return attribute.Int(attrGenAIUsageInputTokens, tokens)
}

func genAIUsageOutputTokens(tokens int) attribute.KeyValue {
	return attribute.Int(attrGenAIUsageOutputTokens, tokens)
}

func genAIToolName(name string) attribute.KeyValue {
	return attribute.String(attrGenAIToolName, name)
}

func genAIToolCallID(id string) attribute.KeyValue {
	return attribute.String(attrGenAIToolCallID, id)
}

func genAIToolCallArguments(args string) attribute.KeyValue {
	return attribute.String(attrGenAIToolCallArguments, args)
}

func genAIToolCallResult(result string) attribute.KeyValue {
	return attribute.String(attrGenAIToolCallResult, result)
}

func genAIInputMessages(json string) attribute.KeyValue {
	return attribute.String(attrGenAIInputMessages, json)
}

func genAIOutputMessages(json string) attribute.KeyValue {
	return attribute.String(attrGenAIOutputMessages, json)
}

func genAIToolDefinitions(json string) attribute.KeyValue {
	return attribute.String(attrGenAIToolDefinitions, json)
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
