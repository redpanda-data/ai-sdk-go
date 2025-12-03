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
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
const (
	// AttrGenAIOperationName is the operation name attribute.
	AttrGenAIOperationName = "gen_ai.operation.name"
	// AttrGenAIProviderName is the provider name (gen_ai.provider.name).
	AttrGenAIProviderName = "gen_ai.provider.name"

	// AttrGenAIAgentName is the agent name.
	AttrGenAIAgentName = "gen_ai.agent.name"
	// AttrGenAIAgentDescription is the agent description.
	AttrGenAIAgentDescription = "gen_ai.agent.description"
	// AttrGenAIConversationID is the conversation/session identifier.
	AttrGenAIConversationID = "gen_ai.conversation.id"
	// AttrGenAISystemInstructions is the system instructions/prompt.
	AttrGenAISystemInstructions = "gen_ai.system_instructions"

	// AttrGenAIRequestModel is the requested model name.
	AttrGenAIRequestModel = "gen_ai.request.model"

	// AttrGenAIResponseID is the response identifier.
	AttrGenAIResponseID = "gen_ai.response.id"
	// AttrGenAIResponseFinishReasons is the finish reasons.
	AttrGenAIResponseFinishReasons = "gen_ai.response.finish_reasons"

	// AttrGenAIUsageInputTokens is the input token count.
	AttrGenAIUsageInputTokens = "gen_ai.usage.input_tokens" //nolint:gosec // Not a credential
	// AttrGenAIUsageOutputTokens is the output token count.
	AttrGenAIUsageOutputTokens = "gen_ai.usage.output_tokens" //nolint:gosec // Not a credential

	// AttrGenAIToolName is the tool name (gen_ai.tool.name).
	AttrGenAIToolName = "gen_ai.tool.name"
	// AttrGenAIToolCallID is the tool call identifier (gen_ai.tool.call.id).
	AttrGenAIToolCallID = "gen_ai.tool.call.id"
	// AttrGenAIToolCallArguments is the tool call arguments (gen_ai.tool.call.arguments).
	AttrGenAIToolCallArguments = "gen_ai.tool.call.arguments"
	// AttrGenAIToolCallResult is the tool call result (gen_ai.tool.call.result).
	AttrGenAIToolCallResult = "gen_ai.tool.call.result"

	// AttrToolArgumentsSize is the size of tool arguments in bytes (vendor attribute).
	// This is a vendor-specific attribute prefixed with "redpanda" to avoid collisions.
	AttrToolArgumentsSize = "redpanda.tool.arguments.size"
	// AttrToolResultSize is the size of tool result in bytes (vendor attribute).
	AttrToolResultSize = "redpanda.tool.result.size"
	// AttrToolExecutionDuration is the tool execution time in milliseconds (vendor attribute).
	AttrToolExecutionDuration = "redpanda.tool.execution.duration"
	// AttrToolResultAvailable indicates whether a tool result was returned (vendor attribute).
	AttrToolResultAvailable = "redpanda.tool.result.available"

	// AttrErrorType is the error type attribute (standard OTel).
	AttrErrorType = "error.type"
)

// Span name prefixes following OTel GenAI semantic conventions.
// Actual span names include additional context (e.g., "chat gpt-4o", "execute_tool get_weather").
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
const (
	// SpanNameAgent is the base span name for agent operations (actual: "invoke_agent {agent_name}").
	SpanNameAgent = "gen_ai.agent"

	// SpanNameChat is the base span name for LLM chat completion (actual: "chat {model_name}").
	SpanNameChat = "gen_ai.chat"

	// SpanNameToolCall is the base span name for tool execution (actual: "execute_tool {tool_name}").
	SpanNameToolCall = "gen_ai.tool"
)

// Operation names for gen_ai.operation.name attribute.
const (
	// OperationInvokeAgent is the operation name for agent invocation.
	OperationInvokeAgent = "invoke_agent"
	// OperationChat is the operation name for chat completion.
	OperationChat = "chat"
	// OperationToolCall is the operation name for tool execution.
	OperationToolCall = "execute_tool"
)

// Metadata keys used for span propagation via InvocationMetadata.
const (
	// MetadataKeyInvocationSpan stores the root invocation span for later retrieval.
	// We store only the span (not context) and use trace.ContextWithSpan for re-parenting.
	MetadataKeyInvocationSpan = "otel.invocation.span"
)

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

// AttributeContext provides context information for custom attribute injection.
//
// This context is passed to AttributeInjector callbacks, allowing users to add
// platform-specific or custom attributes based on the span being created.
//
// Example usage for Langfuse:
//
//	func(ctx AttributeContext) []attribute.KeyValue {
//	    if ctx.SpanType == SpanTypeInvocation {
//	        return []attribute.KeyValue{
//	            attribute.String("langfuse.trace.name", ctx.SpanName),
//	            attribute.String("langfuse.session.id", ctx.SessionID),
//	        }
//	    }
//	    return nil
//	}
type AttributeContext struct {
	// Ctx is the Go context, providing access to request-scoped values
	// (e.g., user ID, tenant ID set by middleware).
	//nolint:containedctx // Intentional: context needed for attribute injectors to access request-scoped values
	Ctx context.Context

	// SpanType identifies the category of span being created.
	SpanType SpanType

	// SpanName is the computed span name (e.g., "invoke_agent my-assistant").
	SpanName string

	// SessionID is the session/conversation identifier for this invocation.
	// This is the value of gen_ai.conversation.id.
	SessionID string

	// Inv provides full access to invocation metadata, including:
	//   - Session state and messages
	//   - Agent snapshot (name, description)
	//   - Turn number and usage statistics
	//   - Custom metadata set by other interceptors
	//
	// Note: This may be nil for some span types. Always check before accessing.
	Inv *agent.InvocationMetadata
}

// AttributeInjector is a callback function that allows users to inject custom attributes
// into OpenTelemetry spans.
//
// The injector is called before span creation, ensuring attributes are available for
// sampling decisions. It receives an AttributeContext with runtime information about
// the span being created.
//
// Injectors must be thread-safe, as they may be called concurrently when tools execute
// in parallel.
//
// Return nil or an empty slice if no attributes should be added for the given context.
type AttributeInjector func(ctx AttributeContext) []attribute.KeyValue

// Event names for model content recording following OTel GenAI conventions.
// Note: These are used for model inference operations only. Tool inputs/outputs
// are recorded as span attributes (gen_ai.tool.call.arguments, gen_ai.tool.call.result).
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
const (
	// EventGenAIContentPrompt is the event name for recording model prompt content.
	EventGenAIContentPrompt = "gen_ai.content.prompt"
	// EventGenAIContentCompletion is the event name for recording model completion content.
	EventGenAIContentCompletion = "gen_ai.content.completion"
)

// Helper functions to create typed attributes.

// GenAIOperationName creates a gen_ai.operation.name attribute.
func GenAIOperationName(op string) attribute.KeyValue {
	return attribute.String(AttrGenAIOperationName, op)
}

// GenAIProviderName creates a gen_ai.provider.name attribute.
func GenAIProviderName(name string) attribute.KeyValue {
	return attribute.String(AttrGenAIProviderName, name)
}

// GenAIAgentName creates a gen_ai.agent.name attribute.
func GenAIAgentName(name string) attribute.KeyValue {
	return attribute.String(AttrGenAIAgentName, name)
}

// GenAIAgentDescription creates a gen_ai.agent.description attribute.
func GenAIAgentDescription(description string) attribute.KeyValue {
	return attribute.String(AttrGenAIAgentDescription, description)
}

// GenAISystemInstructions creates a gen_ai.system_instructions attribute.
func GenAISystemInstructions(instructions string) attribute.KeyValue {
	return attribute.String(AttrGenAISystemInstructions, instructions)
}

// GenAIConversationID creates a gen_ai.conversation.id attribute.
func GenAIConversationID(id string) attribute.KeyValue {
	return attribute.String(AttrGenAIConversationID, id)
}

// GenAIRequestModel creates a gen_ai.request.model attribute.
func GenAIRequestModel(model string) attribute.KeyValue {
	return attribute.String(AttrGenAIRequestModel, model)
}

// GenAIResponseID creates a gen_ai.response.id attribute.
func GenAIResponseID(id string) attribute.KeyValue {
	return attribute.String(AttrGenAIResponseID, id)
}

// GenAIResponseFinishReasons creates a gen_ai.response.finish_reasons attribute.
func GenAIResponseFinishReasons(reasons ...string) attribute.KeyValue {
	return attribute.StringSlice(AttrGenAIResponseFinishReasons, reasons)
}

// GenAIUsageInputTokens creates a gen_ai.usage.input_tokens attribute.
func GenAIUsageInputTokens(tokens int) attribute.KeyValue {
	return attribute.Int(AttrGenAIUsageInputTokens, tokens)
}

// GenAIUsageOutputTokens creates a gen_ai.usage.output_tokens attribute.
func GenAIUsageOutputTokens(tokens int) attribute.KeyValue {
	return attribute.Int(AttrGenAIUsageOutputTokens, tokens)
}

// GenAIToolName creates a gen_ai.tool.name attribute.
func GenAIToolName(name string) attribute.KeyValue {
	return attribute.String(AttrGenAIToolName, name)
}

// GenAIToolCallID creates a gen_ai.tool.call_id attribute.
func GenAIToolCallID(id string) attribute.KeyValue {
	return attribute.String(AttrGenAIToolCallID, id)
}

// GenAIToolCallArguments creates a gen_ai.tool.call.arguments attribute.
func GenAIToolCallArguments(args string) attribute.KeyValue {
	return attribute.String(AttrGenAIToolCallArguments, args)
}

// GenAIToolCallResult creates a gen_ai.tool.call.result attribute.
func GenAIToolCallResult(result string) attribute.KeyValue {
	return attribute.String(AttrGenAIToolCallResult, result)
}

// ErrorType creates an error.type attribute.
func ErrorType(errType string) attribute.KeyValue {
	return attribute.String(AttrErrorType, errType)
}

// ToolArgumentsSize creates a tool.arguments.size attribute.
func ToolArgumentsSize(size int) attribute.KeyValue {
	return attribute.Int(AttrToolArgumentsSize, size)
}

// ToolResultSize creates a tool.result.size attribute.
func ToolResultSize(size int) attribute.KeyValue {
	return attribute.Int(AttrToolResultSize, size)
}

// ToolExecutionDuration creates a tool.execution.duration attribute (in milliseconds).
func ToolExecutionDuration(durationMs int64) attribute.KeyValue {
	return attribute.Int64(AttrToolExecutionDuration, durationMs)
}

// ToolResultAvailable creates a tool.result.available attribute.
func ToolResultAvailable(available bool) attribute.KeyValue {
	return attribute.Bool(AttrToolResultAvailable, available)
}
