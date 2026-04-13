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

// Package genai provides exported OpenTelemetry Gen AI semantic convention
// constants, message types, and helpers for stamping spans.
//
// This package is intentionally decoupled from the agent interceptor in
// plugins/otel so that proxies (like the AI Gateway) can produce identical
// spans without importing the full agent machinery.
//
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
package genai

import (
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// Attribute keys following OpenTelemetry Gen AI semantic conventions.
const (
	AttrGenAIOperationName             = "gen_ai.operation.name"
	AttrGenAIProviderName              = "gen_ai.provider.name"
	AttrGenAIAgentName                 = "gen_ai.agent.name"
	AttrGenAIAgentDescription          = "gen_ai.agent.description"
	AttrGenAIAgentID                   = "gen_ai.agent.id"
	AttrGenAIAgentVersion              = "gen_ai.agent.version"
	AttrGenAIConversationID            = "gen_ai.conversation.id"
	AttrGenAISystemInstructions        = "gen_ai.system_instructions"
	AttrGenAIRequestModel              = "gen_ai.request.model"
	AttrGenAIResponseID                = "gen_ai.response.id"
	AttrGenAIResponseFinishReasons     = "gen_ai.response.finish_reasons"
	AttrGenAIUsageInputTokens          = "gen_ai.usage.input_tokens"            //nolint:gosec // Not a credential
	AttrGenAIUsageOutputTokens         = "gen_ai.usage.output_tokens"           //nolint:gosec // Not a credential
	AttrGenAIUsageCacheReadInputTokens = "gen_ai.usage.cache_read.input_tokens" //nolint:gosec // Not a credential
	AttrGenAIInputMessages             = "gen_ai.input.messages"
	AttrGenAIOutputMessages            = "gen_ai.output.messages"
	AttrGenAIToolDefinitions           = "gen_ai.tool.definitions"
	AttrGenAIToolName                  = "gen_ai.tool.name"
	AttrGenAIToolCallID                = "gen_ai.tool.call.id"
	AttrGenAIToolCallArguments         = "gen_ai.tool.call.arguments"
	AttrGenAIToolCallResult            = "gen_ai.tool.call.result"
	AttrGenAIToolType                  = "gen_ai.tool.type"
	AttrGenAIToolDescription           = "gen_ai.tool.description"
)

// Operation name constants.
const (
	OperationInvokeAgent = "invoke_agent"
	OperationChat        = "chat"
	OperationToolCall    = "execute_tool"
)

// ModelCallAttrs holds fields extracted from an LLM request/response pair.
// Build this from whatever source you have (gjson, struct, etc.) and pass
// it to StampModelCallSpan to produce spec-compliant span attributes.
type ModelCallAttrs struct {
	// Provider is the LLM provider name (e.g. "openai", "anthropic", "google", "bedrock").
	Provider string

	// RequestModel is the model name from the request body.
	RequestModel string

	// ResponseModel is the model name from the response body (may differ from request).
	ResponseModel string

	// ResponseID is the provider's response identifier.
	ResponseID string

	// FinishReason is the normalized finish reason (use OTel enum values:
	// "stop", "length", "content_filter", "tool_call", "error").
	FinishReason string

	// Token usage.
	InputTokens  int
	OutputTokens int
	CachedTokens int
}

// StampModelCallSpan sets gen_ai.* attributes on an existing span.
// The output matches plugins/otel model.go exactly so that spans from
// the AI Gateway are indistinguishable from spans produced by ai-sdk-go.
func StampModelCallSpan(span trace.Span, a *ModelCallAttrs) {
	if a == nil {
		return
	}

	// OTel Gen AI semconv has no gen_ai.response.model attribute.
	// Fall back to ResponseModel when RequestModel is empty so that
	// gen_ai.request.model is always populated when possible.
	model := a.RequestModel
	if model == "" {
		model = a.ResponseModel
	}

	attrs := []attribute.KeyValue{
		attribute.String(AttrGenAIOperationName, OperationChat),
	}

	if model != "" {
		attrs = append(attrs, attribute.String(AttrGenAIRequestModel, model))
	}

	if a.Provider != "" {
		attrs = append(attrs, attribute.String(AttrGenAIProviderName, a.Provider))
	}

	if a.ResponseID != "" {
		attrs = append(attrs, attribute.String(AttrGenAIResponseID, a.ResponseID))
	}

	if a.FinishReason != "" {
		attrs = append(attrs, attribute.StringSlice(AttrGenAIResponseFinishReasons, []string{a.FinishReason}))
	}

	attrs = append(attrs,
		attribute.Int(AttrGenAIUsageInputTokens, a.InputTokens),
		attribute.Int(AttrGenAIUsageOutputTokens, a.OutputTokens),
	)

	if a.CachedTokens > 0 {
		attrs = append(attrs, attribute.Int(AttrGenAIUsageCacheReadInputTokens, a.CachedTokens))
	}

	span.SetAttributes(attrs...)
}

// SpanName returns the canonical span name for a model call: "chat {model}".
// If model is empty, returns just "chat".
func SpanName(model string) string {
	if model == "" {
		return OperationChat
	}

	return OperationChat + " " + model
}
