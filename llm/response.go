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

package llm

// Response represents a standardized response from any AI model.
// All providers map their responses to this unified structure.
type Response struct {
	// Message contains the assistant's response with role and content.
	// The role is always RoleAssistant. This allows responses to be
	// directly reused in follow-up requests without conversion.
	Message Message `json:"message"`

	// Usage provides token consumption statistics if available from the provider.
	// Some providers may not support usage tracking.
	Usage *TokenUsage `json:"usage,omitempty"`

	// FinishReason indicates why the generation stopped.
	// Common values: "stop", "length", "tool_calls", "content_filter"
	FinishReason FinishReason `json:"finish_reason"`

	// ID is a unique identifier for this response, useful for tracing and debugging.
	// The format and availability depends on the provider.
	ID string `json:"id,omitempty"`

	// Metadata provides additional context carried over from the request.
	// This enables request-response correlation for tracing and debugging.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Raw contains the original provider response for debugging purposes.
	// This is optional and may be omitted in production to save memory.
	Raw map[string]any `json:"raw,omitempty"`

	// ServiceTier is the normalized request-processing variant the provider
	// reported for this call. Empty string means default or not reported.
	// This field describes HOW the request was processed and is not
	// meaningfully additive across calls — it lives on Response, not
	// TokenUsage.
	ServiceTier ServiceTier `json:"service_tier,omitempty"`

	// RawServiceTier carries the provider-native tier string for audit/debug
	// when the normalized mapping is lossy (e.g., vendor-specific values the
	// SDK doesn't recognize).
	RawServiceTier string `json:"raw_service_tier,omitempty"`

	// Speed is a provider-specific latency tier. Anthropic:
	// "standard" | "fast". Bedrock PerformanceConfig:
	// "standard" | "optimized". Empty for providers that do not report one.
	Speed string `json:"speed,omitempty"`

	// InferenceRegion is the compute region reported by the provider, if any.
	// Anthropic populates this from inference_geo. Other providers typically
	// leave it empty.
	InferenceRegion string `json:"inference_region,omitempty"`

	// InvokedModelID is the actual model that served the request when a
	// router rewrote it (Bedrock PromptRouter, OpenAI model routing). Empty
	// when no re-routing occurred. Pricing must be computed against the
	// invoked model, not the requested one.
	InvokedModelID string `json:"invoked_model_id,omitempty"`
}

// TextContent extracts and combines all text content from this response.
// Non-text parts are ignored. This is a convenience method that delegates
// to the underlying Message.
func (r *Response) TextContent() string {
	return r.Message.TextContent()
}

// ToolRequests extracts all tool requests from this response.
// This is a convenience method that delegates to the underlying Message.
func (r *Response) ToolRequests() []*ToolRequest {
	return r.Message.ToolRequests()
}

// HasToolRequests returns true if this response contains any tool requests.
// This is a convenience method that delegates to the underlying Message.
func (r *Response) HasToolRequests() bool {
	return r.Message.HasToolRequests()
}

// ToolResponses extracts all tool responses from this response.
// This is a convenience method that delegates to the underlying Message.
func (r *Response) ToolResponses() []*ToolResponse {
	return r.Message.ToolResponses()
}

// FilterParts returns all parts of the specified kind from this response.
// This is a convenience method that delegates to the underlying Message.
func (r *Response) FilterParts(kind PartKind) []*Part {
	return r.Message.FilterParts(kind)
}
