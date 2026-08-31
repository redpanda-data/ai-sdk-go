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

	// ServiceTier is the provider-reported processing tier for this response.
	// Known tiers normalize to the ServiceTier* constants and can be compared
	// in switch statements. Unknown non-empty values are preserved verbatim
	// (lower-cased, trimmed, dashes to underscores) and fall through to the
	// default arm. The empty value means the provider did not report a tier
	// and is distinct from ServiceTierDefault.
	//
	// This belongs on Response rather than TokenUsage because it is request
	// metadata, not something that can be meaningfully summed across multiple
	// calls. Pricing uses it as one dimension when selecting a rate card.
	ServiceTier ServiceTier `json:"service_tier,omitempty"`

	// Speed is the provider-reported latency mode for this response.
	// Known modes normalize to the Speed* constants and can be compared
	// in switch statements. Unknown non-empty values are preserved
	// verbatim (lower-cased, trimmed, dashes to underscores) and fall
	// through to the default arm. The empty value means the provider
	// did not report a speed.
	Speed Speed `json:"speed,omitempty"`

	// InferenceRegion is the region that actually served the request when the
	// provider reports it. This matters for pricing because some providers price
	// the same model differently by region.
	InferenceRegion string `json:"inference_region,omitempty"`

	// InvokedModelID is the model identifier to use for SDK lookups after
	// any provider-side routing. The contract is uniform across providers:
	// when the catalog recognizes the provider-reported model (including
	// timestamped snapshots, via Resolve), this is the catalog offering ID,
	// so exact-ID pricing and capability lookups work directly; a reported
	// ID the catalog does not know passes through verbatim and must be
	// treated as unknown (unpriced), never approximated.
	InvokedModelID string `json:"invoked_model_id,omitempty"`
}

// TextContent extracts and combines all text content from this response.
// Non-text parts are ignored. This is a convenience method that delegates
// to the underlying Message.
func (r *Response) TextContent() string {
	return r.Message.TextContent()
}

// ToolRequests extracts all tool request parts from this response.
// This is a convenience method that delegates to the underlying Message.
func (r *Response) ToolRequests() []*ToolRequestPart {
	return r.Message.ToolRequests()
}

// HasToolRequests returns true if this response contains any tool requests.
// This is a convenience method that delegates to the underlying Message.
func (r *Response) HasToolRequests() bool {
	return r.Message.HasToolRequests()
}

// ToolResponses extracts all tool response parts from this response.
// This is a convenience method that delegates to the underlying Message.
func (r *Response) ToolResponses() []*ToolResponsePart {
	return r.Message.ToolResponses()
}
