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
