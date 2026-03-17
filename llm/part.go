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

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Part represents different types of content within messages and responses.
// This system provides extensibility while maintaining type safety.
// The Part model allows the SDK to grow from simple text to supporting
// multimedia, tool calls, and custom content types without breaking changes.
type Part struct {
	// Kind determines which fields are valid and how to interpret this Part
	Kind PartKind `json:"kind"`

	// Text contains textual content. Valid for PartText.
	Text string `json:"text,omitempty"`

	// ToolRequest contains a request from the model to execute a tool.
	// Valid for PartToolRequest.
	ToolRequest *ToolRequest `json:"tool_request,omitempty"`

	// ToolResponse contains the result of executing a tool.
	// Valid for PartToolResponse.
	ToolResponse *ToolResponse `json:"tool_response,omitempty"`

	// ReasoningTrace contains reasoning thoughts from the model.
	// Valid for PartReasoning.
	ReasoningTrace *ReasoningTrace `json:"reasoning_trace,omitempty"`

	// Metadata provides extensible key-value storage for additional information.
	// This allows for provider-specific data or future extensions without
	// breaking the core Part structure.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// PartKind defines the type of content contained in a Part.
// This enumeration enables type-safe handling while allowing future expansion.
type PartKind int8

const (
	// PartText represents plain text content.
	PartText PartKind = iota

	// PartToolRequest represents a request from the model to execute a tool.
	PartToolRequest

	// PartToolResponse represents the result of a tool execution.
	PartToolResponse

	// PartReasoning represents reasoning thoughts/traces from the model.
	PartReasoning

	// Future expansion possibilities:
	// PartImage, PartAudio, PartVideo, PartCustom.
)

// String returns a human-readable representation of the PartKind.
func (pk PartKind) String() string {
	switch pk {
	case PartText:
		return "text"
	case PartToolRequest:
		return "tool_request"
	case PartToolResponse:
		return "tool_response"
	case PartReasoning:
		return "reasoning"
	default:
		return fmt.Sprintf("unknown(%d)", pk)
	}
}

// NewTextPart creates a Part containing plain text content.
func NewTextPart(text string) *Part {
	return &Part{
		Kind: PartText,
		Text: text,
	}
}

// NewToolRequestPart creates a Part containing a tool execution request.
func NewToolRequestPart(req *ToolRequest) *Part {
	return &Part{
		Kind:        PartToolRequest,
		ToolRequest: req,
	}
}

// NewToolResponsePart creates a Part containing a tool execution result.
func NewToolResponsePart(resp *ToolResponse) *Part {
	return &Part{
		Kind:         PartToolResponse,
		ToolResponse: resp,
	}
}

// NewReasoningPart creates a Part containing reasoning traces from the model.
func NewReasoningPart(trace *ReasoningTrace) *Part {
	return &Part{
		Kind:           PartReasoning,
		ReasoningTrace: trace,
	}
}

// IsText returns true if this Part contains text content.
func (p *Part) IsText() bool {
	return p != nil && p.Kind == PartText
}

// IsToolRequest returns true if this Part contains a tool request.
func (p *Part) IsToolRequest() bool {
	return p != nil && p.Kind == PartToolRequest
}

// IsToolResponse returns true if this Part contains a tool response.
func (p *Part) IsToolResponse() bool {
	return p != nil && p.Kind == PartToolResponse
}

// IsReasoning returns true if this Part contains reasoning traces.
func (p *Part) IsReasoning() bool {
	return p != nil && p.Kind == PartReasoning
}

// JoinTextParts combines all text parts from a slice into a single string.
// Non-text parts are ignored.
func JoinTextParts(parts []*Part) string {
	var texts []string

	for _, part := range parts {
		if part.IsText() {
			texts = append(texts, part.Text)
		}
	}

	return strings.Join(texts, "")
}

// ToolRequest represents a request from the model to execute a tool.
// This corresponds to function calling in various AI models.
type ToolRequest struct {
	// ID is a unique identifier for this tool request within the conversation.
	// This allows matching requests with their corresponding responses.
	ID string `json:"id"`

	// Name is the name of the tool to execute
	Name string `json:"name"`

	// Arguments contains the tool input as JSON.
	// The structure depends on the tool's input schema.
	Arguments json.RawMessage `json:"arguments"`
}

// ToolResponse represents the result of executing a tool.
// This is sent back to the model to continue the conversation.
type ToolResponse struct {
	// ID matches the ID from the corresponding ToolRequest
	ID string `json:"id"`

	// Name is the name of the tool that was executed
	Name string `json:"name"`

	// Result contains the tool output as JSON.
	// The structure depends on the tool's output schema.
	Result json.RawMessage `json:"result"`

	// Error contains error information if the tool execution failed.
	// When non-empty, Result should be ignored.
	Error string `json:"error,omitempty"`
}

// ReasoningTrace represents reasoning thoughts/traces from the model.
// This contains the model's internal reasoning process, which may be
// exposed or summarized depending on the provider and configuration.
type ReasoningTrace struct {
	// ID is a unique identifier for this reasoning trace
	ID string `json:"id,omitempty"`

	// Text contains the reasoning content as text.
	// For streaming responses, this may be built up incrementally.
	Text string `json:"text"`

	// Metadata provides additional context about the reasoning trace.
	// This can include provider-specific information like obfuscation keys,
	// reasoning effort levels, or other debugging information.
	Metadata map[string]any `json:"metadata,omitempty"`
}
