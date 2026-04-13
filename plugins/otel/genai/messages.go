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

package genai

import (
	"encoding/json"
	"fmt"
)

// Message is the OTel Gen AI semantic convention message format.
// This is THE contract for gen_ai.input.messages and gen_ai.output.messages.
//
// All producers (ai-sdk-go agent interceptor, AI Gateway providers, etc.)
// MUST serialize messages to this format. All consumers (Console UI,
// dashboards, etc.) can rely on this schema.
//
// See:
//   - https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-input-messages.json
//   - https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-output-messages.json
type Message struct {
	Role         string `json:"role"`                    // "user", "assistant", "system", "tool"
	Parts        []Part `json:"parts"`                   // Message content parts
	FinishReason string `json:"finish_reason,omitempty"` // Output messages only
}

// Part is a single content part within a Message.
// The Type field determines which other fields are populated.
type Part struct {
	// Type discriminates the part kind.
	// Valid values: "text", "tool_call", "tool_call_response", "reasoning"
	Type string `json:"type"`

	// Content holds the text for "text" and "reasoning" parts.
	Content string `json:"content,omitempty"`

	// Name is the tool name for "tool_call" parts.
	Name string `json:"name,omitempty"`

	// ID is the tool call ID for "tool_call" and "tool_call_response" parts.
	ID string `json:"id,omitempty"`

	// Arguments is the tool call arguments JSON for "tool_call" parts.
	Arguments json.RawMessage `json:"arguments,omitempty"`

	// Response is the tool result JSON for "tool_call_response" parts.
	Response json.RawMessage `json:"response,omitempty"`
}

// Part type constants.
const (
	PartTypeText             = "text"
	PartTypeToolCall         = "tool_call"
	PartTypeToolCallResponse = "tool_call_response"
	PartTypeReasoning        = "reasoning"
)

// Role constants.
const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleSystem    = "system"
	RoleTool      = "tool"
)

// MarshalMessages serializes messages to JSON for use as span attribute values.
func MarshalMessages(msgs []Message) string {
	b, err := json.Marshal(msgs)
	if err != nil {
		return "[]"
	}
	return string(b)
}

// MarshalMessage serializes a single output message (wrapped in array) for
// use as gen_ai.output.messages attribute value.
func MarshalMessage(msg Message) string {
	return MarshalMessages([]Message{msg})
}

// ValidateMessages checks that messages conform to the OTel Gen AI schema.
// Returns an error describing the first violation found, or nil if valid.
func ValidateMessages(msgs []Message) error {
	validRoles := map[string]bool{
		RoleUser: true, RoleAssistant: true, RoleSystem: true, RoleTool: true,
	}
	validPartTypes := map[string]bool{
		PartTypeText: true, PartTypeToolCall: true,
		PartTypeToolCallResponse: true, PartTypeReasoning: true,
	}

	for i, msg := range msgs {
		if !validRoles[msg.Role] {
			return fmt.Errorf("message[%d]: invalid role %q", i, msg.Role)
		}
		if len(msg.Parts) == 0 {
			return fmt.Errorf("message[%d]: parts must not be empty", i)
		}
		for j, part := range msg.Parts {
			if !validPartTypes[part.Type] {
				return fmt.Errorf("message[%d].parts[%d]: invalid type %q", i, j, part.Type)
			}
		}
	}
	return nil
}
