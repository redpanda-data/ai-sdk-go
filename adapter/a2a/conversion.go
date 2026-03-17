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

// Package a2a provides utilities for converting between A2A protocol messages
// and the AI SDK's LLM message format.
//
// # Conversion Architecture
//
// This package provides conversion at the protocol adapter boundary:
//   - Incoming A2A messages → LLM format (for SDK processing)
//   - Outgoing LLM responses → A2A format (for protocol responses)
//
// State tracking is expected to be done internally via the session store using
// llm.Message format, NOT through A2A message history. The conversions are
// one-directional at each boundary and do NOT support lossless round-trip
// conversion (A2A → LLM → A2A).
//
// # Type Discrimination
//
// DataPart uses Metadata["data_type"] to identify the LLM part type:
//   - "tool_request" - Contains llm.ToolRequest
//   - "tool_response" - Contains llm.ToolResponse
//
// # Serialization Safety
//
// All complex types stored in A2A Metadata are converted to JSON-safe
// map[string]any to ensure compatibility with A2A's gob-based task state
// persistence. Only these types are allowed in Metadata:
// nil, bool, int, float, string, []any, map[string]any.
package a2a

import (
	"encoding/json"
	"fmt"

	"github.com/a2aproject/a2a-go/a2a"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// toJSONSafe converts a value to a JSON-safe map[string]any representation.
// This ensures compatibility with A2A's task state persistence (gob encoding)
// which only supports: nil, bool, int, float, string, []any, map[string]any.
func toJSONSafe(v any) (map[string]any, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal to JSON: %w", err)
	}

	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal to map: %w", err)
	}

	return result, nil
}

// MessageToLLM converts an A2A message to LLM SDK message format.
//
// This is used at the ingress boundary when A2A client messages enter the SDK.
// The resulting llm.Message is passed to the runner and stored in the session.
//
// Conversion mapping:
//   - a2a.TextPart → llm.PartText (text content)
//   - a2a.DataPart → llm.PartText (JSON-serialized data)
//   - a2a.FilePart → llm.PartText (JSON-serialized file metadata and content)
//
// DataPart and FilePart are converted to JSON strings since the LLM SDK does not
// have native support for structured data or file parts yet. This preserves all
// information while maintaining compatibility.
//
// Note: This conversion is one-directional. State tracking should use the session
// store's llm.Message format, not A2A message history.
func MessageToLLM(msg *a2a.Message) llm.Message {
	var role llm.MessageRole

	switch msg.Role {
	case a2a.MessageRoleUser:
		role = llm.RoleUser
	case a2a.MessageRoleAgent:
		role = llm.RoleAssistant
	case a2a.MessageRoleUnspecified:
		role = llm.RoleAssistant
	default:
		role = llm.RoleAssistant
	}

	parts := make([]*llm.Part, 0, len(msg.Parts))
	for _, part := range msg.Parts {
		switch p := part.(type) {
		case a2a.TextPart:
			// TextPart → Text part
			parts = append(parts, llm.NewTextPart(p.Text))

		case a2a.DataPart, a2a.FilePart:
			// DataPart/FilePart → JSON text (LLM doesn't have native support for these)
			jsonBytes, err := json.Marshal(p)
			if err != nil {
				// Fallback: encode error as text to preserve visibility
				errMsg := fmt.Sprintf("[ERROR: Failed to serialize part: %v]", err)
				parts = append(parts, llm.NewTextPart(errMsg))
			} else {
				parts = append(parts, llm.NewTextPart(string(jsonBytes)))
			}
		}
	}

	return llm.NewMessage(role, parts...)
}

// MessageFromLLM converts an LLM SDK message to A2A message format.
//
// This is used at the egress boundary when LLM responses exit the SDK to A2A clients.
//
// Conversion mapping:
//   - llm.PartText → a2a.TextPart
//   - llm.PartToolRequest → a2a.DataPart with Metadata["data_type"]="tool_request"
//   - llm.PartToolResponse → a2a.DataPart with Metadata["data_type"]="tool_response"
//   - llm.PartReasoning → a2a.TextPart (reasoning text only, type information lost)
//
// All complex types (ToolRequest, ToolResponse) are converted to JSON-safe map[string]any
// for gob-compatibility. The data_type metadata field enables reverse conversion.
func MessageFromLLM(llmMsg llm.Message) *a2a.Message {
	// Map role
	var role a2a.MessageRole

	switch llmMsg.Role {
	case llm.RoleUser:
		role = a2a.MessageRoleUser
	case llm.RoleAssistant:
		role = a2a.MessageRoleAgent
	case llm.RoleSystem:
		role = a2a.MessageRoleAgent // System messages become agent messages
	default:
		role = a2a.MessageRoleUser // fallback
	}

	// Convert parts
	parts := make([]a2a.Part, 0, len(llmMsg.Content))
	for _, part := range llmMsg.Content {
		switch {
		case part.IsText():
			// Text part: store directly as TextPart
			parts = append(parts, a2a.TextPart{Text: part.Text})

		case part.IsToolRequest() && part.ToolRequest != nil:
			// Tool request: convert to JSON-safe DataPart with data_type metadata
			data, err := toJSONSafe(part.ToolRequest)
			if err != nil {
				// Fallback: encode error as text to preserve visibility
				errMsg := fmt.Sprintf("[ERROR: Failed to serialize tool request '%s': %v]",
					part.ToolRequest.Name, err)
				parts = append(parts, a2a.TextPart{Text: errMsg})
			} else {
				parts = append(parts, a2a.DataPart{
					Data: data,
					Metadata: map[string]any{
						"data_type": "tool_request",
					},
				})
			}
		case part.IsToolResponse() && part.ToolResponse != nil:
			// Tool response: convert to JSON-safe DataPart with data_type metadata
			data, err := toJSONSafe(part.ToolResponse)
			if err != nil {
				// Fallback: encode error as text to preserve visibility
				errMsg := fmt.Sprintf("[ERROR: Failed to serialize tool response for '%s': %v]",
					part.ToolResponse.Name, err)
				parts = append(parts, a2a.TextPart{Text: errMsg})
			} else {
				parts = append(parts, a2a.DataPart{
					Data: data,
					Metadata: map[string]any{
						"data_type": "tool_response",
					},
				})
			}
		case part.IsReasoning() && part.ReasoningTrace != nil:
			// Reasoning trace: store text as TextPart (like regular text)
			parts = append(parts, a2a.TextPart{Text: part.ReasoningTrace.Text})
		}
	}

	return a2a.NewMessage(role, parts...)
}
