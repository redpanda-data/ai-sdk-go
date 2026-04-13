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

import (
	"encoding/json"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/otel/genai"
)

// transformInputMessages converts SDK messages to OTel-compliant input format.
func transformInputMessages(messages []llm.Message) []genai.Message {
	result := make([]genai.Message, 0, len(messages))

	for _, msg := range messages {
		result = append(result, transformMessage(msg, ""))
	}

	return result
}

// transformOutputMessage converts SDK message to OTel-compliant output format.
func transformOutputMessage(msg llm.Message, finishReason string) genai.Message {
	return transformMessage(msg, finishReason)
}

// mapRole normalizes SDK roles to OTel enum.
// OTel spec requires: "user", "assistant", "system", "tool"
// We map user messages containing tool responses to "tool" role per spec.
func mapRole(msg llm.Message) string {
	// Tool responses should use "tool" role per OTel spec
	if msg.Role == llm.RoleUser && len(msg.ToolResponses()) > 0 {
		return "tool"
	}

	// Standard role mapping
	return string(msg.Role)
}

// mapFinishReason normalizes SDK finish reasons to OTel enum.
// OTel spec allows: "stop", "length", "content_filter", "tool_call", "error".
func mapFinishReason(fr llm.FinishReason) string {
	switch fr {
	case llm.FinishReasonStop:
		return "stop"
	case llm.FinishReasonLength:
		return "length"
	case llm.FinishReasonContentFilter:
		return "content_filter"
	case llm.FinishReasonToolCalls:
		return "tool_call" // Map plural to singular per OTel spec
	case llm.FinishReasonInterrupted, llm.FinishReasonUnknown:
		return "error" // Map non-standard reasons to "error"
	default:
		return "error" // Fallback for unknown values
	}
}

// transformMessage converts a single SDK message to OTel format.
func transformMessage(msg llm.Message, finishReason string) genai.Message {
	otelMsg := genai.Message{
		Role:  mapRole(msg),
		Parts: make([]genai.Part, 0, len(msg.Content)),
	}

	// Add finish_reason for output messages if provided
	if finishReason != "" {
		otelMsg.FinishReason = finishReason
	}

	// Transform each part
	for _, part := range msg.Content {
		if part == nil {
			continue
		}

		otelMsg.Parts = append(otelMsg.Parts, transformPart(part))
	}

	return otelMsg
}

// transformPart converts a SDK Part to OTel-compliant format.
func transformPart(part *llm.Part) genai.Part {
	switch part.Kind {
	case llm.PartText:
		return genai.Part{
			Type:    genai.PartTypeText,
			Content: part.Text,
		}

	case llm.PartToolRequest:
		if part.ToolRequest == nil {
			// OTel JSON schema requires "name" field for tool_call parts
			return genai.Part{
				Type: genai.PartTypeToolCall,
				Name: "unknown",
			}
		}

		return genai.Part{
			Type:      genai.PartTypeToolCall,
			Name:      part.ToolRequest.Name,
			ID:        part.ToolRequest.ID,
			Arguments: part.ToolRequest.Arguments,
		}

	case llm.PartToolResponse:
		// OTel JSON schema requires "response" field to always be present.
		// Default to null to satisfy this requirement.
		response := json.RawMessage("null")

		if part.ToolResponse == nil {
			return genai.Part{
				Type:     genai.PartTypeToolCallResponse,
				Response: response,
			}
		}

		// Convert the result to response field
		// If there's an error, we should include it in the response
		if part.ToolResponse.Error != "" {
			// Create an error response structure
			errorResp := map[string]string{"error": part.ToolResponse.Error}

			if b, err := json.Marshal(errorResp); err == nil {
				response = b
			}
		} else if len(part.ToolResponse.Result) > 0 {
			response = part.ToolResponse.Result
		}
		// else: keep default null value

		return genai.Part{
			Type:     genai.PartTypeToolCallResponse,
			ID:       part.ToolResponse.ID,
			Response: response,
		}

	case llm.PartReasoning:
		if part.ReasoningTrace == nil {
			return genai.Part{Type: genai.PartTypeReasoning}
		}

		return genai.Part{
			Type:    genai.PartTypeReasoning,
			Content: part.ReasoningTrace.Text,
		}

	default:
		// Unknown part type - create a generic text representation
		return genai.Part{
			Type:    genai.PartTypeText,
			Content: "",
		}
	}
}
