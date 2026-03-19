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

package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Request-side native types for JSON unmarshaling. Prefixed with "nativeReq"
// to avoid collision with the response-side native types in native_response.go.

type nativeReqBody struct {
	Model         string              `json:"model"`
	Messages      []nativeReqMessage  `json:"messages"`
	System        json.RawMessage     `json:"system,omitempty"`
	MaxTokens     int                 `json:"max_tokens"`
	Temperature   *float64            `json:"temperature,omitempty"`
	TopP          *float64            `json:"top_p,omitempty"`
	TopK          *int                `json:"top_k,omitempty"`
	StopSequences []string            `json:"stop_sequences,omitempty"`
	Tools         []nativeReqTool     `json:"tools,omitempty"`
	ToolChoice    json.RawMessage     `json:"tool_choice,omitempty"`
	Stream        bool                `json:"stream,omitempty"`
}

type nativeReqMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type nativeReqContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // tool_result content: string or array
	IsError   bool            `json:"is_error,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
}

type nativeReqTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type nativeReqToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type nativeReqSystemBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// FromNative parses a raw Anthropic /v1/messages request JSON body into
// a unified llm.Request and extracts the model name.
func (rm *RequestMapper) FromNative(body []byte) (*llm.Request, string, error) {
	var nr nativeReqBody
	if err := json.Unmarshal(body, &nr); err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal native request: %w", err)
	}

	req := &llm.Request{}

	// Parse system message
	if len(nr.System) > 0 {
		systemMsg, err := parseSystemField(nr.System)
		if err != nil {
			return nil, "", fmt.Errorf("failed to parse system field: %w", err)
		}
		if systemMsg != nil {
			req.Messages = append(req.Messages, *systemMsg)
		}
	}

	// Parse messages
	for i, msg := range nr.Messages {
		parsed, err := parseNativeReqMessage(msg)
		if err != nil {
			return nil, "", fmt.Errorf("failed to parse message %d: %w", i, err)
		}
		req.Messages = append(req.Messages, *parsed)
	}

	// Parse tools
	for _, tool := range nr.Tools {
		req.Tools = append(req.Tools, llm.ToolDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.InputSchema,
		})
	}

	// Parse tool choice
	if len(nr.ToolChoice) > 0 {
		tc, err := parseNativeReqToolChoice(nr.ToolChoice)
		if err != nil {
			return nil, "", fmt.Errorf("failed to parse tool_choice: %w", err)
		}
		req.ToolChoice = tc
	}

	return req, nr.Model, nil
}

// parseSystemField handles the Anthropic system field which can be either
// a plain string or an array of {"type":"text","text":"..."} blocks.
func parseSystemField(raw json.RawMessage) (*llm.Message, error) {
	// Try string first
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return nil, nil
		}
		msg := llm.NewMessage(llm.RoleSystem, llm.NewTextPart(s))
		return &msg, nil
	}

	// Try array of text blocks
	var blocks []nativeReqSystemBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, fmt.Errorf("system field is neither a string nor an array of text blocks: %w", err)
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	var parts []*llm.Part
	for _, b := range blocks {
		parts = append(parts, llm.NewTextPart(b.Text))
	}

	msg := llm.NewMessage(llm.RoleSystem, parts...)
	return &msg, nil
}

// parseNativeReqMessage converts a native Anthropic request message to a unified llm.Message.
func parseNativeReqMessage(msg nativeReqMessage) (*llm.Message, error) {
	var role llm.MessageRole
	switch msg.Role {
	case "user":
		role = llm.RoleUser
	case "assistant":
		role = llm.RoleAssistant
	default:
		return nil, fmt.Errorf("unsupported message role: %s", msg.Role)
	}

	// Content can be a plain string or an array of content blocks.
	// Try string first.
	var text string
	if err := json.Unmarshal(msg.Content, &text); err == nil {
		result := llm.NewMessage(role, llm.NewTextPart(text))
		return &result, nil
	}

	// Parse as array of content blocks
	var blocks []nativeReqContentBlock
	if err := json.Unmarshal(msg.Content, &blocks); err != nil {
		return nil, fmt.Errorf("content is neither a string nor an array: %w", err)
	}

	var parts []*llm.Part
	for _, block := range blocks {
		part, err := parseReqContentBlock(block)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}

	result := llm.NewMessage(role, parts...)
	return &result, nil
}

// parseReqContentBlock converts a single Anthropic request content block to an llm.Part.
func parseReqContentBlock(block nativeReqContentBlock) (*llm.Part, error) {
	switch block.Type {
	case "text":
		return llm.NewTextPart(block.Text), nil

	case "tool_use":
		args := block.Input
		if len(args) == 0 || string(args) == "null" {
			args = json.RawMessage(`{}`)
		}
		return llm.NewToolRequestPart(&llm.ToolRequest{
			ID:        block.ID,
			Name:      block.Name,
			Arguments: args,
		}), nil

	case "tool_result":
		resp := &llm.ToolResponse{
			ID: block.ToolUseID,
		}
		if block.IsError {
			resp.Error = extractToolResultText(block.Content)
		} else {
			resp.Result = json.RawMessage(extractToolResultText(block.Content))
		}
		return llm.NewToolResponsePart(resp), nil

	case "thinking":
		return llm.NewReasoningPart(&llm.ReasoningTrace{
			Text: block.Thinking,
		}), nil

	default:
		return nil, fmt.Errorf("unsupported content block type: %s", block.Type)
	}
}

// extractToolResultText extracts text from a tool_result content field,
// which can be a plain string or an array of content blocks.
func extractToolResultText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	// Try string
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}

	// Try array of content blocks (extract text from them)
	var blocks []nativeReqContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return string(raw)
	}

	var text string
	for _, b := range blocks {
		if b.Type == "text" {
			text += b.Text
		}
	}
	return text
}

// parseNativeReqToolChoice converts Anthropic tool_choice JSON to llm.ToolChoice.
func parseNativeReqToolChoice(raw json.RawMessage) (*llm.ToolChoice, error) {
	var tc nativeReqToolChoice
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil, fmt.Errorf("failed to unmarshal tool_choice: %w", err)
	}

	switch tc.Type {
	case "auto":
		return &llm.ToolChoice{Type: llm.ToolChoiceAuto}, nil
	case "any":
		return &llm.ToolChoice{Type: llm.ToolChoiceRequired}, nil
	case "tool":
		name := tc.Name
		return &llm.ToolChoice{Type: llm.ToolChoiceSpecific, Name: &name}, nil
	default:
		return nil, fmt.Errorf("unsupported tool_choice type: %s", tc.Type)
	}
}
