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

package openai

import (
	"encoding/json"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Request-side native types for JSON unmarshaling of OpenAI /v1/chat/completions
// requests. Prefixed with "nativeReq" to avoid collision with response-side types.

type nativeReqBody struct {
	Model            string              `json:"model"`
	Messages         []nativeReqMessage  `json:"messages"`
	Tools            []nativeReqTool     `json:"tools,omitempty"`
	ToolChoice       json.RawMessage     `json:"tool_choice,omitempty"`
	Temperature      *float64            `json:"temperature,omitempty"`
	MaxTokens        *int                `json:"max_tokens,omitempty"`
	TopP             *float64            `json:"top_p,omitempty"`
	FrequencyPenalty *float64            `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64            `json:"presence_penalty,omitempty"`
	Stop             json.RawMessage     `json:"stop,omitempty"`
	Stream           bool                `json:"stream,omitempty"`
	StreamOptions    *nativeStreamOpts   `json:"stream_options,omitempty"`
	ResponseFormat   *nativeRespFormat   `json:"response_format,omitempty"`
	Seed             *int                `json:"seed,omitempty"`
}

type nativeStreamOpts struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type nativeRespFormat struct {
	Type string `json:"type"`
}

type nativeReqMessage struct {
	Role       string              `json:"role"`
	Content    json.RawMessage     `json:"content"`
	ToolCalls  []nativeReqToolCall `json:"tool_calls,omitempty"`
	ToolCallID string              `json:"tool_call_id,omitempty"`
}

type nativeReqContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type nativeReqToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function nativeReqToolCallFn `json:"function"`
}

type nativeReqToolCallFn struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type nativeReqTool struct {
	Type     string            `json:"type"`
	Function nativeReqToolDef  `json:"function"`
}

type nativeReqToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type nativeReqToolChoiceObj struct {
	Type     string                     `json:"type"`
	Function *nativeReqToolChoiceObjFn  `json:"function,omitempty"`
}

type nativeReqToolChoiceObjFn struct {
	Name string `json:"name"`
}

// FromNative parses a raw OpenAI /v1/chat/completions request JSON body into
// a unified llm.Request and extracts the model name.
func (rm *RequestMapper) FromNative(body []byte) (*llm.Request, string, error) {
	var nr nativeReqBody
	if err := json.Unmarshal(body, &nr); err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal native request: %w", err)
	}

	req := &llm.Request{}

	// Parse messages
	for i, msg := range nr.Messages {
		parsed, err := parseNativeReqOpenAIMessage(msg)
		if err != nil {
			return nil, "", fmt.Errorf("failed to parse message %d: %w", i, err)
		}
		req.Messages = append(req.Messages, parsed...)
	}

	// Parse tools
	for _, tool := range nr.Tools {
		req.Tools = append(req.Tools, llm.ToolDefinition{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
		})
	}

	// Parse tool choice
	if len(nr.ToolChoice) > 0 {
		tc, err := parseNativeReqOpenAIToolChoice(nr.ToolChoice)
		if err != nil {
			return nil, "", fmt.Errorf("failed to parse tool_choice: %w", err)
		}
		req.ToolChoice = tc
	}

	// Parse response format
	if nr.ResponseFormat != nil {
		switch nr.ResponseFormat.Type {
		case "json_object":
			req.ResponseFormat = &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}
		case "text":
			req.ResponseFormat = &llm.ResponseFormat{Type: llm.ResponseFormatText}
		}
	}

	return req, nr.Model, nil
}

// parseNativeReqOpenAIMessage converts a native OpenAI request message to
// unified llm.Message(s). An assistant message with tool_calls produces a single
// message with multiple parts. A tool role message produces a user message with
// a tool response part.
func parseNativeReqOpenAIMessage(msg nativeReqMessage) ([]llm.Message, error) {
	switch msg.Role {
	case "system":
		text, err := parseOpenAIContent(msg.Content)
		if err != nil {
			return nil, fmt.Errorf("failed to parse system content: %w", err)
		}
		return []llm.Message{llm.NewMessage(llm.RoleSystem, text...)}, nil

	case "user":
		parts, err := parseOpenAIContent(msg.Content)
		if err != nil {
			return nil, fmt.Errorf("failed to parse user content: %w", err)
		}
		return []llm.Message{llm.NewMessage(llm.RoleUser, parts...)}, nil

	case "assistant":
		var parts []*llm.Part

		// Parse text content if present
		if len(msg.Content) > 0 && string(msg.Content) != "null" {
			textParts, err := parseOpenAIContent(msg.Content)
			if err != nil {
				return nil, fmt.Errorf("failed to parse assistant content: %w", err)
			}
			parts = append(parts, textParts...)
		}

		// Parse tool calls
		for _, tc := range msg.ToolCalls {
			parts = append(parts, llm.NewToolRequestPart(&llm.ToolRequest{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			}))
		}

		return []llm.Message{llm.NewMessage(llm.RoleAssistant, parts...)}, nil

	case "tool":
		// Tool results map to a user message with a tool response part
		content := string(msg.Content)
		// Try to unquote if it's a JSON string
		var unquoted string
		if err := json.Unmarshal(msg.Content, &unquoted); err == nil {
			content = unquoted
		}

		resp := &llm.ToolResponse{
			ID:     msg.ToolCallID,
			Result: json.RawMessage(content),
		}
		return []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(resp))}, nil

	default:
		return nil, fmt.Errorf("unsupported message role: %s", msg.Role)
	}
}

// parseOpenAIContent handles the OpenAI content field which can be either
// a plain string or an array of content parts.
func parseOpenAIContent(raw json.RawMessage) ([]*llm.Part, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	// Try string first
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []*llm.Part{llm.NewTextPart(s)}, nil
	}

	// Try array of content parts
	var parts []nativeReqContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("content is neither a string nor an array: %w", err)
	}

	var result []*llm.Part
	for _, p := range parts {
		switch p.Type {
		case "text":
			result = append(result, llm.NewTextPart(p.Text))
		default:
			return nil, fmt.Errorf("unsupported content part type: %s", p.Type)
		}
	}

	return result, nil
}

// parseNativeReqOpenAIToolChoice converts OpenAI tool_choice JSON to llm.ToolChoice.
// OpenAI tool_choice can be a string ("auto", "none", "required") or an object
// with type "function" and a function name.
func parseNativeReqOpenAIToolChoice(raw json.RawMessage) (*llm.ToolChoice, error) {
	// Try string first
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto":
			return &llm.ToolChoice{Type: llm.ToolChoiceAuto}, nil
		case "none":
			return &llm.ToolChoice{Type: llm.ToolChoiceNone}, nil
		case "required":
			return &llm.ToolChoice{Type: llm.ToolChoiceRequired}, nil
		default:
			return nil, fmt.Errorf("unsupported tool_choice string: %s", s)
		}
	}

	// Try object
	var obj nativeReqToolChoiceObj
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("tool_choice is neither a string nor an object: %w", err)
	}

	if obj.Type == "function" && obj.Function != nil {
		name := obj.Function.Name
		return &llm.ToolChoice{Type: llm.ToolChoiceSpecific, Name: &name}, nil
	}

	return nil, fmt.Errorf("unsupported tool_choice object type: %s", obj.Type)
}
