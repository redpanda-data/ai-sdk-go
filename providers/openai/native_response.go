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

// Outbound types for serializing llm.Response to OpenAI /v1/chat/completions format.

type outboundCompletion struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Model   string            `json:"model"`
	Choices []outboundChoice  `json:"choices"`
	Usage   *outboundUsage    `json:"usage,omitempty"`
}

type outboundChoice struct {
	Index        int              `json:"index"`
	Message      outboundMessage  `json:"message"`
	FinishReason string           `json:"finish_reason"`
}

type outboundMessage struct {
	Role      string                  `json:"role"`
	Content   *string                 `json:"content"`
	ToolCalls []outboundToolCall      `json:"tool_calls,omitempty"`
}

type outboundToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function outboundToolCallFn  `json:"function"`
}

type outboundToolCallFn struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type outboundUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Streaming chunk types for OpenAI SSE format.

type outboundChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Model   string               `json:"model"`
	Choices []outboundChunkChoice `json:"choices"`
	Usage   *outboundUsage       `json:"usage,omitempty"`
}

type outboundChunkChoice struct {
	Index        int             `json:"index"`
	Delta        outboundDelta   `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

type outboundDelta struct {
	Role      string              `json:"role,omitempty"`
	Content   *string             `json:"content,omitempty"`
	ToolCalls []outboundToolCall  `json:"tool_calls,omitempty"`
}

// ResponseToNative serializes a unified llm.Response to OpenAI's native
// /v1/chat/completions JSON format.
func ResponseToNative(resp *llm.Response, model string) ([]byte, error) {
	if resp == nil {
		return nil, fmt.Errorf("nil response")
	}

	msg := outboundMessage{
		Role: "assistant",
	}

	// Collect text content and tool calls separately
	var textContent string
	var toolCalls []outboundToolCall

	for _, part := range resp.Message.Content {
		switch {
		case part.IsText():
			textContent += part.Text

		case part.IsToolRequest() && part.ToolRequest != nil:
			toolCalls = append(toolCalls, outboundToolCall{
				ID:   part.ToolRequest.ID,
				Type: "function",
				Function: outboundToolCallFn{
					Name:      part.ToolRequest.Name,
					Arguments: string(part.ToolRequest.Arguments),
				},
			})
		}
	}

	if textContent != "" {
		msg.Content = &textContent
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	completion := outboundCompletion{
		ID:     resp.ID,
		Object: "chat.completion",
		Model:  model,
		Choices: []outboundChoice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: openaiFinishReason(resp.FinishReason),
			},
		},
	}

	if resp.Usage != nil {
		completion.Usage = &outboundUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	return json.Marshal(completion)
}

// EventToNative serializes a single llm.Event to one or more OpenAI SSE data
// payloads. Some events map to multiple payloads (e.g., StreamEndEvent produces
// a finish_reason chunk, optionally a usage chunk, and the [DONE] sentinel).
func EventToNative(event llm.Event, model string) ([][]byte, error) {
	if event == nil {
		return nil, fmt.Errorf("nil event")
	}

	switch e := event.(type) {
	case llm.ContentPartEvent:
		return contentPartEventToOpenAINative(e, model)

	case llm.StreamEndEvent:
		return streamEndEventToOpenAINative(e, model)

	default:
		return nil, fmt.Errorf("unsupported event type: %T", event)
	}
}

func contentPartEventToOpenAINative(e llm.ContentPartEvent, model string) ([][]byte, error) {
	if e.Part == nil {
		return nil, fmt.Errorf("nil part in content event")
	}

	switch {
	case e.Part.IsText():
		text := e.Part.Text
		chunk := outboundChunk{
			ID:     "chatcmpl-native",
			Object: "chat.completion.chunk",
			Model:  model,
			Choices: []outboundChunkChoice{
				{
					Index: e.Index,
					Delta: outboundDelta{
						Content: &text,
					},
					FinishReason: nil,
				},
			},
		}

		data, err := json.Marshal(chunk)
		if err != nil {
			return nil, err
		}

		return [][]byte{data}, nil

	case e.Part.IsToolRequest() && e.Part.ToolRequest != nil:
		chunk := outboundChunk{
			ID:     "chatcmpl-native",
			Object: "chat.completion.chunk",
			Model:  model,
			Choices: []outboundChunkChoice{
				{
					Index: e.Index,
					Delta: outboundDelta{
						ToolCalls: []outboundToolCall{
							{
								ID:   e.Part.ToolRequest.ID,
								Type: "function",
								Function: outboundToolCallFn{
									Name:      e.Part.ToolRequest.Name,
									Arguments: string(e.Part.ToolRequest.Arguments),
								},
							},
						},
					},
					FinishReason: nil,
				},
			},
		}

		data, err := json.Marshal(chunk)
		if err != nil {
			return nil, err
		}

		return [][]byte{data}, nil

	default:
		return nil, fmt.Errorf("unsupported part kind: %s", e.Part.Kind)
	}
}

func streamEndEventToOpenAINative(e llm.StreamEndEvent, model string) ([][]byte, error) {
	var results [][]byte

	if e.Response != nil {
		// Emit chunk with finish_reason
		reason := openaiFinishReason(e.Response.FinishReason)
		chunk := outboundChunk{
			ID:     "chatcmpl-native",
			Object: "chat.completion.chunk",
			Model:  model,
			Choices: []outboundChunkChoice{
				{
					Index:        0,
					Delta:        outboundDelta{},
					FinishReason: &reason,
				},
			},
		}

		data, err := json.Marshal(chunk)
		if err != nil {
			return nil, err
		}

		results = append(results, data)

		// Emit usage chunk if available
		if e.Response.Usage != nil {
			usageChunk := outboundChunk{
				ID:      "chatcmpl-native",
				Object:  "chat.completion.chunk",
				Model:   model,
				Choices: []outboundChunkChoice{},
				Usage: &outboundUsage{
					PromptTokens:     e.Response.Usage.InputTokens,
					CompletionTokens: e.Response.Usage.OutputTokens,
					TotalTokens:      e.Response.Usage.TotalTokens,
				},
			}

			data, err := json.Marshal(usageChunk)
			if err != nil {
				return nil, err
			}

			results = append(results, data)
		}
	}

	// [DONE] sentinel is always the final payload
	results = append(results, []byte("[DONE]"))

	return results, nil
}

// openaiFinishReason maps a unified FinishReason to OpenAI's finish_reason string.
func openaiFinishReason(reason llm.FinishReason) string {
	switch reason {
	case llm.FinishReasonStop:
		return "stop"
	case llm.FinishReasonLength:
		return "length"
	case llm.FinishReasonToolCalls:
		return "tool_calls"
	case llm.FinishReasonContentFilter:
		return "content_filter"
	default:
		return "stop"
	}
}
