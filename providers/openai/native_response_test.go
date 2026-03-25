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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestResponseToNative_SimpleText(t *testing.T) {
	resp := &llm.Response{
		ID: "chatcmpl-abc123",
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Content: []*llm.Part{
				llm.NewTextPart("Hello!"),
			},
		},
		FinishReason: llm.FinishReasonStop,
		Usage: &llm.TokenUsage{
			InputTokens:  10,
			OutputTokens: 20,
			TotalTokens:  30,
		},
	}

	data, err := ResponseToNative(resp, "gpt-4o")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	assert.Equal(t, "chatcmpl-abc123", got["id"])
	assert.Equal(t, "chat.completion", got["object"])
	assert.Equal(t, "gpt-4o", got["model"])

	choices := got["choices"].([]any)
	require.Len(t, choices, 1)

	choice := choices[0].(map[string]any)
	assert.Equal(t, float64(0), choice["index"])
	assert.Equal(t, "stop", choice["finish_reason"])

	msg := choice["message"].(map[string]any)
	assert.Equal(t, "assistant", msg["role"])
	assert.Equal(t, "Hello!", msg["content"])

	usage := got["usage"].(map[string]any)
	assert.Equal(t, float64(10), usage["prompt_tokens"])
	assert.Equal(t, float64(20), usage["completion_tokens"])
	assert.Equal(t, float64(30), usage["total_tokens"])
}

func TestResponseToNative_ToolCalls(t *testing.T) {
	resp := &llm.Response{
		ID: "chatcmpl-tool",
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Content: []*llm.Part{
				llm.NewTextPart("Let me check the weather."),
				llm.NewToolRequestPart(&llm.ToolRequest{
					ID:        "call_abc123",
					Name:      "get_weather",
					Arguments: json.RawMessage(`{"location":"San Francisco"}`),
				}),
			},
		},
		FinishReason: llm.FinishReasonToolCalls,
		Usage: &llm.TokenUsage{
			InputTokens:  80,
			OutputTokens: 40,
			TotalTokens:  120,
		},
	}

	data, err := ResponseToNative(resp, "gpt-4o")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	choices := got["choices"].([]any)
	choice := choices[0].(map[string]any)
	assert.Equal(t, "tool_calls", choice["finish_reason"])

	msg := choice["message"].(map[string]any)
	assert.Equal(t, "Let me check the weather.", msg["content"])

	toolCalls := msg["tool_calls"].([]any)
	require.Len(t, toolCalls, 1)

	tc := toolCalls[0].(map[string]any)
	assert.Equal(t, "call_abc123", tc["id"])
	assert.Equal(t, "function", tc["type"])

	fn := tc["function"].(map[string]any)
	assert.Equal(t, "get_weather", fn["name"])
	assert.JSONEq(t, `{"location":"San Francisco"}`, fn["arguments"].(string))
}

func TestResponseToNative_NilContent(t *testing.T) {
	resp := &llm.Response{
		ID: "chatcmpl-nil",
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Content: []*llm.Part{
				llm.NewToolRequestPart(&llm.ToolRequest{
					ID:        "call_only",
					Name:      "search",
					Arguments: json.RawMessage(`{}`),
				}),
			},
		},
		FinishReason: llm.FinishReasonToolCalls,
	}

	data, err := ResponseToNative(resp, "gpt-4o")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	choices := got["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	// Content should be null when there's no text
	assert.Nil(t, msg["content"])
	assert.NotNil(t, msg["tool_calls"])
}

func TestResponseToNative_FinishReasons(t *testing.T) {
	tests := []struct {
		name     string
		reason   llm.FinishReason
		expected string
	}{
		{"stop", llm.FinishReasonStop, "stop"},
		{"length", llm.FinishReasonLength, "length"},
		{"tool_calls", llm.FinishReasonToolCalls, "tool_calls"},
		{"content_filter", llm.FinishReasonContentFilter, "content_filter"},
		{"unknown", llm.FinishReasonUnknown, "stop"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &llm.Response{
				ID: "chatcmpl-test",
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: []*llm.Part{llm.NewTextPart("test")},
				},
				FinishReason: tt.reason,
			}

			data, err := ResponseToNative(resp, "gpt-4o")
			require.NoError(t, err)

			var got map[string]any
			require.NoError(t, json.Unmarshal(data, &got))

			choices := got["choices"].([]any)
			assert.Equal(t, tt.expected, choices[0].(map[string]any)["finish_reason"])
		})
	}
}

func TestResponseToNative_NilResponse(t *testing.T) {
	_, err := ResponseToNative(nil, "gpt-4o")
	require.Error(t, err)
}

func TestResponseToNative_NoUsage(t *testing.T) {
	resp := &llm.Response{
		ID: "chatcmpl-nousage",
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: []*llm.Part{llm.NewTextPart("hi")},
		},
		FinishReason: llm.FinishReasonStop,
	}

	data, err := ResponseToNative(resp, "gpt-4o")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))
	assert.Nil(t, got["usage"])
}

func TestEventToNative_TextDelta(t *testing.T) {
	event := llm.ContentPartEvent{
		Index: 0,
		Part:  llm.NewTextPart("Hello"),
	}

	payloads, err := EventToNative(event, "gpt-4o")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	assert.Equal(t, "chat.completion.chunk", got["object"])
	assert.Equal(t, "gpt-4o", got["model"])

	choices := got["choices"].([]any)
	require.Len(t, choices, 1)

	choice := choices[0].(map[string]any)
	assert.Equal(t, float64(0), choice["index"])
	assert.Nil(t, choice["finish_reason"])

	delta := choice["delta"].(map[string]any)
	assert.Equal(t, "Hello", delta["content"])
}

func TestEventToNative_ToolCallDelta(t *testing.T) {
	event := llm.ContentPartEvent{
		Index: 1,
		Part: llm.NewToolRequestPart(&llm.ToolRequest{
			ID:        "call_123",
			Name:      "search",
			Arguments: json.RawMessage(`{"query":"test"}`),
		}),
	}

	payloads, err := EventToNative(event, "gpt-4o")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	choices := got["choices"].([]any)
	choice := choices[0].(map[string]any)
	assert.Equal(t, float64(1), choice["index"])

	delta := choice["delta"].(map[string]any)
	toolCalls := delta["tool_calls"].([]any)
	require.Len(t, toolCalls, 1)

	tc := toolCalls[0].(map[string]any)
	assert.Equal(t, "call_123", tc["id"])
	assert.Equal(t, "function", tc["type"])

	fn := tc["function"].(map[string]any)
	assert.Equal(t, "search", fn["name"])
}

func TestEventToNative_StreamEnd(t *testing.T) {
	event := llm.StreamEndEvent{
		Response: &llm.Response{
			ID: "chatcmpl-end",
			Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: []*llm.Part{llm.NewTextPart("done")},
			},
			FinishReason: llm.FinishReasonStop,
			Usage: &llm.TokenUsage{
				InputTokens:  50,
				OutputTokens: 25,
				TotalTokens:  75,
			},
		},
	}

	payloads, err := EventToNative(event, "gpt-4o")
	require.NoError(t, err)
	require.Len(t, payloads, 3) // finish_reason chunk + usage chunk + [DONE]

	// First: finish_reason chunk
	var chunk1 map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &chunk1))
	choices := chunk1["choices"].([]any)
	choice := choices[0].(map[string]any)
	assert.Equal(t, "stop", choice["finish_reason"])

	// Second: usage chunk
	var chunk2 map[string]any
	require.NoError(t, json.Unmarshal(payloads[1], &chunk2))
	usage := chunk2["usage"].(map[string]any)
	assert.Equal(t, float64(50), usage["prompt_tokens"])
	assert.Equal(t, float64(25), usage["completion_tokens"])
	assert.Equal(t, float64(75), usage["total_tokens"])

	// Third: [DONE]
	assert.Equal(t, []byte("[DONE]"), payloads[2])
}

func TestEventToNative_StreamEndNoUsage(t *testing.T) {
	event := llm.StreamEndEvent{
		Response: &llm.Response{
			ID: "chatcmpl-end",
			Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: []*llm.Part{llm.NewTextPart("done")},
			},
			FinishReason: llm.FinishReasonStop,
		},
	}

	payloads, err := EventToNative(event, "gpt-4o")
	require.NoError(t, err)
	require.Len(t, payloads, 2) // finish_reason chunk + [DONE] (no usage chunk)

	assert.Equal(t, []byte("[DONE]"), payloads[1])
}

func TestEventToNative_StreamEndNilResponse(t *testing.T) {
	// StreamEndEvent with nil Response should still emit [DONE]
	event := llm.StreamEndEvent{}

	payloads, err := EventToNative(event, "gpt-4o")
	require.NoError(t, err)
	require.Len(t, payloads, 1)
	assert.Equal(t, []byte("[DONE]"), payloads[0])
}

func TestEventToNative_NilEvent(t *testing.T) {
	_, err := EventToNative(nil, "gpt-4o")
	require.Error(t, err)
}

func TestEventToNative_StreamEndToolCalls(t *testing.T) {
	event := llm.StreamEndEvent{
		Response: &llm.Response{
			ID: "chatcmpl-tc",
			Message: llm.Message{
				Role: llm.RoleAssistant,
				Content: []*llm.Part{
					llm.NewToolRequestPart(&llm.ToolRequest{
						ID:        "call_1",
						Name:      "search",
						Arguments: json.RawMessage(`{}`),
					}),
				},
			},
			FinishReason: llm.FinishReasonToolCalls,
		},
	}

	payloads, err := EventToNative(event, "gpt-4o")
	require.NoError(t, err)

	// finish_reason chunk + [DONE]
	require.Len(t, payloads, 2)

	var chunk map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &chunk))
	choices := chunk["choices"].([]any)
	assert.Equal(t, "tool_calls", choices[0].(map[string]any)["finish_reason"])
}
