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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestResponseToNative_SimpleText(t *testing.T) {
	resp := &llm.Response{
		ID: "msg_01XFDUDYJgAACzvnptvVoYEL",
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Content: []*llm.Part{
				llm.NewTextPart("Hello!"),
			},
		},
		FinishReason: llm.FinishReasonStop,
		Usage: &llm.TokenUsage{
			InputTokens:  100,
			OutputTokens: 50,
			TotalTokens:  150,
		},
	}

	data, err := ResponseToNative(resp, "claude-3-5-haiku-20241022")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	assert.Equal(t, "msg_01XFDUDYJgAACzvnptvVoYEL", got["id"])
	assert.Equal(t, "message", got["type"])
	assert.Equal(t, "assistant", got["role"])
	assert.Equal(t, "claude-3-5-haiku-20241022", got["model"])
	assert.Equal(t, "end_turn", got["stop_reason"])

	content := got["content"].([]any)
	require.Len(t, content, 1)

	block := content[0].(map[string]any)
	assert.Equal(t, "text", block["type"])
	assert.Equal(t, "Hello!", block["text"])

	usage := got["usage"].(map[string]any)
	assert.Equal(t, float64(100), usage["input_tokens"])
	assert.Equal(t, float64(50), usage["output_tokens"])
}

func TestResponseToNative_ToolUse(t *testing.T) {
	resp := &llm.Response{
		ID: "msg_tool",
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Content: []*llm.Part{
				llm.NewTextPart("Let me check the weather."),
				llm.NewToolRequestPart(&llm.ToolRequest{
					ID:        "toolu_01A09q90qw90lq917835lq9",
					Name:      "get_weather",
					Arguments: json.RawMessage(`{"location":"San Francisco"}`),
				}),
			},
		},
		FinishReason: llm.FinishReasonToolCalls,
		Usage: &llm.TokenUsage{
			InputTokens:  80,
			OutputTokens: 40,
		},
	}

	data, err := ResponseToNative(resp, "claude-3-5-haiku-20241022")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	assert.Equal(t, "tool_use", got["stop_reason"])

	content := got["content"].([]any)
	require.Len(t, content, 2)

	textBlock := content[0].(map[string]any)
	assert.Equal(t, "text", textBlock["type"])
	assert.Equal(t, "Let me check the weather.", textBlock["text"])

	toolBlock := content[1].(map[string]any)
	assert.Equal(t, "tool_use", toolBlock["type"])
	assert.Equal(t, "toolu_01A09q90qw90lq917835lq9", toolBlock["id"])
	assert.Equal(t, "get_weather", toolBlock["name"])

	input := toolBlock["input"].(map[string]any)
	assert.Equal(t, "San Francisco", input["location"])
}

func TestResponseToNative_Thinking(t *testing.T) {
	resp := &llm.Response{
		ID: "msg_think",
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Content: []*llm.Part{
				llm.NewReasoningPart(&llm.ReasoningTrace{
					ID:   "sig_abc123",
					Text: "Let me think about this carefully...",
				}),
				llm.NewTextPart("The answer is 42."),
			},
		},
		FinishReason: llm.FinishReasonStop,
	}

	data, err := ResponseToNative(resp, "claude-sonnet-4-5-20250514")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	content := got["content"].([]any)
	require.Len(t, content, 2)

	thinkBlock := content[0].(map[string]any)
	assert.Equal(t, "thinking", thinkBlock["type"])
	assert.Equal(t, "Let me think about this carefully...", thinkBlock["thinking"])
	assert.Equal(t, "sig_abc123", thinkBlock["signature"])

	textBlock := content[1].(map[string]any)
	assert.Equal(t, "text", textBlock["type"])
	assert.Equal(t, "The answer is 42.", textBlock["text"])
}

func TestResponseToNative_StopReasons(t *testing.T) {
	tests := []struct {
		name     string
		reason   llm.FinishReason
		expected string
	}{
		{"stop", llm.FinishReasonStop, "end_turn"},
		{"length", llm.FinishReasonLength, "max_tokens"},
		{"tool_calls", llm.FinishReasonToolCalls, "tool_use"},
		{"content_filter", llm.FinishReasonContentFilter, "refusal"},
		{"unknown", llm.FinishReasonUnknown, "end_turn"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &llm.Response{
				ID: "msg_test",
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: []*llm.Part{llm.NewTextPart("test")},
				},
				FinishReason: tt.reason,
			}

			data, err := ResponseToNative(resp, "claude-3-5-haiku-20241022")
			require.NoError(t, err)

			var got map[string]any
			require.NoError(t, json.Unmarshal(data, &got))
			assert.Equal(t, tt.expected, got["stop_reason"])
		})
	}
}

func TestResponseToNative_UsageSerialization(t *testing.T) {
	resp := &llm.Response{
		ID: "msg_usage",
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: []*llm.Part{llm.NewTextPart("hi")},
		},
		FinishReason: llm.FinishReasonStop,
		Usage: &llm.TokenUsage{
			InputTokens:  1234,
			OutputTokens: 567,
			TotalTokens:  1801,
			CachedTokens: 200,
		},
	}

	data, err := ResponseToNative(resp, "claude-3-5-haiku-20241022")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	usage := got["usage"].(map[string]any)
	assert.Equal(t, float64(1234), usage["input_tokens"])
	assert.Equal(t, float64(567), usage["output_tokens"])
}

func TestResponseToNative_NilResponse(t *testing.T) {
	_, err := ResponseToNative(nil, "claude-3-5-haiku-20241022")
	require.Error(t, err)
}

func TestEventToNative_TextDelta(t *testing.T) {
	event := llm.ContentPartEvent{
		Index: 0,
		Part:  llm.NewTextPart("Hello"),
	}

	payloads, err := EventToNative(event, "claude-3-5-haiku-20241022")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	assert.Equal(t, "content_block_delta", got["type"])
	assert.Equal(t, float64(0), got["index"])

	delta := got["delta"].(map[string]any)
	assert.Equal(t, "text_delta", delta["type"])
	assert.Equal(t, "Hello", delta["text"])
}

func TestEventToNative_ToolCall(t *testing.T) {
	event := llm.ContentPartEvent{
		Index: 1,
		Part: llm.NewToolRequestPart(&llm.ToolRequest{
			ID:        "toolu_123",
			Name:      "search",
			Arguments: json.RawMessage(`{"query":"test"}`),
		}),
	}

	payloads, err := EventToNative(event, "claude-3-5-haiku-20241022")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	assert.Equal(t, "content_block_start", got["type"])
	assert.Equal(t, float64(1), got["index"])

	block := got["content_block"].(map[string]any)
	assert.Equal(t, "tool_use", block["type"])
	assert.Equal(t, "toolu_123", block["id"])
	assert.Equal(t, "search", block["name"])
}

func TestEventToNative_ThinkingDelta(t *testing.T) {
	event := llm.ContentPartEvent{
		Index: 0,
		Part: llm.NewReasoningPart(&llm.ReasoningTrace{
			ID:   "sig_xyz",
			Text: "Hmm, let me think...",
		}),
	}

	payloads, err := EventToNative(event, "claude-sonnet-4-5-20250514")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	assert.Equal(t, "content_block_delta", got["type"])

	delta := got["delta"].(map[string]any)
	assert.Equal(t, "thinking_delta", delta["type"])
	assert.Equal(t, "Hmm, let me think...", delta["text"])
}

func TestEventToNative_StreamEnd(t *testing.T) {
	event := llm.StreamEndEvent{
		Response: &llm.Response{
			ID: "msg_end",
			Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: []*llm.Part{llm.NewTextPart("done")},
			},
			FinishReason: llm.FinishReasonStop,
			Usage: &llm.TokenUsage{
				InputTokens:  50,
				OutputTokens: 25,
			},
		},
	}

	payloads, err := EventToNative(event, "claude-3-5-haiku-20241022")
	require.NoError(t, err)
	require.Len(t, payloads, 2)

	// First payload: message_delta
	var delta map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &delta))
	assert.Equal(t, "message_delta", delta["type"])

	d := delta["delta"].(map[string]any)
	assert.Equal(t, "end_turn", d["stop_reason"])

	usage := delta["usage"].(map[string]any)
	assert.Equal(t, float64(50), usage["input_tokens"])
	assert.Equal(t, float64(25), usage["output_tokens"])

	// Second payload: message_stop
	var stop map[string]any
	require.NoError(t, json.Unmarshal(payloads[1], &stop))
	assert.Equal(t, "message_stop", stop["type"])
}

func TestEventToNative_StreamEndWithError(t *testing.T) {
	// StreamEndEvent with nil Response (error case) should still emit message_stop
	event := llm.StreamEndEvent{}

	payloads, err := EventToNative(event, "claude-3-5-haiku-20241022")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))
	assert.Equal(t, "message_stop", got["type"])
}

func TestEventToNative_NilEvent(t *testing.T) {
	_, err := EventToNative(nil, "claude-3-5-haiku-20241022")
	require.Error(t, err)
}
