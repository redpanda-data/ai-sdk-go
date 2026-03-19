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

package google

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestResponseToNative_SimpleText(t *testing.T) {
	resp := &llm.Response{
		ID: "resp-001",
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

	data, err := ResponseToNative(resp, "gemini-2.5-flash")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	assert.Equal(t, "gemini-2.5-flash", got["modelVersion"])

	candidates := got["candidates"].([]any)
	require.Len(t, candidates, 1)

	candidate := candidates[0].(map[string]any)
	assert.Equal(t, "STOP", candidate["finishReason"])
	assert.Equal(t, float64(0), candidate["index"])

	content := candidate["content"].(map[string]any)
	assert.Equal(t, "model", content["role"])

	parts := content["parts"].([]any)
	require.Len(t, parts, 1)
	assert.Equal(t, "Hello!", parts[0].(map[string]any)["text"])

	usage := got["usageMetadata"].(map[string]any)
	assert.Equal(t, float64(10), usage["promptTokenCount"])
	assert.Equal(t, float64(20), usage["candidatesTokenCount"])
	assert.Equal(t, float64(30), usage["totalTokenCount"])
}

func TestResponseToNative_ToolCall(t *testing.T) {
	resp := &llm.Response{
		ID: "resp-tool",
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Content: []*llm.Part{
				llm.NewTextPart("Let me check the weather."),
				llm.NewToolRequestPart(&llm.ToolRequest{
					ID:        "call_123",
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

	data, err := ResponseToNative(resp, "gemini-2.5-flash")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	candidates := got["candidates"].([]any)
	candidate := candidates[0].(map[string]any)
	// FinishReasonToolCalls maps to STOP in Gemini
	assert.Equal(t, "STOP", candidate["finishReason"])

	content := candidate["content"].(map[string]any)
	parts := content["parts"].([]any)
	require.Len(t, parts, 2)

	textPart := parts[0].(map[string]any)
	assert.Equal(t, "Let me check the weather.", textPart["text"])

	funcPart := parts[1].(map[string]any)
	funcCall := funcPart["functionCall"].(map[string]any)
	assert.Equal(t, "get_weather", funcCall["name"])
	args := funcCall["args"].(map[string]any)
	assert.Equal(t, "San Francisco", args["location"])
}

func TestResponseToNative_Thinking(t *testing.T) {
	resp := &llm.Response{
		ID: "resp-think",
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Content: []*llm.Part{
				llm.NewReasoningPart(&llm.ReasoningTrace{
					Text: "Let me think about this carefully...",
				}),
				llm.NewTextPart("The answer is 42."),
			},
		},
		FinishReason: llm.FinishReasonStop,
	}

	data, err := ResponseToNative(resp, "gemini-2.5-pro")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))

	candidates := got["candidates"].([]any)
	content := candidates[0].(map[string]any)["content"].(map[string]any)
	parts := content["parts"].([]any)
	require.Len(t, parts, 2)

	thinkPart := parts[0].(map[string]any)
	assert.Equal(t, true, thinkPart["thought"])
	assert.Equal(t, "Let me think about this carefully...", thinkPart["text"])

	textPart := parts[1].(map[string]any)
	assert.Equal(t, "The answer is 42.", textPart["text"])
}

func TestResponseToNative_FinishReasons(t *testing.T) {
	tests := []struct {
		name     string
		reason   llm.FinishReason
		expected string
	}{
		{"stop", llm.FinishReasonStop, "STOP"},
		{"length", llm.FinishReasonLength, "MAX_TOKENS"},
		{"tool_calls", llm.FinishReasonToolCalls, "STOP"},
		{"content_filter", llm.FinishReasonContentFilter, "SAFETY"},
		{"unknown", llm.FinishReasonUnknown, "STOP"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &llm.Response{
				ID: "resp-test",
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: []*llm.Part{llm.NewTextPart("test")},
				},
				FinishReason: tt.reason,
			}

			data, err := ResponseToNative(resp, "gemini-2.5-flash")
			require.NoError(t, err)

			var got map[string]any
			require.NoError(t, json.Unmarshal(data, &got))
			candidates := got["candidates"].([]any)
			candidate := candidates[0].(map[string]any)
			assert.Equal(t, tt.expected, candidate["finishReason"])
		})
	}
}

func TestResponseToNative_NoUsage(t *testing.T) {
	resp := &llm.Response{
		ID: "resp-no-usage",
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: []*llm.Part{llm.NewTextPart("hi")},
		},
		FinishReason: llm.FinishReasonStop,
	}

	data, err := ResponseToNative(resp, "gemini-2.5-flash")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(data, &got))
	assert.Nil(t, got["usageMetadata"])
}

func TestResponseToNative_NilResponse(t *testing.T) {
	_, err := ResponseToNative(nil, "gemini-2.5-flash")
	require.Error(t, err)
}

func TestEventToNative_TextDelta(t *testing.T) {
	event := llm.ContentPartEvent{
		Index: 0,
		Part:  llm.NewTextPart("Hello"),
	}

	payloads, err := EventToNative(event, "gemini-2.5-flash")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	candidates := got["candidates"].([]any)
	require.Len(t, candidates, 1)

	candidate := candidates[0].(map[string]any)
	content := candidate["content"].(map[string]any)
	assert.Equal(t, "model", content["role"])

	parts := content["parts"].([]any)
	require.Len(t, parts, 1)
	assert.Equal(t, "Hello", parts[0].(map[string]any)["text"])
}

func TestEventToNative_ToolCall(t *testing.T) {
	event := llm.ContentPartEvent{
		Index: 0,
		Part: llm.NewToolRequestPart(&llm.ToolRequest{
			ID:        "call_456",
			Name:      "search",
			Arguments: json.RawMessage(`{"query":"test"}`),
		}),
	}

	payloads, err := EventToNative(event, "gemini-2.5-flash")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	candidates := got["candidates"].([]any)
	content := candidates[0].(map[string]any)["content"].(map[string]any)
	parts := content["parts"].([]any)
	require.Len(t, parts, 1)

	funcCall := parts[0].(map[string]any)["functionCall"].(map[string]any)
	assert.Equal(t, "search", funcCall["name"])
	args := funcCall["args"].(map[string]any)
	assert.Equal(t, "test", args["query"])
}

func TestEventToNative_ThinkingDelta(t *testing.T) {
	event := llm.ContentPartEvent{
		Index: 0,
		Part: llm.NewReasoningPart(&llm.ReasoningTrace{
			Text: "Hmm, let me think...",
		}),
	}

	payloads, err := EventToNative(event, "gemini-2.5-pro")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	candidates := got["candidates"].([]any)
	content := candidates[0].(map[string]any)["content"].(map[string]any)
	parts := content["parts"].([]any)
	require.Len(t, parts, 1)

	part := parts[0].(map[string]any)
	assert.Equal(t, true, part["thought"])
	assert.Equal(t, "Hmm, let me think...", part["text"])
}

func TestEventToNative_StreamEnd(t *testing.T) {
	event := llm.StreamEndEvent{
		Response: &llm.Response{
			ID: "resp-end",
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

	payloads, err := EventToNative(event, "gemini-2.5-flash")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	assert.Equal(t, "gemini-2.5-flash", got["modelVersion"])

	candidates := got["candidates"].([]any)
	candidate := candidates[0].(map[string]any)
	assert.Equal(t, "STOP", candidate["finishReason"])

	usage := got["usageMetadata"].(map[string]any)
	assert.Equal(t, float64(50), usage["promptTokenCount"])
	assert.Equal(t, float64(25), usage["candidatesTokenCount"])
	assert.Equal(t, float64(75), usage["totalTokenCount"])
}

func TestEventToNative_StreamEndNoResponse(t *testing.T) {
	event := llm.StreamEndEvent{}

	payloads, err := EventToNative(event, "gemini-2.5-flash")
	require.NoError(t, err)
	require.Len(t, payloads, 1)

	var got map[string]any
	require.NoError(t, json.Unmarshal(payloads[0], &got))

	candidates := got["candidates"].([]any)
	candidate := candidates[0].(map[string]any)
	assert.Equal(t, "STOP", candidate["finishReason"])
	assert.Equal(t, "gemini-2.5-flash", got["modelVersion"])
}

func TestEventToNative_NilEvent(t *testing.T) {
	_, err := EventToNative(nil, "gemini-2.5-flash")
	require.Error(t, err)
}

func TestEventToNative_NilPart(t *testing.T) {
	event := llm.ContentPartEvent{
		Index: 0,
		Part:  nil,
	}

	_, err := EventToNative(event, "gemini-2.5-flash")
	require.Error(t, err)
}
