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
	"testing"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFromNative(t *testing.T) {
	rm := NewRequestMapper(&Config{
		ModelName: "gpt-4o",
	})

	tests := []struct {
		name      string
		body      string
		wantModel string
		check     func(t *testing.T, req *llm.Request)
		wantErr   bool
	}{
		{
			name: "simple text message",
			body: `{
				"model": "gpt-4o",
				"messages": [
					{"role": "user", "content": "Hello, world!"}
				]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				assert.Equal(t, llm.RoleUser, req.Messages[0].Role)
				assert.Equal(t, "Hello, world!", req.Messages[0].TextContent())
			},
		},
		{
			name: "system message",
			body: `{
				"model": "gpt-4o",
				"messages": [
					{"role": "system", "content": "You are a helpful assistant."},
					{"role": "user", "content": "Hi"}
				]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 2)
				assert.Equal(t, llm.RoleSystem, req.Messages[0].Role)
				assert.Equal(t, "You are a helpful assistant.", req.Messages[0].TextContent())
				assert.Equal(t, llm.RoleUser, req.Messages[1].Role)
			},
		},
		{
			name: "multi-turn conversation",
			body: `{
				"model": "gpt-4o",
				"messages": [
					{"role": "user", "content": "What is 2+2?"},
					{"role": "assistant", "content": "4"},
					{"role": "user", "content": "And 3+3?"}
				]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 3)
				assert.Equal(t, llm.RoleUser, req.Messages[0].Role)
				assert.Equal(t, "What is 2+2?", req.Messages[0].TextContent())
				assert.Equal(t, llm.RoleAssistant, req.Messages[1].Role)
				assert.Equal(t, "4", req.Messages[1].TextContent())
				assert.Equal(t, llm.RoleUser, req.Messages[2].Role)
			},
		},
		{
			name: "user content as array of text parts",
			body: `{
				"model": "gpt-4o",
				"messages": [
					{"role": "user", "content": [
						{"type": "text", "text": "Hello "},
						{"type": "text", "text": "world"}
					]}
				]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				require.Len(t, req.Messages[0].Content, 2)
				assert.Equal(t, "Hello ", req.Messages[0].Content[0].Text)
				assert.Equal(t, "world", req.Messages[0].Content[1].Text)
			},
		},
		{
			name: "tool definitions",
			body: `{
				"model": "gpt-4o",
				"tools": [
					{
						"type": "function",
						"function": {
							"name": "get_weather",
							"description": "Get the weather for a location",
							"parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
						}
					}
				],
				"messages": [
					{"role": "user", "content": "What's the weather in London?"}
				]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Tools, 1)
				assert.Equal(t, "get_weather", req.Tools[0].Name)
				assert.Equal(t, "Get the weather for a location", req.Tools[0].Description)
				assert.Contains(t, string(req.Tools[0].Parameters), `"location"`)
			},
		},
		{
			name: "assistant with tool calls",
			body: `{
				"model": "gpt-4o",
				"messages": [
					{"role": "user", "content": "Weather in London?"},
					{"role": "assistant", "content": "Let me check.", "tool_calls": [
						{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"London\"}"}}
					]},
					{"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 22C"}
				]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 3)

				// Assistant message with text + tool_call
				assistantMsg := req.Messages[1]
				assert.Equal(t, llm.RoleAssistant, assistantMsg.Role)
				require.Len(t, assistantMsg.Content, 2)
				assert.True(t, assistantMsg.Content[0].IsText())
				assert.Equal(t, "Let me check.", assistantMsg.Content[0].Text)
				assert.True(t, assistantMsg.Content[1].IsToolRequest())
				assert.Equal(t, "call_123", assistantMsg.Content[1].ToolRequest.ID)
				assert.Equal(t, "get_weather", assistantMsg.Content[1].ToolRequest.Name)
				assert.JSONEq(t, `{"location":"London"}`, string(assistantMsg.Content[1].ToolRequest.Arguments))

				// Tool result maps to user message with tool response
				toolMsg := req.Messages[2]
				assert.Equal(t, llm.RoleUser, toolMsg.Role)
				require.Len(t, toolMsg.Content, 1)
				assert.True(t, toolMsg.Content[0].IsToolResponse())
				assert.Equal(t, "call_123", toolMsg.Content[0].ToolResponse.ID)
				assert.Equal(t, "Sunny, 22C", string(toolMsg.Content[0].ToolResponse.Result))
			},
		},
		{
			name: "assistant with only tool calls no content",
			body: `{
				"model": "gpt-4o",
				"messages": [
					{"role": "assistant", "content": null, "tool_calls": [
						{"id": "call_abc", "type": "function", "function": {"name": "search", "arguments": "{\"q\":\"test\"}"}}
					]}
				]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				assistantMsg := req.Messages[0]
				require.Len(t, assistantMsg.Content, 1)
				assert.True(t, assistantMsg.Content[0].IsToolRequest())
				assert.Equal(t, "search", assistantMsg.Content[0].ToolRequest.Name)
			},
		},
		{
			name: "tool choice auto string",
			body: `{
				"model": "gpt-4o",
				"tool_choice": "auto",
				"tools": [{"type":"function","function":{"name":"t","description":"d","parameters":{}}}],
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceAuto, req.ToolChoice.Type)
				assert.Nil(t, req.ToolChoice.Name)
			},
		},
		{
			name: "tool choice none string",
			body: `{
				"model": "gpt-4o",
				"tool_choice": "none",
				"tools": [{"type":"function","function":{"name":"t","description":"d","parameters":{}}}],
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceNone, req.ToolChoice.Type)
			},
		},
		{
			name: "tool choice required string",
			body: `{
				"model": "gpt-4o",
				"tool_choice": "required",
				"tools": [{"type":"function","function":{"name":"t","description":"d","parameters":{}}}],
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceRequired, req.ToolChoice.Type)
			},
		},
		{
			name: "tool choice specific function object",
			body: `{
				"model": "gpt-4o",
				"tool_choice": {"type": "function", "function": {"name": "get_weather"}},
				"tools": [{"type":"function","function":{"name":"get_weather","description":"d","parameters":{}}}],
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceSpecific, req.ToolChoice.Type)
				require.NotNil(t, req.ToolChoice.Name)
				assert.Equal(t, "get_weather", *req.ToolChoice.Name)
			},
		},
		{
			name: "response format json_object",
			body: `{
				"model": "gpt-4o",
				"response_format": {"type": "json_object"},
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ResponseFormat)
				assert.Equal(t, llm.ResponseFormatJSONObject, req.ResponseFormat.Type)
			},
		},
		{
			name: "response format text",
			body: `{
				"model": "gpt-4o",
				"response_format": {"type": "text"},
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ResponseFormat)
				assert.Equal(t, llm.ResponseFormatText, req.ResponseFormat.Type)
			},
		},
		{
			name: "stream flag is parsed",
			body: `{
				"model": "gpt-4o",
				"stream": true,
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				// Stream is parsed but not mapped to llm.Request (transport concern).
				require.Len(t, req.Messages, 1)
			},
		},
		{
			name: "empty messages array",
			body: `{
				"model": "gpt-4o",
				"messages": []
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				assert.Empty(t, req.Messages)
			},
		},
		{
			name: "no tools or tool_choice",
			body: `{
				"model": "gpt-4o",
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				assert.Empty(t, req.Tools)
				assert.Nil(t, req.ToolChoice)
			},
		},
		{
			name: "extracts different model name",
			body: `{
				"model": "gpt-4o-mini",
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "gpt-4o-mini",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
			},
		},
		{
			name: "multiple tool calls in one assistant message",
			body: `{
				"model": "gpt-4o",
				"messages": [
					{"role": "assistant", "content": null, "tool_calls": [
						{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{\"q\":\"a\"}"}},
						{"id": "call_2", "type": "function", "function": {"name": "lookup", "arguments": "{\"id\":\"b\"}"}}
					]}
				]
			}`,
			wantModel: "gpt-4o",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				require.Len(t, req.Messages[0].Content, 2)
				assert.Equal(t, "search", req.Messages[0].Content[0].ToolRequest.Name)
				assert.Equal(t, "lookup", req.Messages[0].Content[1].ToolRequest.Name)
			},
		},
		{
			name:    "invalid JSON",
			body:    `{invalid`,
			wantErr: true,
		},
		{
			name:    "unsupported role",
			body:    `{"model": "gpt-4o", "messages": [{"role": "function", "content": "hi"}]}`,
			wantErr: true,
		},
		{
			name:    "unsupported content part type",
			body:    `{"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, model, err := rm.FromNative([]byte(tt.body))
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantModel, model)
			if tt.check != nil {
				tt.check(t, req)
			}
		})
	}
}
