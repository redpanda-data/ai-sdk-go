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

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFromNative(t *testing.T) {
	rm := NewRequestMapper(&Config{
		ModelName: "claude-sonnet-4-20250514",
		MaxTokens: 1024,
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
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "Hello, world!"}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				assert.Equal(t, llm.RoleUser, req.Messages[0].Role)
				assert.Equal(t, "Hello, world!", req.Messages[0].TextContent())
			},
		},
		{
			name: "multi-turn conversation",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "What is 2+2?"},
					{"role": "assistant", "content": [{"type": "text", "text": "4"}]},
					{"role": "user", "content": "And 3+3?"}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 3)
				assert.Equal(t, llm.RoleUser, req.Messages[0].Role)
				assert.Equal(t, "What is 2+2?", req.Messages[0].TextContent())
				assert.Equal(t, llm.RoleAssistant, req.Messages[1].Role)
				assert.Equal(t, "4", req.Messages[1].TextContent())
				assert.Equal(t, llm.RoleUser, req.Messages[2].Role)
				assert.Equal(t, "And 3+3?", req.Messages[2].TextContent())
			},
		},
		{
			name: "system message as string",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"system": "You are a helpful assistant.",
				"messages": [
					{"role": "user", "content": "Hi"}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 2)
				assert.Equal(t, llm.RoleSystem, req.Messages[0].Role)
				assert.Equal(t, "You are a helpful assistant.", req.Messages[0].TextContent())
				assert.Equal(t, llm.RoleUser, req.Messages[1].Role)
			},
		},
		{
			name: "system message as array of text blocks",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"system": [
					{"type": "text", "text": "You are a helpful assistant."},
					{"type": "text", "text": "Be concise."}
				],
				"messages": [
					{"role": "user", "content": "Hi"}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 2)
				assert.Equal(t, llm.RoleSystem, req.Messages[0].Role)
				require.Len(t, req.Messages[0].Content, 2)
				assert.Equal(t, "You are a helpful assistant.", req.Messages[0].Content[0].Text)
				assert.Equal(t, "Be concise.", req.Messages[0].Content[1].Text)
			},
		},
		{
			name: "tool definitions",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"tools": [
					{
						"name": "get_weather",
						"description": "Get the weather for a location",
						"input_schema": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
					}
				],
				"messages": [
					{"role": "user", "content": "What's the weather in London?"}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Tools, 1)
				assert.Equal(t, "get_weather", req.Tools[0].Name)
				assert.Equal(t, "Get the weather for a location", req.Tools[0].Description)
				assert.Contains(t, string(req.Tools[0].Parameters), `"location"`)
			},
		},
		{
			name: "tool calling flow",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"tools": [
					{
						"name": "get_weather",
						"description": "Get weather",
						"input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
					}
				],
				"messages": [
					{"role": "user", "content": "Weather in London?"},
					{"role": "assistant", "content": [
						{"type": "text", "text": "Let me check."},
						{"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "London"}}
					]},
					{"role": "user", "content": [
						{"type": "tool_result", "tool_use_id": "toolu_123", "content": "Sunny, 22C"}
					]}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 3)

				// Assistant message with text + tool_use
				assistantMsg := req.Messages[1]
				assert.Equal(t, llm.RoleAssistant, assistantMsg.Role)
				require.Len(t, assistantMsg.Content, 2)
				assert.True(t, assistantMsg.Content[0].IsText())
				assert.Equal(t, "Let me check.", assistantMsg.Content[0].Text)
				assert.True(t, assistantMsg.Content[1].IsToolRequest())
				assert.Equal(t, "toolu_123", assistantMsg.Content[1].ToolRequest.ID)
				assert.Equal(t, "get_weather", assistantMsg.Content[1].ToolRequest.Name)
				assert.JSONEq(t, `{"location": "London"}`, string(assistantMsg.Content[1].ToolRequest.Arguments))

				// User message with tool_result
				userMsg := req.Messages[2]
				assert.Equal(t, llm.RoleUser, userMsg.Role)
				require.Len(t, userMsg.Content, 1)
				assert.True(t, userMsg.Content[0].IsToolResponse())
				assert.Equal(t, "toolu_123", userMsg.Content[0].ToolResponse.ID)
				assert.Equal(t, "Sunny, 22C", string(userMsg.Content[0].ToolResponse.Result))
			},
		},
		{
			name: "tool_result with error",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": [
						{"type": "tool_result", "tool_use_id": "toolu_456", "content": "connection timeout", "is_error": true}
					]}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				part := req.Messages[0].Content[0]
				assert.True(t, part.IsToolResponse())
				assert.Equal(t, "toolu_456", part.ToolResponse.ID)
				assert.Equal(t, "connection timeout", part.ToolResponse.Error)
			},
		},
		{
			name: "tool choice auto",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"tool_choice": {"type": "auto"},
				"tools": [{"name": "t", "description": "d", "input_schema": {}}],
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceAuto, req.ToolChoice.Type)
				assert.Nil(t, req.ToolChoice.Name)
			},
		},
		{
			name: "tool choice any maps to required",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"tool_choice": {"type": "any"},
				"tools": [{"name": "t", "description": "d", "input_schema": {}}],
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceRequired, req.ToolChoice.Type)
			},
		},
		{
			name: "tool choice specific",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"tool_choice": {"type": "tool", "name": "get_weather"},
				"tools": [{"name": "get_weather", "description": "d", "input_schema": {}}],
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceSpecific, req.ToolChoice.Type)
				require.NotNil(t, req.ToolChoice.Name)
				assert.Equal(t, "get_weather", *req.ToolChoice.Name)
			},
		},
		{
			name: "thinking block in assistant message",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "Think about this."},
					{"role": "assistant", "content": [
						{"type": "thinking", "thinking": "Let me reason about this..."},
						{"type": "text", "text": "Here is my answer."}
					]}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 2)
				assistantMsg := req.Messages[1]
				require.Len(t, assistantMsg.Content, 2)
				assert.True(t, assistantMsg.Content[0].IsReasoning())
				assert.Equal(t, "Let me reason about this...", assistantMsg.Content[0].ReasoningTrace.Text)
				assert.True(t, assistantMsg.Content[1].IsText())
				assert.Equal(t, "Here is my answer.", assistantMsg.Content[1].Text)
			},
		},
		{
			name: "stream flag",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"stream": true,
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				// Stream is parsed but not mapped to llm.Request (it's a transport concern).
				// Just verify the request parsed successfully.
				require.Len(t, req.Messages, 1)
			},
		},
		{
			name: "user content as array of text blocks",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": [
						{"type": "text", "text": "Hello "},
						{"type": "text", "text": "world"}
					]}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				require.Len(t, req.Messages[0].Content, 2)
				assert.Equal(t, "Hello ", req.Messages[0].Content[0].Text)
				assert.Equal(t, "world", req.Messages[0].Content[1].Text)
			},
		},
		{
			name: "empty messages array",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": []
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				assert.Empty(t, req.Messages)
			},
		},
		{
			name: "no system field",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				assert.Equal(t, llm.RoleUser, req.Messages[0].Role)
			},
		},
		{
			name: "no tools or tool_choice",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				assert.Empty(t, req.Tools)
				assert.Nil(t, req.ToolChoice)
			},
		},
		{
			name: "tool_result with array content",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": [
						{"type": "tool_result", "tool_use_id": "toolu_789", "content": [{"type": "text", "text": "result data"}]}
					]}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				part := req.Messages[0].Content[0]
				assert.True(t, part.IsToolResponse())
				assert.Equal(t, "result data", string(part.ToolResponse.Result))
			},
		},
		{
			name:    "invalid JSON",
			body:    `{invalid`,
			wantErr: true,
		},
		{
			name:    "invalid content type",
			body:    `{"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [{"role": "user", "content": [{"type": "unknown_type"}]}]}`,
			wantErr: true,
		},
		{
			name: "tool_use with null input",
			body: `{
				"model": "claude-sonnet-4-20250514",
				"max_tokens": 1024,
				"messages": [
					{"role": "assistant", "content": [
						{"type": "tool_use", "id": "toolu_abc", "name": "no_args", "input": null}
					]}
				]
			}`,
			wantModel: "claude-sonnet-4-20250514",
			check: func(t *testing.T, req *llm.Request) {
				part := req.Messages[0].Content[0]
				assert.True(t, part.IsToolRequest())
				assert.Equal(t, "no_args", part.ToolRequest.Name)
				assert.JSONEq(t, `{}`, string(part.ToolRequest.Arguments))
			},
		},
		{
			name: "extracts different model name",
			body: `{
				"model": "claude-3-5-haiku-20241022",
				"max_tokens": 512,
				"messages": [{"role": "user", "content": "hi"}]
			}`,
			wantModel: "claude-3-5-haiku-20241022",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
			},
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

func TestFromNativeToolResultContent(t *testing.T) {
	// Verify extractToolResultText handles various content formats.
	tests := []struct {
		name string
		raw  json.RawMessage
		want string
	}{
		{"string content", json.RawMessage(`"hello"`), "hello"},
		{"array content", json.RawMessage(`[{"type":"text","text":"data"}]`), "data"},
		{"empty", nil, ""},
		{"multi-block array", json.RawMessage(`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`), "ab"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractToolResultText(tt.raw)
			assert.Equal(t, tt.want, got)
		})
	}
}
