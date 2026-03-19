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
	"testing"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFromNative(t *testing.T) {
	rm := NewRequestMapper(&Config{
		ModelName: "gemini-2.5-flash",
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
				"contents": [
					{"role": "user", "parts": [{"text": "Hello, world!"}]}
				]
			}`,
			wantModel: "",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				assert.Equal(t, llm.RoleUser, req.Messages[0].Role)
				assert.Equal(t, "Hello, world!", req.Messages[0].TextContent())
			},
		},
		{
			name: "model in body",
			body: `{
				"model": "gemini-2.5-flash",
				"contents": [
					{"role": "user", "parts": [{"text": "Hi"}]}
				]
			}`,
			wantModel: "gemini-2.5-flash",
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
			},
		},
		{
			name: "multi-turn conversation",
			body: `{
				"contents": [
					{"role": "user", "parts": [{"text": "What is 2+2?"}]},
					{"role": "model", "parts": [{"text": "4"}]},
					{"role": "user", "parts": [{"text": "And 3+3?"}]}
				]
			}`,
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
			name: "system instruction",
			body: `{
				"systemInstruction": {"parts": [{"text": "You are a helpful assistant."}]},
				"contents": [
					{"role": "user", "parts": [{"text": "Hi"}]}
				]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 2)
				assert.Equal(t, llm.RoleSystem, req.Messages[0].Role)
				assert.Equal(t, "You are a helpful assistant.", req.Messages[0].TextContent())
				assert.Equal(t, llm.RoleUser, req.Messages[1].Role)
			},
		},
		{
			name: "system instruction with multiple parts",
			body: `{
				"systemInstruction": {"parts": [
					{"text": "You are helpful."},
					{"text": "Be concise."}
				]},
				"contents": [
					{"role": "user", "parts": [{"text": "Hi"}]}
				]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 2)
				assert.Equal(t, llm.RoleSystem, req.Messages[0].Role)
				require.Len(t, req.Messages[0].Content, 2)
				assert.Equal(t, "You are helpful.", req.Messages[0].Content[0].Text)
				assert.Equal(t, "Be concise.", req.Messages[0].Content[1].Text)
			},
		},
		{
			name: "tool definitions",
			body: `{
				"contents": [
					{"role": "user", "parts": [{"text": "What's the weather?"}]}
				],
				"tools": [{
					"functionDeclarations": [{
						"name": "get_weather",
						"description": "Get weather for a location",
						"parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
					}]
				}]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Tools, 1)
				assert.Equal(t, "get_weather", req.Tools[0].Name)
				assert.Equal(t, "Get weather for a location", req.Tools[0].Description)
				assert.Contains(t, string(req.Tools[0].Parameters), `"location"`)
			},
		},
		{
			name: "function call in model response",
			body: `{
				"contents": [
					{"role": "user", "parts": [{"text": "Weather in London?"}]},
					{"role": "model", "parts": [
						{"text": "Let me check."},
						{"functionCall": {"name": "get_weather", "args": {"location": "London"}}}
					]},
					{"role": "user", "parts": [
						{"functionResponse": {"name": "get_weather", "response": {"result": "sunny"}}}
					]}
				]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 3)

				// Model message with text + functionCall
				modelMsg := req.Messages[1]
				assert.Equal(t, llm.RoleAssistant, modelMsg.Role)
				require.Len(t, modelMsg.Content, 2)
				assert.True(t, modelMsg.Content[0].IsText())
				assert.Equal(t, "Let me check.", modelMsg.Content[0].Text)
				assert.True(t, modelMsg.Content[1].IsToolRequest())
				assert.Equal(t, "get_weather", modelMsg.Content[1].ToolRequest.Name)
				assert.JSONEq(t, `{"location": "London"}`, string(modelMsg.Content[1].ToolRequest.Arguments))

				// User message with functionResponse
				userMsg := req.Messages[2]
				assert.Equal(t, llm.RoleUser, userMsg.Role)
				require.Len(t, userMsg.Content, 1)
				assert.True(t, userMsg.Content[0].IsToolResponse())
				assert.Equal(t, "get_weather", userMsg.Content[0].ToolResponse.ID)
				assert.Equal(t, "get_weather", userMsg.Content[0].ToolResponse.Name)
				assert.JSONEq(t, `{"result": "sunny"}`, string(userMsg.Content[0].ToolResponse.Result))
			},
		},
		{
			name: "function call with null args",
			body: `{
				"contents": [
					{"role": "model", "parts": [
						{"functionCall": {"name": "no_args", "args": null}}
					]}
				]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				part := req.Messages[0].Content[0]
				assert.True(t, part.IsToolRequest())
				assert.Equal(t, "no_args", part.ToolRequest.Name)
				assert.JSONEq(t, `{}`, string(part.ToolRequest.Arguments))
			},
		},
		{
			name: "function call with empty args",
			body: `{
				"contents": [
					{"role": "model", "parts": [
						{"functionCall": {"name": "no_args"}}
					]}
				]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				part := req.Messages[0].Content[0]
				assert.True(t, part.IsToolRequest())
				assert.JSONEq(t, `{}`, string(part.ToolRequest.Arguments))
			},
		},
		{
			name: "tool config AUTO",
			body: `{
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
				"tools": [{"functionDeclarations": [{"name": "t", "description": "d"}]}],
				"toolConfig": {"functionCallingConfig": {"mode": "AUTO"}}
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceAuto, req.ToolChoice.Type)
				assert.Nil(t, req.ToolChoice.Name)
			},
		},
		{
			name: "tool config ANY maps to required",
			body: `{
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
				"tools": [{"functionDeclarations": [{"name": "t", "description": "d"}]}],
				"toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceRequired, req.ToolChoice.Type)
			},
		},
		{
			name: "tool config NONE",
			body: `{
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
				"toolConfig": {"functionCallingConfig": {"mode": "NONE"}}
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceNone, req.ToolChoice.Type)
			},
		},
		{
			name: "tool config ANY with single allowed function",
			body: `{
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
				"tools": [{"functionDeclarations": [{"name": "get_weather", "description": "d"}]}],
				"toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}}
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceSpecific, req.ToolChoice.Type)
				require.NotNil(t, req.ToolChoice.Name)
				assert.Equal(t, "get_weather", *req.ToolChoice.Name)
			},
		},
		{
			name: "tool config ANY with multiple allowed functions stays required",
			body: `{
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
				"tools": [{"functionDeclarations": [{"name": "a", "description": "d"}, {"name": "b", "description": "d"}]}],
				"toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["a", "b"]}}
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceRequired, req.ToolChoice.Type)
				assert.Nil(t, req.ToolChoice.Name)
			},
		},
		{
			name: "no system instruction",
			body: `{
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				assert.Equal(t, llm.RoleUser, req.Messages[0].Role)
			},
		},
		{
			name: "no tools or tool config",
			body: `{
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				assert.Empty(t, req.Tools)
				assert.Nil(t, req.ToolChoice)
			},
		},
		{
			name: "multiple text parts in single message",
			body: `{
				"contents": [
					{"role": "user", "parts": [
						{"text": "Hello "},
						{"text": "world"}
					]}
				]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 1)
				require.Len(t, req.Messages[0].Content, 2)
				assert.Equal(t, "Hello ", req.Messages[0].Content[0].Text)
				assert.Equal(t, "world", req.Messages[0].Content[1].Text)
			},
		},
		{
			name: "thought part in model message",
			body: `{
				"contents": [
					{"role": "user", "parts": [{"text": "Think about this."}]},
					{"role": "model", "parts": [
						{"thought": true, "text": "Let me reason..."},
						{"text": "Here is my answer."}
					]}
				]
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.Len(t, req.Messages, 2)
				modelMsg := req.Messages[1]
				require.Len(t, modelMsg.Content, 2)
				assert.True(t, modelMsg.Content[0].IsReasoning())
				assert.Equal(t, "Let me reason...", modelMsg.Content[0].ReasoningTrace.Text)
				assert.True(t, modelMsg.Content[1].IsText())
				assert.Equal(t, "Here is my answer.", modelMsg.Content[1].Text)
			},
		},
		{
			name:    "invalid JSON",
			body:    `{invalid`,
			wantErr: true,
		},
		{
			name:    "invalid role",
			body:    `{"contents": [{"role": "system", "parts": [{"text": "hi"}]}]}`,
			wantErr: true,
		},
		{
			name: "empty contents",
			body: `{"contents": []}`,
			check: func(t *testing.T, req *llm.Request) {
				assert.Empty(t, req.Messages)
			},
		},
		{
			name: "lowercase tool config mode",
			body: `{
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
				"toolConfig": {"functionCallingConfig": {"mode": "auto"}}
			}`,
			check: func(t *testing.T, req *llm.Request) {
				require.NotNil(t, req.ToolChoice)
				assert.Equal(t, llm.ToolChoiceAuto, req.ToolChoice.Type)
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
