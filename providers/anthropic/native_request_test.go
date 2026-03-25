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
	"os"
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

// TestFromNative_ToProvider_ClaudeCodePayload reproduces the exact failure seen
// when proxying Claude Code through the AI Gateway. The gateway does
// FromNative → ToProvider and sends the result to Anthropic. Two bugs:
//   1. max_tokens is lost (becomes 0) because ToProvider uses Config.MaxTokens
//   2. Tool schemas with enum, default, minItems etc. are mangled by
//      AdaptSchemaForAnthropic (Anthropic rejects with "JSON schema is invalid")
func TestFromNative_ToProvider_ClaudeCodePayload(t *testing.T) {
	payload, err := os.ReadFile("testdata/claude_code_request.json")
	require.NoError(t, err)

	rm := NewRequestMapper(&Config{})

	req, modelName, err := rm.FromNative(payload)
	require.NoError(t, err)
	assert.Equal(t, "claude-opus-4-6", modelName)
	require.Len(t, req.Tools, 4)

	apiReq, err := rm.ToProvider(req)
	require.NoError(t, err)

	// Bug 1: max_tokens must be preserved from the original request.
	assert.Equal(t, int64(64000), apiReq.MaxTokens, "max_tokens must be preserved from native request")

	// Bug 2: tool count must match.
	require.Len(t, apiReq.Tools, 4)

	toolsJSON, err := json.Marshal(apiReq.Tools)
	require.NoError(t, err)

	var tools []map[string]any
	require.NoError(t, json.Unmarshal(toolsJSON, &tools))

	for _, tool := range tools {
		name, _ := tool["name"].(string)
		_, hasCustom := tool["custom"]
		assert.False(t, hasCustom, "tool %s must not have 'custom' wrapper", name)

		schema, ok := tool["input_schema"].(map[string]any)
		require.True(t, ok, "tool %s must have input_schema as object", name)
		assert.Equal(t, "object", schema["type"], "tool %s schema type", name)
		_, hasProps := schema["properties"]
		assert.True(t, hasProps, "tool %s schema must have properties", name)
	}

	// Agent tool: enum on "model" must survive.
	agentTool := findToolByName(tools, "Agent")
	require.NotNil(t, agentTool)
	agentProps := agentTool["input_schema"].(map[string]any)["properties"].(map[string]any)
	modelProp := agentProps["model"].(map[string]any)
	assert.Contains(t, modelProp, "enum", "Agent.model must preserve enum")

	// AskUserQuestion: nested minItems/maxItems must survive.
	askTool := findToolByName(tools, "AskUserQuestion")
	require.NotNil(t, askTool)
	askProps := askTool["input_schema"].(map[string]any)["properties"].(map[string]any)
	questions := askProps["questions"].(map[string]any)
	assert.Contains(t, questions, "minItems", "questions must preserve minItems")
	assert.Contains(t, questions, "maxItems", "questions must preserve maxItems")

	// Edit: default on replace_all must survive.
	editTool := findToolByName(tools, "Edit")
	require.NotNil(t, editTool)
	editProps := editTool["input_schema"].(map[string]any)["properties"].(map[string]any)
	replaceAll := editProps["replace_all"].(map[string]any)
	assert.Contains(t, replaceAll, "default", "Edit.replace_all must preserve default")
}

func findToolByName(tools []map[string]any, name string) map[string]any {
	for _, t := range tools {
		if t["name"] == name {
			return t
		}
	}
	return nil
}

// TestFromNative_ToProvider_ToolSchemaRoundtrip_Complex tests with schemas that
// match what Claude Code actually sends -- nested objects, oneOf, $ref, etc.
// These are the schemas that trigger the "tools.N.custom.input_schema: JSON
// schema is invalid" error from Anthropic.
func TestFromNative_ToProvider_ToolSchemaRoundtrip_Complex(t *testing.T) {
	nativeRequest := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 8096,
		"messages": [
			{"role": "user", "content": "hello"}
		],
		"tools": [
			{
				"name": "Edit",
				"description": "Edit a file",
				"input_schema": {

					"type": "object",
					"properties": {
						"file_path": {"type": "string", "description": "Absolute path"},
						"old_string": {"type": "string", "description": "Text to replace"},
						"new_string": {"type": "string", "description": "Replacement text"},
						"replace_all": {"type": "boolean", "default": false}
					},
					"required": ["file_path", "old_string", "new_string"],
					"additionalProperties": false
				}
			},
			{
				"name": "Agent",
				"description": "Launch a sub-agent",
				"input_schema": {

					"type": "object",
					"additionalProperties": false,
					"properties": {
						"prompt": {"type": "string"},
						"description": {"type": "string"},
						"model": {
							"type": "string",
							"enum": ["sonnet", "opus", "haiku"]
						},
						"run_in_background": {"type": "boolean"},
						"isolation": {
							"type": "string",
							"enum": ["worktree"]
						},
						"subagent_type": {"type": "string"}
					},
					"required": ["description", "prompt"]
				}
			},
			{
				"name": "AskUserQuestion",
				"description": "Ask the user a question",
				"input_schema": {

					"type": "object",
					"additionalProperties": false,
					"properties": {
						"questions": {
							"type": "array",
							"items": {
								"type": "object",
								"additionalProperties": false,
								"properties": {
									"question": {"type": "string"},
									"header": {"type": "string"},
									"multiSelect": {"type": "boolean", "default": false},
									"options": {
										"type": "array",
										"items": {
											"type": "object",
											"additionalProperties": false,
											"properties": {
												"label": {"type": "string"},
												"description": {"type": "string"},
												"preview": {"type": "string"}
											},
											"required": ["label", "description"]
										},
										"minItems": 2,
										"maxItems": 4
									}
								},
								"required": ["question", "header", "options", "multiSelect"]
							},
							"minItems": 1,
							"maxItems": 4
						}
					},
					"required": ["questions"]
				}
			}
		]
	}`

	rm := NewRequestMapper(&Config{
		ModelName: "claude-sonnet-4-20250514",
		MaxTokens: 8096,
	})

	req, _, err := rm.FromNative([]byte(nativeRequest))
	require.NoError(t, err)
	require.Len(t, req.Tools, 3)

	apiReq, err := rm.ToProvider(req)
	require.NoError(t, err)
	require.Len(t, apiReq.Tools, 3)

	// Serialize to check the wire format
	toolsJSON, err := json.Marshal(apiReq.Tools)
	require.NoError(t, err)

	var tools []map[string]any
	require.NoError(t, json.Unmarshal(toolsJSON, &tools))

	t.Logf("Wire format: %s", string(toolsJSON))

	for _, tool := range tools {
		name, _ := tool["name"].(string)

		// Must NOT have "custom" wrapper
		_, hasCustom := tool["custom"]
		assert.False(t, hasCustom, "tool %s must not have 'custom' wrapper", name)

		// input_schema must be a valid object schema
		schema, ok := tool["input_schema"].(map[string]any)
		require.True(t, ok, "tool %s must have input_schema as object", name)
		assert.Equal(t, "object", schema["type"], "tool %s schema type", name)

	}
}

// TestFromNative_ToProvider_ToolSchemaRoundtrip verifies that tool input_schema
// survives the FromNative → ToProvider roundtrip without corruption.
// This is the exact path the AI Gateway takes: parse a native Anthropic request,
// then re-serialize it to call the Anthropic API via the SDK.
//
// Regression: Claude Code sends tools with JSON schemas that were mangled by
// AdaptSchemaForAnthropic / BetaToolInputSchema, causing Anthropic to reject
// the request with "tools.N.custom.input_schema: JSON schema is invalid".
func TestFromNative_ToProvider_ToolSchemaRoundtrip(t *testing.T) {
	nativeRequest := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "What is the weather in Berlin?"}
		],
		"tools": [
			{
				"name": "get_weather",
				"description": "Get current weather for a city",
				"input_schema": {
					"type": "object",
					"properties": {
						"city": {
							"type": "string",
							"description": "City name"
						},
						"units": {
							"type": "string",
							"enum": ["celsius", "fahrenheit"],
							"default": "celsius"
						}
					},
					"required": ["city"],
					"additionalProperties": false
				}
			},
			{
				"name": "read_file",
				"description": "Read a file from disk",
				"input_schema": {
					"type": "object",
					"properties": {
						"path": {
							"type": "string",
							"description": "Absolute file path"
						},
						"offset": {
							"type": "integer",
							"description": "Line offset"
						},
						"limit": {
							"type": "integer",
							"description": "Max lines to read"
						}
					},
					"required": ["path"]
				}
			}
		]
	}`

	rm := NewRequestMapper(&Config{
		ModelName: "claude-sonnet-4-20250514",
		MaxTokens: 1024,
	})

	// Step 1: Parse native request
	req, modelName, err := rm.FromNative([]byte(nativeRequest))
	require.NoError(t, err)
	assert.Equal(t, "claude-sonnet-4-20250514", modelName)
	require.Len(t, req.Tools, 2)

	// Step 2: Convert back to Anthropic API format (what model.Generate does internally)
	apiReq, err := rm.ToProvider(req)
	require.NoError(t, err)

	// Step 3: Verify tools survived the roundtrip
	require.Len(t, apiReq.Tools, 2)

	// Serialize to JSON to inspect the actual wire format
	toolsJSON, err := json.Marshal(apiReq.Tools)
	require.NoError(t, err)

	// The tool schemas must not be wrapped in a "custom" key or otherwise mangled.
	var tools []map[string]any
	require.NoError(t, json.Unmarshal(toolsJSON, &tools))

	for _, tool := range tools {
		// Each tool should have input_schema at the top level, not nested under "custom"
		schema, ok := tool["input_schema"].(map[string]any)
		if !ok {
			t.Fatalf("tool %v missing input_schema as object", tool["name"])
		}
		// Schema must have "type": "object" directly, not wrapped
		assert.Equal(t, "object", schema["type"], "tool %v schema type", tool["name"])
		// Must have "properties" directly
		_, hasProps := schema["properties"]
		assert.True(t, hasProps, "tool %v schema must have properties", tool["name"])
	}
}
