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

package openaicompat

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestNormalizeBaseURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "URL without /v1",
			input:    "https://api.openai.com",
			expected: "https://api.openai.com/v1",
		},
		{
			name:     "URL with /v1",
			input:    "https://api.openai.com/v1",
			expected: "https://api.openai.com/v1",
		},
		{
			name:     "URL with trailing slash",
			input:    "https://api.openai.com/",
			expected: "https://api.openai.com/v1",
		},
		{
			name:     "URL with /v1 and trailing slash",
			input:    "https://api.openai.com/v1/",
			expected: "https://api.openai.com/v1",
		},
		{
			name:     "custom URL without /v1 (e.g., DeepSeek)",
			input:    "https://api.deepseek.com",
			expected: "https://api.deepseek.com/v1",
		},
		{
			name:     "custom URL with /v1",
			input:    "https://api.deepseek.com/v1",
			expected: "https://api.deepseek.com/v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := normalizeBaseURL(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestWithBaseURLNormalization(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		inputURL    string
		expectedURL string
	}{
		{
			name:        "URL without /v1 gets normalized",
			inputURL:    "https://api.deepseek.com",
			expectedURL: "https://api.deepseek.com/v1",
		},
		{
			name:        "URL with /v1 stays unchanged",
			inputURL:    "https://api.deepseek.com/v1",
			expectedURL: "https://api.deepseek.com/v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider, err := NewProvider("sk-test-key", WithBaseURL(tt.inputURL))
			require.NoError(t, err)
			assert.Equal(t, tt.expectedURL, provider.BaseURL)
		})
	}
}

func TestProviderCreation(t *testing.T) {
	t.Parallel()
	// Valid provider creation
	provider, err := NewProvider("sk-test-key-123")
	require.NoError(t, err)
	assert.NotNil(t, provider)
	assert.NotNil(t, provider.client)

	// Empty API key should fail
	_, err = NewProvider("")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "API key is required")

	// Invalid options should fail
	_, err = NewProvider("sk-test", WithTimeout(-1*time.Second))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "timeout must be positive")
}

func TestProviderModels(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	models := provider.Models()
	// openaicompat returns empty list - it supports dynamic model names
	// determined by the OpenAI-compatible API endpoint being used
	assert.Empty(t, models, "openaicompat should return empty model list - supports any model name")
}

func TestModelCreation(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	// Valid model creation with any name (openaicompat is flexible)
	model, err := provider.NewModel("gpt-4o-mini")
	require.NoError(t, err)
	assert.NotNil(t, model)
	assert.Equal(t, "gpt-4o-mini", model.Name())

	// Valid model with options
	model, err = provider.NewModel("gpt-4o-mini", WithTemperature(0.7), WithMaxTokens(100))
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o-mini", model.Name())

	// Empty model name should fail
	_, err = provider.NewModel("")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model name cannot be empty")
}

func TestModelConstraints(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	// openaicompat uses permissive constraints - should accept wide range
	_, err = provider.NewModel("any-model", WithTemperature(1.0))
	require.NoError(t, err)

	_, err = provider.NewModel("any-model", WithTemperature(1.5))
	require.NoError(t, err) // Permissive constraints allow 0-2

	// Temperature out of permissive range should still fail
	_, err = provider.NewModel("any-model", WithTemperature(3.0))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "out of range")
}

func TestModelCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	// openaicompat uses default capabilities for all models
	model, err := provider.NewModel("any-model")
	require.NoError(t, err)

	caps := model.Capabilities()
	assert.True(t, caps.Streaming)
	assert.True(t, caps.Tools)
	assert.True(t, caps.Vision)
	assert.True(t, caps.StructuredOutput)
	assert.False(t, caps.Audio)     // Audio not commonly supported in Chat API
	assert.False(t, caps.Reasoning) // Reasoning is model-specific, not general
}

func TestRequestMapping(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	model, err := provider.NewModel("gpt-4o-mini")
	require.NoError(t, err)

	m, _ := model.(*Model) // Type assertion for testing internal methods

	// Basic request mapping
	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("Hello!"),
				},
			},
		},
	}

	mappedReq, err := m.requestMapper.ToProvider(req)
	require.NoError(t, err)
	assert.NotNil(t, mappedReq)
}

func TestFinishReasonMapping(t *testing.T) {
	t.Parallel()

	mapper := &ResponseMapper{}

	tests := []struct {
		name         string
		reason       string
		hasToolCalls bool
		expected     llm.FinishReason
	}{
		{
			name:     "stop reason",
			reason:   "stop",
			expected: llm.FinishReasonStop,
		},
		{
			name:     "length reason",
			reason:   "length",
			expected: llm.FinishReasonLength,
		},
		{
			name:     "content_filter reason",
			reason:   "content_filter",
			expected: llm.FinishReasonContentFilter,
		},
		{
			name:     "tool_calls reason",
			reason:   "tool_calls",
			expected: llm.FinishReasonToolCalls,
		},
		{
			name:         "has tool calls overrides reason",
			reason:       "stop",
			hasToolCalls: true,
			expected:     llm.FinishReasonToolCalls,
		},
		{
			name:     "unknown reason",
			reason:   "unknown_reason",
			expected: llm.FinishReasonUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result, _ := mapper.mapFinishReason(tt.reason, tt.hasToolCalls)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestToolMapping(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	model, err := provider.NewModel("gpt-4o-mini")
	require.NoError(t, err)

	m, _ := model.(*Model) // Type assertion for testing internal methods

	tests := []struct {
		name     string
		tools    []llm.ToolDefinition
		wantErr  bool
		validate func(t *testing.T, tools []openai.ChatCompletionToolUnionParam)
	}{
		{
			name: "single tool with parameters",
			tools: []llm.ToolDefinition{
				{
					Name:        "get_weather",
					Description: "Get current weather information for a location",
					Parameters: json.RawMessage(`{
						"type": "object",
						"properties": {
							"location": {
								"type": "string",
								"description": "The city and state/country"
							}
						},
						"required": ["location"]
					}`),
				},
			},
			wantErr: false,
			validate: func(t *testing.T, tools []openai.ChatCompletionToolUnionParam) {
				t.Helper()
				require.Len(t, tools, 1)

				tool := tools[0]
				require.NotNil(t, tool.OfFunction)
				assert.Equal(t, "get_weather", tool.OfFunction.Function.Name)
				assert.True(t, tool.OfFunction.Function.Strict.Value)
				assert.Equal(t, "Get current weather information for a location", tool.OfFunction.Function.Description.Value)

				// Verify parameters structure
				params := tool.OfFunction.Function.Parameters
				assert.Equal(t, "object", params["type"])
				properties, ok := params["properties"].(map[string]any)
				require.True(t, ok)
				location, ok := properties["location"].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, "string", location["type"])
			},
		},
		{
			name: "tool without parameters",
			tools: []llm.ToolDefinition{
				{
					Name:        "get_current_time",
					Description: "Get the current time",
				},
			},
			wantErr: false,
			validate: func(t *testing.T, tools []openai.ChatCompletionToolUnionParam) {
				t.Helper()
				require.Len(t, tools, 1)

				tool := tools[0]
				require.NotNil(t, tool.OfFunction)
				assert.Equal(t, "get_current_time", tool.OfFunction.Function.Name)

				// Should have default empty object schema
				params := tool.OfFunction.Function.Parameters
				assert.Equal(t, "object", params["type"])
				properties, ok := params["properties"].(map[string]any)
				require.True(t, ok)
				assert.Empty(t, properties)
			},
		},
		{
			name: "multiple tools",
			tools: []llm.ToolDefinition{
				{
					Name:        "tool_one",
					Description: "First tool",
					Parameters:  json.RawMessage(`{"type": "object", "properties": {}}`),
				},
				{
					Name:        "tool_two",
					Description: "Second tool",
					Parameters:  json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
			wantErr: false,
			validate: func(t *testing.T, tools []openai.ChatCompletionToolUnionParam) {
				t.Helper()
				require.Len(t, tools, 2)
				assert.Equal(t, "tool_one", tools[0].OfFunction.Function.Name)
				assert.Equal(t, "tool_two", tools[1].OfFunction.Function.Name)
			},
		},
		{
			name: "tool without description",
			tools: []llm.ToolDefinition{
				{
					Name:       "simple_tool",
					Parameters: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
			wantErr: false,
			validate: func(t *testing.T, tools []openai.ChatCompletionToolUnionParam) {
				t.Helper()
				require.Len(t, tools, 1)

				tool := tools[0]
				require.NotNil(t, tool.OfFunction)
				assert.Equal(t, "simple_tool", tool.OfFunction.Function.Name)
				// Description should be empty string when not provided
				assert.Empty(t, tool.OfFunction.Function.Description.Value)
			},
		},
		{
			name: "invalid JSON parameters",
			tools: []llm.ToolDefinition{
				{
					Name:       "bad_tool",
					Parameters: json.RawMessage(`{invalid json}`),
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			tools, err := m.requestMapper.mapToolDefinitions(tt.tools)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)

			if tt.validate != nil {
				tt.validate(t, tools)
			}
		})
	}
}

func TestRequestMappingWithTools(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	model, err := provider.NewModel("gpt-4o-mini")
	require.NoError(t, err)

	m, _ := model.(*Model) // Type assertion for testing internal methods

	// Test request with tools
	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("What's the weather like in Paris?"),
				},
			},
		},
		Tools: []llm.ToolDefinition{
			{
				Name:        "get_weather",
				Description: "Get current weather information for a location",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"location": {"type": "string", "description": "The city and state/country"}
					},
					"required": ["location"]
				}`),
			},
		},
	}

	mappedReq, err := m.requestMapper.ToProvider(req)
	require.NoError(t, err)
	assert.NotNil(t, mappedReq)

	// Verify tools were mapped
	require.Len(t, mappedReq.Tools, 1)
	tool := mappedReq.Tools[0]
	require.NotNil(t, tool.OfFunction)
	assert.Equal(t, "get_weather", tool.OfFunction.Function.Name)
	assert.Equal(t, "Get current weather information for a location", tool.OfFunction.Function.Description.Value)
}

func TestMessageMappingWithToolParts(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	model, err := provider.NewModel("gpt-4o-mini")
	require.NoError(t, err)

	m, _ := model.(*Model) // Type assertion for testing internal methods

	tests := []struct {
		name     string
		messages []llm.Message
		wantErr  bool
		validate func(t *testing.T, messages []openai.ChatCompletionMessageParamUnion)
	}{
		{
			name: "text message only",
			messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("Hello!"),
					},
				},
			},
			validate: func(t *testing.T, messages []openai.ChatCompletionMessageParamUnion) {
				t.Helper()
				require.Len(t, messages, 1)
				require.NotNil(t, messages[0].OfUser)
				assert.Equal(t, "Hello!", messages[0].OfUser.Content.OfString.Value)
			},
		},
		{
			name: "assistant message with tool request",
			messages: []llm.Message{
				{
					Role: llm.RoleAssistant,
					Content: []*llm.Part{
						llm.NewTextPart("I'll check the weather for you."),
						llm.NewToolRequestPart(&llm.ToolRequest{
							ID:        "call_123",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"location": "Paris, France"}`),
						}),
					},
				},
			},
			validate: func(t *testing.T, messages []openai.ChatCompletionMessageParamUnion) {
				t.Helper()
				require.Len(t, messages, 1) // Single assistant message with text and tool calls

				// Should be assistant message
				require.NotNil(t, messages[0].OfAssistant)
				assert.Equal(t, "I'll check the weather for you.", messages[0].OfAssistant.Content.OfString.Value)

				// With tool calls
				require.Len(t, messages[0].OfAssistant.ToolCalls, 1)
				assert.Equal(t, "call_123", messages[0].OfAssistant.ToolCalls[0].OfFunction.ID)
				assert.Equal(t, "get_weather", messages[0].OfAssistant.ToolCalls[0].OfFunction.Function.Name)
				assert.JSONEq(t, `{"location": "Paris, France"}`, messages[0].OfAssistant.ToolCalls[0].OfFunction.Function.Arguments)
			},
		},
		{
			name: "tool response message",
			messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     "call_123",
							Name:   "get_weather",
							Result: json.RawMessage(`{"temperature": "22°C", "condition": "sunny"}`),
						}),
					},
				},
			},
			validate: func(t *testing.T, messages []openai.ChatCompletionMessageParamUnion) {
				t.Helper()
				require.Len(t, messages, 1)

				require.NotNil(t, messages[0].OfTool)
				assert.Equal(t, "call_123", messages[0].OfTool.ToolCallID)
				assert.JSONEq(t, `{"temperature": "22°C", "condition": "sunny"}`, messages[0].OfTool.Content.OfString.Value)
			},
		},
		{
			name: "tool response with error",
			messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:    "call_123",
							Name:  "get_weather",
							Error: "API rate limit exceeded",
						}),
					},
				},
			},
			validate: func(t *testing.T, messages []openai.ChatCompletionMessageParamUnion) {
				t.Helper()
				require.Len(t, messages, 1)

				require.NotNil(t, messages[0].OfTool)
				assert.Equal(t, "call_123", messages[0].OfTool.ToolCallID)
				assert.Contains(t, messages[0].OfTool.Content.OfString.Value, "API rate limit exceeded")
			},
		},
		{
			name: "reasoning trace (skipped in Chat API)",
			messages: []llm.Message{
				{
					Role: llm.RoleAssistant,
					Content: []*llm.Part{
						llm.NewReasoningPart(&llm.ReasoningTrace{
							ID:   "reasoning_123",
							Text: "Let me think about this step by step...",
						}),
					},
				},
			},
			validate: func(t *testing.T, messages []openai.ChatCompletionMessageParamUnion) {
				t.Helper()
				// Chat Completion API doesn't support reasoning traces, so they're skipped
				// The message should be empty since it only had reasoning content
				require.Empty(t, messages)
			},
		},
		{
			name: "mixed content types",
			messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("Check the weather"),
					},
				},
				{
					Role: llm.RoleAssistant,
					Content: []*llm.Part{
						llm.NewTextPart("I'll check that for you."),
						llm.NewToolRequestPart(&llm.ToolRequest{
							ID:        "call_456",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"location": "London"}`),
						}),
					},
				},
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     "call_456",
							Name:   "get_weather",
							Result: json.RawMessage(`{"temperature": "15°C"}`),
						}),
					},
				},
			},
			validate: func(t *testing.T, messages []openai.ChatCompletionMessageParamUnion) {
				t.Helper()
				require.Len(t, messages, 3) // User + Assistant + Tool

				// User message
				require.NotNil(t, messages[0].OfUser)
				assert.Equal(t, "Check the weather", messages[0].OfUser.Content.OfString.Value)

				// Assistant message with tool call
				require.NotNil(t, messages[1].OfAssistant)
				assert.Equal(t, "I'll check that for you.", messages[1].OfAssistant.Content.OfString.Value)
				require.Len(t, messages[1].OfAssistant.ToolCalls, 1)
				assert.Equal(t, "call_456", messages[1].OfAssistant.ToolCalls[0].OfFunction.ID)

				// Tool response
				require.NotNil(t, messages[2].OfTool)
				assert.Equal(t, "call_456", messages[2].OfTool.ToolCallID)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			messages, err := m.requestMapper.mapMessages(tt.messages)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)

			if tt.validate != nil {
				tt.validate(t, messages)
			}
		})
	}
}
