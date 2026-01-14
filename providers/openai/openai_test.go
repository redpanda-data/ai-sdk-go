package openai

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/responses"
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
			name:     "custom URL without /v1",
			input:    "https://custom-api.example.com",
			expected: "https://custom-api.example.com/v1",
		},
		{
			name:     "custom URL with /v1",
			input:    "https://custom-api.example.com/v1",
			expected: "https://custom-api.example.com/v1",
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
			inputURL:    "https://api.openai.com",
			expectedURL: "https://api.openai.com/v1",
		},
		{
			name:        "URL with /v1 stays unchanged",
			inputURL:    "https://api.openai.com/v1",
			expectedURL: "https://api.openai.com/v1",
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
	assert.NotEmpty(t, models, "Should return available models")

	// Collect model names for verification
	modelNames := make([]string, len(models))
	for i, model := range models {
		modelNames[i] = model.Name
		assert.NotEmpty(t, model.Name, "Model name should not be empty")
		assert.NotEmpty(t, model.Label, "Model label should not be empty")
		assert.Equal(t, "openai", model.Provider, "Provider should be 'openai'")
	}

	// Verify expected models are present
	expectedModels := []string{"gpt-4o", "gpt-4o-mini", "o3", "gpt-5", "gpt-5.2"}
	for _, expected := range expectedModels {
		assert.Contains(t, modelNames, expected, "Should include %s", expected)
	}
}

func TestModelCreation(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	// Valid model creation
	model, err := provider.NewModel(ModelGPT5Mini)
	require.NoError(t, err)
	assert.NotNil(t, model)
	assert.Equal(t, ModelGPT5Mini, model.Name())

	// Valid model with options
	model, err = provider.NewModel(ModelGPT5Mini, WithTemperature(0.7), WithMaxTokens(100))
	require.NoError(t, err)
	assert.Equal(t, ModelGPT5Mini, model.Name())

	// Error cases
	_, err = provider.NewModel("nonexistent-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported OpenAI model")

	// Conflicting options
	_, err = provider.NewModel(ModelGPT5Mini, WithTemperature(0.7), WithTopP(0.9))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot use")
}

func TestModelConstraints(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	// O3 model constraints
	_, err = provider.NewModel(ModelO3, WithTemperature(1.0))
	require.NoError(t, err)

	_, err = provider.NewModel(ModelO3, WithTemperature(1.5)) // O3 max is 1.0
	require.Error(t, err)
	assert.Contains(t, err.Error(), "out of range")

	_, err = provider.NewModel(ModelO3, WithTopP(0.9)) // O3 doesn't support top_p
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not supported")
}

func TestModelCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	// GPT-5-mini capabilities
	model, err := provider.NewModel(ModelGPT5Mini)
	require.NoError(t, err)

	caps := model.Capabilities()
	assert.True(t, caps.Streaming)
	assert.True(t, caps.Tools)
	assert.True(t, caps.Vision)
	assert.True(t, caps.Audio)

	// O3 capabilities (different from GPT models)
	model, err = provider.NewModel(ModelO3)
	require.NoError(t, err)

	caps = model.Capabilities()
	assert.True(t, caps.Vision)
	assert.False(t, caps.Audio)
	assert.False(t, caps.StructuredOutput)
}

func TestRequestMapping(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	model, err := provider.NewModel(ModelGPT5Mini)
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
		name              string
		status            string
		incompleteDetails responses.ResponseIncompleteDetails
		expected          llm.FinishReason
	}{
		{
			name:     "completed status",
			status:   "completed",
			expected: llm.FinishReasonStop,
		},
		{
			name:   "incomplete with max_output_tokens",
			status: "incomplete",
			incompleteDetails: responses.ResponseIncompleteDetails{
				Reason: "max_output_tokens",
			},
			expected: llm.FinishReasonLength,
		},
		{
			name:   "incomplete with content_filter",
			status: "incomplete",
			incompleteDetails: responses.ResponseIncompleteDetails{
				Reason: "content_filter",
			},
			expected: llm.FinishReasonContentFilter,
		},
		{
			name:   "incomplete with unknown reason",
			status: "incomplete",
			incompleteDetails: responses.ResponseIncompleteDetails{
				Reason: "unknown_reason",
			},
			expected: llm.FinishReasonUnknown,
		},
		{
			name:     "unknown status",
			status:   "unknown_status",
			expected: llm.FinishReasonUnknown,
		},
		{
			name:     "empty status",
			status:   "",
			expected: llm.FinishReasonUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := mapper.mapFinishReasonFromStatus(tt.status, tt.incompleteDetails)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestToolMapping(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	model, err := provider.NewModel(ModelGPT4OMini)
	require.NoError(t, err)

	m, _ := model.(*Model) // Type assertion for testing internal methods

	tests := []struct {
		name     string
		tools    []llm.ToolDefinition
		wantErr  bool
		validate func(t *testing.T, tools []responses.ToolUnionParam)
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
			validate: func(t *testing.T, tools []responses.ToolUnionParam) {
				t.Helper()
				require.Len(t, tools, 1)

				tool := tools[0]
				require.NotNil(t, tool.OfFunction)
				assert.Equal(t, "get_weather", tool.OfFunction.Name)
				assert.True(t, tool.OfFunction.Strict.Value)
				assert.Equal(t, "Get current weather information for a location", tool.OfFunction.Description.Value)

				// Verify parameters structure
				params := tool.OfFunction.Parameters
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
			validate: func(t *testing.T, tools []responses.ToolUnionParam) {
				t.Helper()
				require.Len(t, tools, 1)

				tool := tools[0]
				require.NotNil(t, tool.OfFunction)
				assert.Equal(t, "get_current_time", tool.OfFunction.Name)

				// Should have default empty object schema
				params := tool.OfFunction.Parameters
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
			validate: func(t *testing.T, tools []responses.ToolUnionParam) {
				t.Helper()
				require.Len(t, tools, 2)
				assert.Equal(t, "tool_one", tools[0].OfFunction.Name)
				assert.Equal(t, "tool_two", tools[1].OfFunction.Name)
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
			validate: func(t *testing.T, tools []responses.ToolUnionParam) {
				t.Helper()
				require.Len(t, tools, 1)

				tool := tools[0]
				require.NotNil(t, tool.OfFunction)
				assert.Equal(t, "simple_tool", tool.OfFunction.Name)
				// Description should be empty string when not provided
				assert.Empty(t, tool.OfFunction.Description.Value)
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

	model, err := provider.NewModel(ModelGPT4OMini)
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
	assert.Equal(t, "get_weather", tool.OfFunction.Name)
	assert.Equal(t, "Get current weather information for a location", tool.OfFunction.Description.Value)
}

func TestMessageMappingWithToolParts(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	model, err := provider.NewModel(ModelGPT4OMini)
	require.NoError(t, err)

	m, _ := model.(*Model) // Type assertion for testing internal methods

	tests := []struct {
		name     string
		messages []llm.Message
		wantErr  bool
		validate func(t *testing.T, items []responses.ResponseInputItemUnionParam)
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
			validate: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				t.Helper()
				require.Len(t, items, 1)
				require.NotNil(t, items[0].OfMessage)
				assert.Equal(t, "Hello!", items[0].OfMessage.Content.OfString.Value)
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
			validate: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				t.Helper()
				require.Len(t, items, 2) // Text message + tool request

				// First item should be the text message
				require.NotNil(t, items[0].OfMessage)
				assert.Equal(t, "I'll check the weather for you.", items[0].OfMessage.Content.OfString.Value)

				// Second item should be the tool request
				require.NotNil(t, items[1].OfFunctionCall)
				assert.Equal(t, "call_123", items[1].OfFunctionCall.CallID)
				assert.Equal(t, "get_weather", items[1].OfFunctionCall.Name)
				assert.JSONEq(t, `{"location": "Paris, France"}`, items[1].OfFunctionCall.Arguments)
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
			validate: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				t.Helper()
				require.Len(t, items, 1)

				require.NotNil(t, items[0].OfFunctionCallOutput)
				assert.Equal(t, "call_123", items[0].OfFunctionCallOutput.CallID)
				assert.JSONEq(t, `{"temperature": "22°C", "condition": "sunny"}`, items[0].OfFunctionCallOutput.Output.OfString.Value)
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
			validate: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				t.Helper()
				require.Len(t, items, 1)

				require.NotNil(t, items[0].OfFunctionCallOutput)
				assert.Equal(t, "call_123", items[0].OfFunctionCallOutput.CallID)
				assert.Contains(t, items[0].OfFunctionCallOutput.Output.OfString.Value, "API rate limit exceeded")
			},
		},
		{
			name: "reasoning trace",
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
			validate: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				t.Helper()
				require.Len(t, items, 1)

				require.NotNil(t, items[0].OfReasoning)
				require.Len(t, items[0].OfReasoning.Summary, 1)
				assert.Equal(t, "Let me think about this step by step...", items[0].OfReasoning.Summary[0].Text)
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
			validate: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				t.Helper()
				require.Len(t, items, 4) // User text + Assistant text + Tool request + Tool response

				// User message
				require.NotNil(t, items[0].OfMessage)
				assert.Equal(t, "Check the weather", items[0].OfMessage.Content.OfString.Value)

				// Assistant text
				require.NotNil(t, items[1].OfMessage)
				assert.Equal(t, "I'll check that for you.", items[1].OfMessage.Content.OfString.Value)

				// Tool request
				require.NotNil(t, items[2].OfFunctionCall)
				assert.Equal(t, "call_456", items[2].OfFunctionCall.CallID)

				// Tool response
				require.NotNil(t, items[3].OfFunctionCallOutput)
				assert.Equal(t, "call_456", items[3].OfFunctionCallOutput.CallID)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			items, err := m.requestMapper.mapMessagesToInputItems(tt.messages)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)

			if tt.validate != nil {
				tt.validate(t, items)
			}
		})
	}
}

func TestGPT52ReasoningEffort(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	tests := []struct {
		name          string
		model         string
		reasoningOpts []Option
		wantErr       bool
		errContains   string
	}{
		{
			name:          "gpt-5.2 with ReasoningEffortNone (supported)",
			model:         ModelGPT5_2,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortNone)},
			wantErr:       false,
		},
		{
			name:          "gpt-5.2 with ReasoningEffortLow (supported)",
			model:         ModelGPT5_2,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortLow)},
			wantErr:       false,
		},
		{
			name:          "gpt-5.2 with ReasoningEffortMedium (supported)",
			model:         ModelGPT5_2,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortMedium)},
			wantErr:       false,
		},
		{
			name:          "gpt-5.2 with ReasoningEffortHigh (supported)",
			model:         ModelGPT5_2,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortHigh)},
			wantErr:       false,
		},
		{
			name:          "gpt-5.2 with ReasoningEffortMinimal (unsupported)",
			model:         ModelGPT5_2,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortMinimal)},
			wantErr:       true,
			errContains:   "does not support reasoning effort 'minimal'",
		},
		{
			name:          "gpt-5.2-pro with ReasoningEffortNone (unsupported)",
			model:         ModelGPT5_2Pro,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortNone)},
			wantErr:       true,
			errContains:   "does not support reasoning effort 'none'",
		},
		{
			name:          "gpt-5.1 with ReasoningEffortNone (supported)",
			model:         ModelGPT5_1,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortNone)},
			wantErr:       false,
		},
		{
			name:          "gpt-5.1 with ReasoningEffortMinimal (unsupported)",
			model:         ModelGPT5_1,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortMinimal)},
			wantErr:       true,
			errContains:   "does not support reasoning effort 'minimal'",
		},
		{
			name:          "gpt-5 with ReasoningEffortMinimal (supported for older models)",
			model:         ModelGPT5,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortMinimal)},
			wantErr:       false,
		},
		{
			name:          "o3 with ReasoningEffortLow (supported)",
			model:         ModelO3,
			reasoningOpts: []Option{WithReasoningEffort(ReasoningEffortLow)},
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, err := provider.NewModel(tt.model, tt.reasoningOpts...)

			if tt.wantErr {
				require.Error(t, err)

				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}

				return
			}

			require.NoError(t, err)
			assert.NotNil(t, model)

			// Verify the model has reasoning capability
			assert.True(t, model.Capabilities().Reasoning, "Model should support reasoning")
		})
	}
}

func TestMessageRoleValidation(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	model, err := provider.NewModel(ModelGPT4OMini)
	require.NoError(t, err)

	m, _ := model.(*Model) // Type assertion for testing internal methods

	tests := []struct {
		name        string
		messages    []llm.Message
		wantErr     bool
		errContains string
	}{
		{
			name: "tool response with wrong role (RoleAssistant)",
			messages: []llm.Message{
				{
					Role: llm.RoleAssistant,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     "call_123",
							Name:   "get_weather",
							Result: json.RawMessage(`{"temp": 72}`),
						}),
					},
				},
			},
			wantErr:     true,
			errContains: "tool response parts require RoleUser",
		},
		{
			name: "tool request with wrong role (RoleUser)",
			messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewToolRequestPart(&llm.ToolRequest{
							ID:        "call_123",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"location": "Paris"}`),
						}),
					},
				},
			},
			wantErr:     true,
			errContains: "tool request parts require RoleAssistant",
		},
		{
			name: "multiple tool responses in RoleUser message (valid)",
			messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     "call_123",
							Name:   "get_weather",
							Result: json.RawMessage(`{"temp": 72}`),
						}),
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     "call_456",
							Name:   "get_time",
							Result: json.RawMessage(`{"time": "12:00"}`),
						}),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "tool response in correct role (valid)",
			messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     "call_123",
							Name:   "get_weather",
							Result: json.RawMessage(`{"temp": 72}`),
						}),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "tool request in correct role (valid)",
			messages: []llm.Message{
				{
					Role: llm.RoleAssistant,
					Content: []*llm.Part{
						llm.NewToolRequestPart(&llm.ToolRequest{
							ID:        "call_123",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"location": "Paris"}`),
						}),
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := m.requestMapper.mapMessagesToInputItems(tt.messages)

			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
				assert.ErrorIs(t, err, llm.ErrRequestMapping)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
