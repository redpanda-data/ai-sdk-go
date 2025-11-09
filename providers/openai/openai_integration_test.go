package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
)

const (
	// Test timeout for API calls.
	testTimeout = 30 * time.Second
	// Longer timeout for reasoning models which can take more time.
	reasoningTestTimeout = 5 * 60 * time.Second
)

// IntegrationTestSuite provides a test suite for OpenAI provider integration tests.
type IntegrationTestSuite struct {
	suite.Suite

	provider *openai.Provider
	apiKey   string
}

//nolint:paralleltest // Test suite manages its own lifecycle
func TestOpenAIIntegrationSuite(t *testing.T) {
	suite.Run(t, &IntegrationTestSuite{})
}

func (s *IntegrationTestSuite) SetupSuite() {
	s.apiKey = openaitest.GetAPIKeyOrSkipTest(s.T())

	provider, err := openai.NewProvider(s.apiKey)
	s.Require().NoError(err)
	s.provider = provider
}

func (s *IntegrationTestSuite) TestProviderCreation() {
	tests := []struct {
		name     string
		apiKey   string
		options  []openai.ProviderOption
		validate func(t *testing.T, provider *openai.Provider, err error)
	}{
		{
			name:   "valid API key",
			apiKey: s.apiKey,
			validate: func(t *testing.T, provider *openai.Provider, err error) {
				t.Helper()
				require.NoError(t, err)
				assert.NotNil(t, provider)
				assert.Equal(t, s.apiKey, provider.APIKey)
			},
		},
		{
			name:   "with custom timeout",
			apiKey: s.apiKey,
			options: []openai.ProviderOption{
				openai.WithTimeout(45 * time.Second),
			},
			validate: func(t *testing.T, provider *openai.Provider, err error) {
				t.Helper()
				require.NoError(t, err)
				assert.NotNil(t, provider)
				assert.Equal(t, 45*time.Second, provider.Timeout)
			},
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			provider, err := openai.NewProvider(tt.apiKey, tt.options...)
			tt.validate(s.T(), provider, err)
		})
	}
}

func (s *IntegrationTestSuite) TestGenerate() {
	tests := []struct {
		name     string
		request  *llm.Request
		validate func(t *testing.T, response *llm.Response, err error)
	}{
		{
			name: "simple text generation",
			request: &llm.Request{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("Say 'Hello, World!' and nothing else."),
						},
					},
				},
			},
			validate: func(t *testing.T, response *llm.Response, _ error) {
				t.Helper()
				require.NotNil(t, response)
				require.NotEmpty(t, response.Message.Content)

				assert.Equal(t, llm.FinishReasonStop, response.FinishReason)

				// Verify we have text content
				foundText := false

				for _, part := range response.Message.Content {
					if part.Kind == llm.PartText {
						foundText = true

						assert.Contains(t, part.Text, "Hello, World!")

						break
					}
				}

				assert.True(t, foundText, "Should contain text content")

				// Verify usage information
				require.NotNil(t, response.Usage)
				assert.Positive(t, response.Usage.InputTokens)
				assert.Positive(t, response.Usage.OutputTokens)
				assert.Positive(t, response.Usage.TotalTokens)
			},
		},
		{
			name: "conversation with system message",
			request: &llm.Request{
				Messages: []llm.Message{
					{
						Role: llm.RoleSystem,
						Content: []*llm.Part{
							llm.NewTextPart("You are a helpful assistant that responds with exactly one word."),
						},
					},
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("What color is the sky?"),
						},
					},
				},
			},
			validate: func(t *testing.T, response *llm.Response, _ error) {
				t.Helper()
				require.NotNil(t, response)
				require.NotEmpty(t, response.Message.Content)

				assert.Equal(t, llm.FinishReasonStop, response.FinishReason)

				// Should have usage information
				require.NotNil(t, response.Usage)
				assert.Positive(t, response.Usage.TotalTokens)
			},
		},
		{
			name: "with temperature setting",
			request: &llm.Request{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("Generate a random number between 1 and 10."),
						},
					},
				},
			},
			validate: func(t *testing.T, response *llm.Response, _ error) {
				t.Helper()
				require.NotNil(t, response)
				require.NotEmpty(t, response.Message.Content)

				assert.Equal(t, llm.FinishReasonStop, response.FinishReason)
			},
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
			defer cancel()

			model, err := s.provider.NewModel(openaitest.TestModelName)
			s.Require().NoError(err)

			response, err := model.Generate(ctx, tt.request)
			tt.validate(s.T(), response, err)
		})
	}
}

func (s *IntegrationTestSuite) TestGenerateStream() {
	tests := []struct {
		name     string
		request  *llm.Request
		validate func(t *testing.T, stream llm.EventStream, err error)
	}{
		{
			name: "simple streaming",
			request: &llm.Request{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("Count from 1 to 5, one number per sentence."),
						},
					},
				},
			},
			validate: func(t *testing.T, stream llm.EventStream, _ error) {
				t.Helper()

				require.NotNil(t, stream)

				defer func() { _ = stream.Close() }()

				var (
					textParts   []string
					endEvent    llm.StreamEndEvent
					hasEndEvent bool
				)

				// Collect all stream events

				for {
					event, streamErr := stream.Recv()
					if errors.Is(streamErr, io.EOF) {
						break
					}

					require.NoError(t, streamErr)

					switch e := event.(type) {
					case llm.ContentPartEvent:
						if e.Part.Kind == llm.PartText {
							textParts = append(textParts, e.Part.Text)
						}
					case llm.StreamEndEvent:
						endEvent = e
						hasEndEvent = true
					case llm.ErrorEvent:
						t.Fatalf("Received error event: %s (code: %s)", e.Message, e.Code)
					}
				}

				// Verify we received content
				assert.NotEmpty(t, textParts, "Should receive text content")

				// Verify we got an end event
				assert.True(t, hasEndEvent, "Should receive stream end event")
				assert.Equal(t, llm.FinishReasonStop, endEvent.Response.FinishReason)

				// Verify usage information in end event
				require.NotNil(t, endEvent.Response.Usage)
				assert.Positive(t, endEvent.Response.Usage.InputTokens)
				assert.Positive(t, endEvent.Response.Usage.OutputTokens)
				assert.Positive(t, endEvent.Response.Usage.TotalTokens)
			},
		},
		{
			name: "short response streaming",
			request: &llm.Request{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("Say only 'OK'"),
						},
					},
				},
			},
			validate: func(t *testing.T, stream llm.EventStream, _ error) {
				t.Helper()

				require.NotNil(t, stream)

				defer func() { _ = stream.Close() }()

				eventCount := 0

				var (
					hasContent  bool
					hasEndEvent bool
				)

				for {
					event, streamErr := stream.Recv()
					if errors.Is(streamErr, io.EOF) {
						break
					}

					require.NoError(t, streamErr)

					eventCount++

					switch e := event.(type) {
					case llm.ContentPartEvent:
						if e.Part.Kind == llm.PartText {
							hasContent = true
						}
					case llm.StreamEndEvent:
						hasEndEvent = true

						assert.Equal(t, llm.FinishReasonStop, e.Response.FinishReason)
					case llm.ErrorEvent:
						t.Fatalf("Received error event: %s", e.Message)
					}
				}

				assert.Positive(t, eventCount, "Should receive at least one event")
				assert.True(t, hasContent, "Should receive content")
				assert.True(t, hasEndEvent, "Should receive end event")
			},
		},
		{
			name: "stream with system message",
			request: &llm.Request{
				Messages: []llm.Message{
					{
						Role: llm.RoleSystem,
						Content: []*llm.Part{
							llm.NewTextPart("You are a helpful assistant."),
						},
					},
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("What is 2+2?"),
						},
					},
				},
			},
			validate: func(t *testing.T, stream llm.EventStream, _ error) {
				t.Helper()

				require.NotNil(t, stream)

				defer func() { _ = stream.Close() }()

				var (
					hasContent bool
					endEvent   llm.StreamEndEvent
				)

				for {
					event, streamErr := stream.Recv()
					if errors.Is(streamErr, io.EOF) {
						break
					}

					require.NoError(t, streamErr)

					switch e := event.(type) {
					case llm.ContentPartEvent:
						if e.Part.Kind == llm.PartText {
							hasContent = true
						}
					case llm.StreamEndEvent:
						endEvent = e
					case llm.ErrorEvent:
						t.Fatalf("Received error event: %s", e.Message)
					}
				}

				assert.True(t, hasContent, "Should receive content")
				assert.Equal(t, llm.FinishReasonStop, endEvent.Response.FinishReason)
				require.NotNil(t, endEvent.Response.Usage)
				assert.Positive(t, endEvent.Response.Usage.TotalTokens)
			},
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
			defer cancel()

			model, err := s.provider.NewModel(openaitest.TestModelName)
			s.Require().NoError(err)

			stream, err := model.GenerateStream(ctx, tt.request)
			tt.validate(s.T(), stream, err)
		})
	}
}

func (s *IntegrationTestSuite) TestGenerateWithReasoning() {
	s.Run("complex reasoning question", func() {
		ctx, cancel := context.WithTimeout(context.Background(), reasoningTestTimeout)
		defer cancel()

		model, err := s.provider.NewModel(openaitest.TestReasoningModelName,
			openai.WithReasoningEffort(openai.ReasoningEffortHigh),
			openai.WithReasoningSummary(openai.ReasoningSummaryDetailed),
		)
		s.Require().NoError(err)

		// Verify the model has reasoning capability
		caps := model.Capabilities()
		s.Require().True(caps.Reasoning, "Test model should support reasoning")

		request := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("Design a distributed consensus algorithm that can handle both network partitions and Byzantine failures. Explain the key trade-offs and why existing solutions like Raft or PBFT might not be sufficient."),
					},
				},
			},
		}

		response, err := model.Generate(ctx, request)
		s.Require().NoError(err)
		s.Require().NotNil(response)
		s.Require().NotEmpty(response.Message.Content)
		s.Greaterf(len(response.Message.Content), 1, "Should have at least two content parts, one for reasoning and one for the text output")

		// 1. Check if reasoning worked
		hasReasoning := false

		for _, part := range response.Message.Content {
			if part.IsReasoning() {
				hasReasoning = true

				s.NotEmpty(part.ReasoningTrace.Text)
				s.NotEmpty(part.ReasoningTrace.ID, "Reasoning trace should have ID")
				// Complex technical question should trigger substantial reasoning
				s.Greater(len(part.ReasoningTrace.Text), 30, "Should show detailed reasoning process")
			}
		}

		s.True(hasReasoning, "Complex technical question should trigger reasoning")

		// 2. Check Text response
		text := response.TextContent()
		// Should be a comprehensive response
		s.Greater(len(text), 200, "Should provide detailed technical analysis")
		// Should mention relevant concepts
		lowerText := strings.ToLower(text)
		s.True(
			strings.Contains(lowerText, "consensus") ||
				strings.Contains(lowerText, "byzantine") ||
				strings.Contains(lowerText, "partition"),
			"Should discuss relevant distributed systems concepts")

		// Usage should be significant for complex reasoning
		s.Require().NotNil(response.Usage)
		s.Greater(response.Usage.TotalTokens, 200, "Complex reasoning should use many tokens")
	})
}

func (s *IntegrationTestSuite) TestGenerateStreamWithReasoning() {
	s.Run("streaming with reasoning traces", func() {
		ctx, cancel := context.WithTimeout(context.Background(), reasoningTestTimeout)
		defer cancel()

		model, err := s.provider.NewModel(openaitest.TestReasoningModelName,
			openai.WithReasoningEffort(openai.ReasoningEffortHigh),
			openai.WithReasoningSummary(openai.ReasoningSummaryDetailed),
		)
		s.Require().NoError(err)

		// Verify the model has reasoning capability
		caps := model.Capabilities()
		s.Require().True(caps.Reasoning, "Test model should support reasoning")

		request := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("Design a distributed consensus algorithm that can handle both network partitions and Byzantine failures. Explain the key trade-offs and why existing solutions like Raft or PBFT might not be sufficient."),
					},
				},
			},
		}

		stream, err := model.GenerateStream(ctx, request)
		s.Require().NoError(err)

		s.Require().NotNil(stream)

		defer func() { _ = stream.Close() }()

		var (
			textParts      []string
			reasoningParts []string
			endEvent       llm.StreamEndEvent
			hasEndEvent    bool
		)

		eventCount := 0

		// Collect all stream events
		for {
			event, streamErr := stream.Recv()
			if errors.Is(streamErr, io.EOF) {
				break
			}

			s.Require().NoError(streamErr)

			eventCount++

			switch e := event.(type) {
			case llm.ContentPartEvent:
				if e.Part.IsText() {
					textParts = append(textParts, e.Part.Text)
				} else if e.Part.IsReasoning() {
					reasoningParts = append(reasoningParts, e.Part.ReasoningTrace.Text)
				}
			case llm.StreamEndEvent:
				endEvent = e
				hasEndEvent = true
			case llm.ErrorEvent:
				s.T().Fatalf("Received error event: %s (code: %s)", e.Message, e.Code)
			}
		}

		// Verify we received events
		s.Positive(eventCount, "Should receive stream events")

		// Verify we got text content
		s.NotEmpty(textParts, "Should receive text content chunks")

		// Combine text parts and check content quality (same as sync test)
		allText := ""

		var allTextSb502 strings.Builder
		for _, part := range textParts {
			allTextSb502.WriteString(part)
		}

		allText += allTextSb502.String()

		s.Greater(len(allText), 200, "Should provide detailed technical analysis")

		// Should mention relevant concepts (same check as sync test)
		lowerText := strings.ToLower(allText)
		s.True(
			strings.Contains(lowerText, "consensus") ||
				strings.Contains(lowerText, "byzantine") ||
				strings.Contains(lowerText, "partition"),
			"Should discuss relevant distributed systems concepts")

		// Verify reasoning content (should be present for complex question with high effort)
		s.NotEmpty(reasoningParts, "Should receive reasoning traces for complex question")

		// Combine reasoning parts to check total content
		allReasoning := ""

		var allReasoningSb520 strings.Builder
		for _, part := range reasoningParts {
			allReasoningSb520.WriteString(part)
		}

		allReasoning += allReasoningSb520.String()

		s.Greater(len(allReasoning), 50, "Should have substantial reasoning content")

		// Verify we got an end event
		s.True(hasEndEvent, "Should receive stream end event")
		s.Equal(llm.FinishReasonStop, endEvent.Response.FinishReason)

		// Verify usage information (same as sync test)
		s.Require().NotNil(endEvent.Response.Usage)
		s.Greater(endEvent.Response.Usage.TotalTokens, 200, "Complex reasoning should use many tokens")
	})
}

func (s *IntegrationTestSuite) TestAllSupportedModels() {
	t := s.T()

	t.Run("basic generation works for all supported models", func(t *testing.T) {
		models := s.provider.Models()

		for _, m := range models {
			modelName := m.Name
			t.Run("model_"+modelName, func(t *testing.T) {
				model, err := s.provider.NewModel(modelName)
				if err != nil {
					t.Skipf("Skipping model %s due to creation error: %v", modelName, err)
					return
				}

				reqObj := &llm.Request{
					Messages: []llm.Message{{
						Role:    llm.RoleUser,
						Content: []*llm.Part{llm.NewTextPart("Say 'Hello' in one word")},
					}},
				}

				resp, err := model.Generate(t.Context(), reqObj)
				require.NoError(t, err)

				require.NotNil(t, resp)
				assert.NotEmpty(t, resp.Message.Content)
				assert.NotEmpty(t, resp.TextContent())
				assert.NotEmpty(t, resp.FinishReason)

				if resp.Usage != nil {
					assert.Positive(t, resp.Usage.TotalTokens)
				}
			})
		}
	})
}

func (s *IntegrationTestSuite) TestStructuredOutputs() {
	s.Run("JSON Schema structured output", func() {
		model, err := s.provider.NewModel(openaitest.TestModelName)
		s.Require().NoError(err)

		// Define a JSON schema for a simple person object
		schema := `{
			"type": "object",
			"properties": {
				"name": {"type": "string"},
				"age": {"type": "integer", "minimum": 0},
				"city": {"type": "string"}
			},
			"required": ["name", "age", "city"],
			"additionalProperties": false
		}`

		req := &llm.Request{
			Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: []*llm.Part{llm.NewTextPart("Create a person with name John, age 25, living in New York")},
			}},
			ResponseFormat: &llm.ResponseFormat{
				Type: llm.ResponseFormatJSONSchema,
				JSONSchema: &llm.JSONSchema{
					Name:        "person",
					Description: "A person with basic information",
					Schema:      []byte(schema),
				},
			},
		}

		resp, err := model.Generate(s.T().Context(), req)
		s.Require().NoError(err)
		s.Require().NotNil(resp)

		// Get the text content
		textContent := resp.TextContent()
		s.NotEmpty(textContent)

		// Verify it's valid JSON that matches our schema
		var personData map[string]any

		err = json.Unmarshal([]byte(textContent), &personData)
		s.Require().NoError(err, "Response should be valid JSON")

		// Verify required fields are present
		s.Contains(personData, "name")
		s.Contains(personData, "age")
		s.Contains(personData, "city")

		// Verify field types
		s.IsType("", personData["name"])
		s.IsType(float64(0), personData["age"]) // JSON numbers are float64
		s.IsType("", personData["city"])
	})
}

func (s *IntegrationTestSuite) TestJSONObjectOutput() {
	s.Run("JSON Object mode", func() {
		model, err := s.provider.NewModel(openaitest.TestModelName)
		s.Require().NoError(err)

		req := &llm.Request{
			Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: []*llm.Part{llm.NewTextPart("List 3 colors in JSON format with their hex codes")},
			}},
			ResponseFormat: &llm.ResponseFormat{
				Type: llm.ResponseFormatJSONObject,
			},
		}

		resp, err := model.Generate(s.T().Context(), req)
		s.Require().NoError(err)
		s.Require().NotNil(resp)

		// Get the text content
		textContent := resp.TextContent()
		s.NotEmpty(textContent)

		// Verify it's valid JSON (but we don't enforce specific structure)
		var jsonData any

		err = json.Unmarshal([]byte(textContent), &jsonData)
		s.Require().NoError(err, "Response should be valid JSON")
	})
}

func (s *IntegrationTestSuite) TestGenerateStreamWithTools() {
	tests := []struct {
		name     string
		request  *llm.Request
		validate func(t *testing.T, stream llm.EventStream, err error)
	}{
		{
			name: "single tool call in stream",
			request: &llm.Request{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("What is the weather in San Francisco?"),
						},
					},
				},
				Tools: []llm.ToolDefinition{
					{
						Name:        "get_weather",
						Description: "Get the current weather in a given location",
						Parameters: json.RawMessage(`{
							"type": "object",
							"properties": {
								"location": {
									"type": "string",
									"description": "The city and state, e.g. San Francisco, CA"
								},
								"unit": {
									"type": "string",
									"enum": ["celsius", "fahrenheit"],
									"description": "The temperature unit to use"
								}
							},
							"required": ["location"]
						}`),
					},
				},
			},
			validate: func(t *testing.T, stream llm.EventStream, _ error) {
				t.Helper()

				require.NotNil(t, stream)

				defer func() { _ = stream.Close() }()

				var (
					toolRequests []*llm.ToolRequest
					endEvent     llm.StreamEndEvent
					hasEndEvent  bool
				)

				// Collect all stream events

				for {
					event, streamErr := stream.Recv()
					if errors.Is(streamErr, io.EOF) {
						break
					}

					require.NoError(t, streamErr)

					switch e := event.(type) {
					case llm.ContentPartEvent:
						if e.Part.IsToolRequest() {
							toolRequests = append(toolRequests, e.Part.ToolRequest)
						}
					case llm.StreamEndEvent:
						endEvent = e
						hasEndEvent = true
					case llm.ErrorEvent:
						t.Fatalf("Received error event: %s (code: %s)", e.Message, e.Code)
					}
				}

				// Verify we received exactly one tool request
				assert.Len(t, toolRequests, 1, "Should receive exactly one tool request")

				if len(toolRequests) > 0 {
					toolReq := toolRequests[0]

					// Verify tool name
					assert.Equal(t, "get_weather", toolReq.Name, "Tool name should match")

					// Verify tool ID is present
					assert.NotEmpty(t, toolReq.ID, "Tool request ID should be present")

					// Verify arguments are valid JSON
					var args map[string]any

					err := json.Unmarshal(toolReq.Arguments, &args)
					require.NoError(t, err, "Tool arguments should be valid JSON")

					// Verify location is present in arguments
					assert.Contains(t, args, "location", "Arguments should contain location")
					assert.NotEmpty(t, args["location"], "Location should not be empty")
				}

				// Verify we got an end event
				assert.True(t, hasEndEvent, "Should receive stream end event")
				assert.Equal(t, llm.FinishReasonToolCalls, endEvent.Response.FinishReason, "Finish reason should be ToolCalls")

				// Verify usage information in end event
				require.NotNil(t, endEvent.Response.Usage)
				assert.Positive(t, endEvent.Response.Usage.InputTokens)
				assert.Positive(t, endEvent.Response.Usage.TotalTokens)
			},
		},
		{
			name: "multiple tool calls in stream",
			request: &llm.Request{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("What is the weather in San Francisco and New York? Also, what time is it in Tokyo?"),
						},
					},
				},
				Tools: []llm.ToolDefinition{
					{
						Name:        "get_weather",
						Description: "Get the current weather in a given location",
						Parameters: json.RawMessage(`{
							"type": "object",
							"properties": {
								"location": {
									"type": "string",
									"description": "The city and state, e.g. San Francisco, CA"
								}
							},
							"required": ["location"]
						}`),
					},
					{
						Name:        "get_time",
						Description: "Get the current time in a given timezone",
						Parameters: json.RawMessage(`{
							"type": "object",
							"properties": {
								"timezone": {
									"type": "string",
									"description": "The timezone, e.g. America/New_York or Asia/Tokyo"
								}
							},
							"required": ["timezone"]
						}`),
					},
				},
			},
			validate: func(t *testing.T, stream llm.EventStream, _ error) {
				t.Helper()

				require.NotNil(t, stream)

				defer func() { _ = stream.Close() }()

				var (
					toolRequests []*llm.ToolRequest
					endEvent     llm.StreamEndEvent
					hasEndEvent  bool
				)

				toolRequestsByName := make(map[string][]*llm.ToolRequest)

				// Collect all stream events
				for {
					event, streamErr := stream.Recv()
					if errors.Is(streamErr, io.EOF) {
						break
					}

					require.NoError(t, streamErr)

					switch e := event.(type) {
					case llm.ContentPartEvent:
						if e.Part.IsToolRequest() {
							toolRequests = append(toolRequests, e.Part.ToolRequest)
							toolRequestsByName[e.Part.ToolRequest.Name] = append(
								toolRequestsByName[e.Part.ToolRequest.Name],
								e.Part.ToolRequest,
							)
						}
					case llm.StreamEndEvent:
						endEvent = e
						hasEndEvent = true
					case llm.ErrorEvent:
						t.Fatalf("Received error event: %s (code: %s)", e.Message, e.Code)
					}
				}

				// Verify we received multiple tool requests
				assert.Greater(t, len(toolRequests), 1, "Should receive multiple tool requests")

				// Verify each tool request has proper structure
				uniqueIDs := make(map[string]bool)

				for _, toolReq := range toolRequests {
					// Verify tool name is one of our defined tools
					assert.Contains(t, []string{"get_weather", "get_time"}, toolReq.Name,
						"Tool name should be one of the defined tools")

					// Verify tool ID is present and unique
					assert.NotEmpty(t, toolReq.ID, "Tool request ID should be present")
					assert.False(t, uniqueIDs[toolReq.ID], "Tool request IDs should be unique")
					uniqueIDs[toolReq.ID] = true

					// Verify arguments are valid JSON
					var args map[string]any

					err := json.Unmarshal(toolReq.Arguments, &args)
					require.NoError(t, err, "Tool arguments should be valid JSON for tool %s", toolReq.Name)

					// Verify tool-specific argument structure
					switch toolReq.Name {
					case "get_weather":
						assert.Contains(t, args, "location", "get_weather arguments should contain location")
						assert.NotEmpty(t, args["location"], "Location should not be empty")
					case "get_time":
						assert.Contains(t, args, "timezone", "get_time arguments should contain timezone")
						assert.NotEmpty(t, args["timezone"], "Timezone should not be empty")
					}
				}

				// Verify we got requests for different tools
				assert.NotEmpty(t, toolRequestsByName, "Should have tool requests grouped by name")

				// Verify we got an end event
				assert.True(t, hasEndEvent, "Should receive stream end event")
				assert.Equal(t, llm.FinishReasonToolCalls, endEvent.Response.FinishReason, "Finish reason should be ToolCalls")

				// Verify usage information in end event
				require.NotNil(t, endEvent.Response.Usage)
				assert.Positive(t, endEvent.Response.Usage.InputTokens)
				assert.Positive(t, endEvent.Response.Usage.TotalTokens)
			},
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
			defer cancel()

			model, err := s.provider.NewModel(openaitest.TestModelName)
			s.Require().NoError(err)

			stream, err := model.GenerateStream(ctx, tt.request)
			tt.validate(s.T(), stream, err)
		})
	}
}
