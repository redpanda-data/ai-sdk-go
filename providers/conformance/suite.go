package conformance

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	// testTimeout is the timeout for API calls.
	testTimeout = 30 * time.Second
	// reasoningTestTimeout is a longer timeout for reasoning models which can take more time.
	reasoningTestTimeout = 5 * 60 * time.Second
)

// Suite provides generic conformance tests for any provider implementing the llm.Model interface.
// Provider implementations should create their own test file that instantiates this suite with a
// provider-specific fixture.
//
// Usage:
//
//	func TestProviderConformance(t *testing.T) {
//	    fixture := NewMyProviderFixture(t)
//	    suite.Run(t, conformance.NewSuite(fixture))
//	}
type Suite struct {
	suite.Suite

	fixture Fixture
}

// NewSuite creates a new conformance test suite with the given fixture.
func NewSuite(fixture Fixture) *Suite {
	return &Suite{
		fixture: fixture,
	}
}

func (s *Suite) TestGenerate() {
	model := s.fixture.StandardModel()
	if model == nil {
		s.T().Skip("No standard model available")
	}

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

			response, err := model.Generate(ctx, tt.request)
			tt.validate(s.T(), response, err)
		})
	}
}

func (s *Suite) TestGenerateEvents() {
	model := s.fixture.StandardModel()
	if model == nil {
		s.T().Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.Streaming {
		s.T().Skip("Model does not support streaming")
	}

	tests := []struct {
		name     string
		request  *llm.Request
		validate func(t *testing.T, model llm.Model, ctx context.Context, request *llm.Request)
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
			validate: func(t *testing.T, model llm.Model, ctx context.Context, request *llm.Request) {
				t.Helper()

				var (
					contentParts []*llm.Part
					endEvent     llm.StreamEndEvent
					hasEndEvent  bool
				)

				// Collect all stream events using range loop
				for event, err := range model.GenerateEvents(ctx, request) {
					require.NoError(t, err)

					switch e := event.(type) {
					case llm.ContentPartEvent:
						contentParts = append(contentParts, e.Part)
					case llm.StreamEndEvent:
						endEvent = e
						hasEndEvent = true
					case llm.ErrorEvent:
						t.Fatalf("Received error event: %s (code: %s)", e.Message, e.Code)
					}
				}

				// Verify we received content
				require.NotEmpty(t, contentParts, "Should receive content parts")

				// Verify we got an end event
				require.True(t, hasEndEvent, "Should receive stream end event")
				assert.Equal(t, llm.FinishReasonStop, endEvent.Response.FinishReason)

				// Verify usage information in end event
				require.NotNil(t, endEvent.Response.Usage)
				assert.Positive(t, endEvent.Response.Usage.InputTokens)
				assert.Positive(t, endEvent.Response.Usage.OutputTokens)
				assert.Positive(t, endEvent.Response.Usage.TotalTokens)

				// Verify StreamEndEvent.Response.Message contains aggregated content
				// Streaming may send many small text chunks, but final response combines them
				require.NotEmpty(t, endEvent.Response.Message.Content,
					"StreamEndEvent.Response.Message should contain content")

				// Aggregate streamed text and compare with final response text
				var streamedText strings.Builder
				for _, part := range contentParts {
					if part.Kind == llm.PartText {
						streamedText.WriteString(part.Text)
					}
				}

				finalText := endEvent.Response.TextContent()
				assert.Equal(t, streamedText.String(), finalText,
					"StreamEndEvent aggregated text should match all streamed text chunks")
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
			validate: func(t *testing.T, model llm.Model, ctx context.Context, request *llm.Request) {
				t.Helper()

				eventCount := 0

				var (
					hasContent  bool
					hasEndEvent bool
				)

				for event, err := range model.GenerateEvents(ctx, request) {
					require.NoError(t, err)

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
			validate: func(t *testing.T, model llm.Model, ctx context.Context, request *llm.Request) {
				t.Helper()

				var (
					contentParts []*llm.Part
					endEvent     llm.StreamEndEvent
					hasEndEvent  bool
				)

				for event, err := range model.GenerateEvents(ctx, request) {
					require.NoError(t, err)

					switch e := event.(type) {
					case llm.ContentPartEvent:
						contentParts = append(contentParts, e.Part)
					case llm.StreamEndEvent:
						endEvent = e
						hasEndEvent = true
					case llm.ErrorEvent:
						t.Fatalf("Received error event: %s", e.Message)
					}
				}

				// Verify we got an end event
				require.True(t, hasEndEvent, "Should receive stream end event")
				assert.Equal(t, llm.FinishReasonStop, endEvent.Response.FinishReason)

				// Verify usage information
				require.NotNil(t, endEvent.Response.Usage)
				assert.Positive(t, endEvent.Response.Usage.TotalTokens)

				// Verify StreamEndEvent.Response.Message contains aggregated content
				require.NotEmpty(t, contentParts, "Should have received content parts")
				require.NotEmpty(t, endEvent.Response.Message.Content,
					"StreamEndEvent.Response.Message should contain content")

				// Aggregate streamed text and compare with final response text
				var streamedText strings.Builder
				for _, part := range contentParts {
					if part.Kind == llm.PartText {
						streamedText.WriteString(part.Text)
					}
				}

				finalText := endEvent.Response.TextContent()
				assert.Equal(t, streamedText.String(), finalText,
					"StreamEndEvent aggregated text should match all streamed text chunks")
			},
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
			defer cancel()

			tt.validate(s.T(), model, ctx, tt.request)
		})
	}
}

func (s *Suite) TestGenerateWithReasoning() {
	model := s.fixture.ReasoningModel()
	if model == nil {
		s.T().Skip("No reasoning model available")
	}

	caps := model.Capabilities()
	if !caps.Reasoning {
		s.T().Skip("Model does not support reasoning")
	}

	s.Run("complex reasoning question", func() {
		ctx, cancel := context.WithTimeout(context.Background(), reasoningTestTimeout)
		defer cancel()

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

func (s *Suite) TestGenerateEventsWithReasoning() {
	model := s.fixture.ReasoningModel()
	if model == nil {
		s.T().Skip("No reasoning model available")
	}

	caps := model.Capabilities()
	if !caps.Reasoning {
		s.T().Skip("Model does not support reasoning")
	}

	if !caps.Streaming {
		s.T().Skip("Model does not support streaming")
	}

	s.Run("streaming with reasoning traces", func() {
		ctx, cancel := context.WithTimeout(context.Background(), reasoningTestTimeout)
		defer cancel()

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

		var (
			contentParts []*llm.Part
			endEvent     llm.StreamEndEvent
			hasEndEvent  bool
		)

		eventCount := 0

		// Collect all stream events using range loop
		for event, streamErr := range model.GenerateEvents(ctx, request) {
			s.Require().NoError(streamErr)

			eventCount++

			switch e := event.(type) {
			case llm.ContentPartEvent:
				contentParts = append(contentParts, e.Part)
			case llm.StreamEndEvent:
				endEvent = e
				hasEndEvent = true
			case llm.ErrorEvent:
				s.T().Fatalf("Received error event: %s (code: %s)", e.Message, e.Code)
			}
		}

		// Verify we received events
		s.Positive(eventCount, "Should receive stream events")

		// Verify we got content
		s.NotEmpty(contentParts, "Should receive content parts")

		// Extract and combine text parts for quality checks
		var (
			allTextSb      strings.Builder
			allReasoningSb strings.Builder
		)

		for _, part := range contentParts {
			if part.IsText() {
				allTextSb.WriteString(part.Text)
			} else if part.IsReasoning() {
				allReasoningSb.WriteString(part.ReasoningTrace.Text)
			}
		}

		allText := allTextSb.String()
		s.Greater(len(allText), 200, "Should provide detailed technical analysis")

		// Should mention relevant concepts
		lowerText := strings.ToLower(allText)
		s.True(
			strings.Contains(lowerText, "consensus") ||
				strings.Contains(lowerText, "byzantine") ||
				strings.Contains(lowerText, "partition"),
			"Should discuss relevant distributed systems concepts")

		// Verify reasoning content (should be present for complex question)
		allReasoning := allReasoningSb.String()
		s.NotEmpty(allReasoning, "Should receive reasoning traces for complex question")
		s.Greater(len(allReasoning), 50, "Should have substantial reasoning content")

		// Verify we got an end event
		s.Require().True(hasEndEvent, "Should receive stream end event")
		s.Equal(llm.FinishReasonStop, endEvent.Response.FinishReason)

		// Verify usage information
		s.Require().NotNil(endEvent.Response.Usage)
		s.Greater(endEvent.Response.Usage.TotalTokens, 200, "Complex reasoning should use many tokens")

		// Verify StreamEndEvent.Response.Message contains aggregated content
		// Streaming may send many small chunks, but final response combines them
		s.Require().NotEmpty(endEvent.Response.Message.Content,
			"StreamEndEvent.Response.Message should contain content")

		// Aggregate streamed text and reasoning, compare with final response
		var (
			streamedTextSb      strings.Builder
			streamedReasoningSb strings.Builder
		)

		for _, part := range contentParts {
			if part.Kind == llm.PartText {
				streamedTextSb.WriteString(part.Text)
			} else if part.Kind == llm.PartReasoning {
				streamedReasoningSb.WriteString(part.ReasoningTrace.Text)
			}
		}

		finalText := endEvent.Response.TextContent()
		s.Equal(streamedTextSb.String(), finalText,
			"StreamEndEvent aggregated text should match all streamed text chunks")

		// Verify reasoning traces are present in final response
		streamedReasoning := streamedReasoningSb.String()
		s.NotEmpty(streamedReasoning, "Should have streamed reasoning content")

		var finalReasoningSb strings.Builder
		for _, part := range endEvent.Response.Message.Content {
			if part.Kind == llm.PartReasoning && part.ReasoningTrace != nil {
				finalReasoningSb.WriteString(part.ReasoningTrace.Text)
			}
		}

		finalReasoning := finalReasoningSb.String()
		s.Equal(streamedReasoning, finalReasoning,
			"StreamEndEvent aggregated reasoning should match all streamed reasoning chunks")
	})
}

func (s *Suite) TestStructuredOutputs() {
	model := s.fixture.StandardModel()
	if model == nil {
		s.T().Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.StructuredOutput {
		s.T().Skip("Model does not support structured output")
	}

	s.Run("JSON Schema structured output", func() {
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

		ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
		defer cancel()

		resp, err := model.Generate(ctx, req)
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

func (s *Suite) TestJSONObjectOutput() {
	model := s.fixture.StandardModel()
	if model == nil {
		s.T().Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.StructuredOutput {
		s.T().Skip("Model does not support structured output")
	}

	s.Run("JSON Object mode", func() {
		req := &llm.Request{
			Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: []*llm.Part{llm.NewTextPart("List 3 colors in JSON format with their hex codes")},
			}},
			ResponseFormat: &llm.ResponseFormat{
				Type: llm.ResponseFormatJSONObject,
			},
		}

		ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
		defer cancel()

		resp, err := model.Generate(ctx, req)
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

func (s *Suite) TestGenerateEventsWithTools() {
	model := s.fixture.StandardModel()
	if model == nil {
		s.T().Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.Tools {
		s.T().Skip("Model does not support tools")
	}

	if !caps.Streaming {
		s.T().Skip("Model does not support streaming")
	}

	tests := []struct {
		name     string
		request  *llm.Request
		validate func(t *testing.T, model llm.Model, ctx context.Context, request *llm.Request)
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
			validate: func(t *testing.T, model llm.Model, ctx context.Context, request *llm.Request) {
				t.Helper()

				var (
					toolRequests []*llm.ToolRequest
					endEvent     llm.StreamEndEvent
					hasEndEvent  bool
				)

				// Collect all stream events using range loop
				for event, err := range model.GenerateEvents(ctx, request) {
					require.NoError(t, err)

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
			validate: func(t *testing.T, model llm.Model, ctx context.Context, request *llm.Request) {
				t.Helper()

				var (
					toolRequests []*llm.ToolRequest
					endEvent     llm.StreamEndEvent
					hasEndEvent  bool
				)

				toolRequestsByName := make(map[string][]*llm.ToolRequest)

				// Collect all stream events using range loop
				for event, err := range model.GenerateEvents(ctx, request) {
					require.NoError(t, err)

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

			tt.validate(s.T(), model, ctx, tt.request)
		})
	}
}

func (s *Suite) TestToolExecutionLoop() {
	model := s.fixture.StandardModel()
	if model == nil {
		s.T().Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.Tools {
		s.T().Skip("Model does not support tools")
	}

	s.Run("tool execution with result feedback", func() {
		ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
		defer cancel()

		// Step 1: Initial request that should trigger tool call
		initialRequest := &llm.Request{
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
							}
						},
						"required": ["location"]
					}`),
				},
			},
		}

		// Get initial response with tool call
		response, err := model.Generate(ctx, initialRequest)
		s.Require().NoError(err)
		s.Require().NotNil(response)
		s.Equal(llm.FinishReasonToolCalls, response.FinishReason, "Should request tool call")

		// Extract tool requests
		var toolRequests []*llm.ToolRequest

		for _, part := range response.Message.Content {
			if part.IsToolRequest() {
				toolRequests = append(toolRequests, part.ToolRequest)
			}
		}

		s.Require().NotEmpty(toolRequests, "Should have at least one tool request")
		s.Equal("get_weather", toolRequests[0].Name, "Should request get_weather tool")

		// Step 2: Simulate tool execution and send result back
		followUpRequest := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("What is the weather in San Francisco?"),
					},
				},
				response.Message, // Add the assistant's tool call message
				{
					Role: llm.RoleTool,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     toolRequests[0].ID,
							Name:   toolRequests[0].Name,
							Result: json.RawMessage(`{"temperature": 72, "condition": "sunny", "unit": "fahrenheit"}`),
						}),
					},
				},
			},
			Tools: initialRequest.Tools, // Keep tool definitions
		}

		// Step 3: Get final response with tool result incorporated
		finalResponse, err := model.Generate(ctx, followUpRequest)
		s.Require().NoError(err)
		s.Require().NotNil(finalResponse)

		// Verify final response
		s.Equal(llm.FinishReasonStop, finalResponse.FinishReason, "Should complete after tool result")

		// Verify response contains information from tool result
		finalText := finalResponse.TextContent()
		s.NotEmpty(finalText, "Should have text response")

		// Check that the response mentions the weather data
		lowerText := strings.ToLower(finalText)
		s.True(
			strings.Contains(lowerText, "72") || strings.Contains(lowerText, "sunny") || strings.Contains(lowerText, "san francisco"),
			"Response should incorporate tool result data")

		// Verify usage information
		s.Require().NotNil(finalResponse.Usage)
		s.Positive(finalResponse.Usage.TotalTokens)
	})

	s.Run("multi-turn tool execution streaming", func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*testTimeout)
		defer cancel()

		// Skip if streaming not supported
		if !caps.Streaming {
			s.T().Skip("Model does not support streaming")
		}

		// Step 1: Initial request that should trigger tool call
		initialRequest := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("What is the current time in Tokyo?"),
					},
				},
			},
			Tools: []llm.ToolDefinition{
				{
					Name:        "get_time",
					Description: "Get the current time in a given timezone",
					Parameters: json.RawMessage(`{
						"type": "object",
						"properties": {
							"timezone": {
								"type": "string",
								"description": "The timezone, e.g. Asia/Tokyo"
							}
						},
						"required": ["timezone"]
					}`),
				},
			},
		}

		// Collect tool requests from stream
		var (
			toolRequests     []*llm.ToolRequest
			assistantMessage llm.Message
		)

		for event, err := range model.GenerateEvents(ctx, initialRequest) {
			s.Require().NoError(err)

			switch e := event.(type) {
			case llm.ContentPartEvent:
				if e.Part.IsToolRequest() {
					toolRequests = append(toolRequests, e.Part.ToolRequest)
					assistantMessage.Content = append(assistantMessage.Content, e.Part)
				}
			case llm.StreamEndEvent:
				s.Equal(llm.FinishReasonToolCalls, e.Response.FinishReason)

				assistantMessage.Role = llm.RoleAssistant
			}
		}

		s.Require().NotEmpty(toolRequests, "Should have tool requests")
		s.Equal("get_time", toolRequests[0].Name)

		// Step 2: Send tool result and get final response via streaming
		followUpRequest := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("What is the current time in Tokyo?"),
					},
				},
				assistantMessage,
				{
					Role: llm.RoleTool,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     toolRequests[0].ID,
							Name:   toolRequests[0].Name,
							Result: json.RawMessage(`{"time": "14:30", "timezone": "Asia/Tokyo", "date": "2025-11-10"}`),
						}),
					},
				},
			},
			Tools: initialRequest.Tools,
		}

		var (
			finalText         strings.Builder
			finalFinishReason llm.FinishReason
		)

		for event, err := range model.GenerateEvents(ctx, followUpRequest) {
			s.Require().NoError(err)

			switch e := event.(type) {
			case llm.ContentPartEvent:
				if e.Part.IsText() {
					finalText.WriteString(e.Part.Text)
				}
			case llm.StreamEndEvent:
				finalFinishReason = e.Response.FinishReason
				s.Require().NotNil(e.Response.Usage)
				s.Positive(e.Response.Usage.TotalTokens)
			}
		}

		s.Equal(llm.FinishReasonStop, finalFinishReason, "Should complete after tool result")
		s.NotEmpty(finalText.String(), "Should have text response")

		// Verify response mentions the time data
		lowerText := strings.ToLower(finalText.String())
		s.True(
			strings.Contains(lowerText, "tokyo") || strings.Contains(lowerText, "14:30") || strings.Contains(lowerText, "time"),
			"Response should incorporate tool result data")
	})

	s.Run("sequential tool calls with dependency", func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*testTimeout)
		defer cancel()

		// Step 1: Ask for weather in current location (requires two tools: location, then weather)
		initialRequest := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("What's the weather like in my current location?"),
					},
				},
			},
			Tools: []llm.ToolDefinition{
				{
					Name:        "get_current_location",
					Description: "Get the user's current location",
					Parameters:  json.RawMessage(`{"type": "object", "properties": {}}`),
				},
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
			},
		}

		// Get first response - should request get_current_location
		response1, err := model.Generate(ctx, initialRequest)
		s.Require().NoError(err)
		s.Require().NotNil(response1)
		s.Equal(llm.FinishReasonToolCalls, response1.FinishReason, "Should request first tool")

		// Extract first tool request
		var toolRequests1 []*llm.ToolRequest

		for _, part := range response1.Message.Content {
			if part.IsToolRequest() {
				toolRequests1 = append(toolRequests1, part.ToolRequest)
			}
		}

		s.Require().NotEmpty(toolRequests1, "Should have tool request")
		s.Equal("get_current_location", toolRequests1[0].Name, "Should request location first")

		// Step 2: Provide location result
		request2 := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("What's the weather like in my current location?"),
					},
				},
				response1.Message,
				{
					Role: llm.RoleTool,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     toolRequests1[0].ID,
							Name:   toolRequests1[0].Name,
							Result: json.RawMessage(`{"city": "San Francisco", "state": "CA", "country": "USA"}`),
						}),
					},
				},
			},
			Tools: initialRequest.Tools,
		}

		// Get second response - should now request get_weather with location
		response2, err := model.Generate(ctx, request2)
		s.Require().NoError(err)
		s.Require().NotNil(response2)
		s.Equal(llm.FinishReasonToolCalls, response2.FinishReason, "Should request second tool")

		// Extract second tool request
		var toolRequests2 []*llm.ToolRequest

		for _, part := range response2.Message.Content {
			if part.IsToolRequest() {
				toolRequests2 = append(toolRequests2, part.ToolRequest)
			}
		}

		s.Require().NotEmpty(toolRequests2, "Should have second tool request")
		s.Equal("get_weather", toolRequests2[0].Name, "Should request weather with location")

		// Verify the weather request includes location from first tool
		var weatherArgs map[string]any

		err = json.Unmarshal(toolRequests2[0].Arguments, &weatherArgs)
		s.Require().NoError(err)
		s.Contains(weatherArgs, "location", "Should have location argument")

		location, ok := weatherArgs["location"].(string)
		s.Require().True(ok, "location should be a string")
		s.True(
			strings.Contains(strings.ToLower(location), "san francisco") || strings.Contains(strings.ToLower(location), "sf"),
			"Location should reference San Francisco from first tool result")

		// Step 3: Provide weather result
		request3 := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("What's the weather like in my current location?"),
					},
				},
				response1.Message,
				{
					Role: llm.RoleTool,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     toolRequests1[0].ID,
							Name:   toolRequests1[0].Name,
							Result: json.RawMessage(`{"city": "San Francisco", "state": "CA", "country": "USA"}`),
						}),
					},
				},
				response2.Message,
				{
					Role: llm.RoleTool,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     toolRequests2[0].ID,
							Name:   toolRequests2[0].Name,
							Result: json.RawMessage(`{"temperature": 65, "condition": "foggy", "humidity": 85}`),
						}),
					},
				},
			},
			Tools: initialRequest.Tools,
		}

		// Get final response with weather answer
		finalResponse, err := model.Generate(ctx, request3)
		s.Require().NoError(err)
		s.Require().NotNil(finalResponse)
		s.Equal(llm.FinishReasonStop, finalResponse.FinishReason, "Should complete after all tools")

		// Verify final response mentions both location and weather
		finalText := finalResponse.TextContent()
		s.NotEmpty(finalText, "Should have final text response")

		lowerText := strings.ToLower(finalText)
		s.True(
			strings.Contains(lowerText, "65") || strings.Contains(lowerText, "foggy") || strings.Contains(lowerText, "san francisco"),
			"Response should mention weather and location from tool results")

		s.Require().NotNil(finalResponse.Usage)
		s.Positive(finalResponse.Usage.TotalTokens)
	})
}

func (s *Suite) TestAllSupportedModels() {
	t := s.T()

	models := s.fixture.Models()
	if len(models) == 0 {
		t.Skip("No models available for testing")
	}

	t.Run("basic generation works for all supported models", func(t *testing.T) {
		for _, m := range models {
			modelName := m.Name
			t.Run("model_"+modelName, func(t *testing.T) {
				model, err := s.fixture.NewModel(modelName)
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

				ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
				defer cancel()

				resp, err := model.Generate(ctx, reqObj)
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
