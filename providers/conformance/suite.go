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

package conformance

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Suite provides generic conformance tests for any provider implementing the llm.Model interface.
type Suite struct {
	fixture Fixture
}

// NewSuite creates a new conformance test suite with the given fixture.
func NewSuite(fixture Fixture) *Suite {
	return &Suite{
		fixture: fixture,
	}
}

func (s *Suite) TestGenerate(t *testing.T) {
	t.Helper()
	testGenerate(t, s.fixture)
}

func (s *Suite) TestGenerateEvents(t *testing.T) {
	t.Helper()
	testGenerateEvents(t, s.fixture)
}

func (s *Suite) TestGenerateWithReasoning(t *testing.T) {
	t.Helper()
	testGenerateWithReasoning(t, s.fixture)
}

func (s *Suite) TestGenerateEventsWithReasoning(t *testing.T) {
	t.Helper()
	testGenerateEventsWithReasoning(t, s.fixture)
}

func (s *Suite) TestStructuredOutputs(t *testing.T) {
	t.Helper()
	testStructuredOutputs(t, s.fixture)
}

func (s *Suite) TestJSONObjectOutput(t *testing.T) {
	t.Helper()
	testJSONObjectOutput(t, s.fixture)
}

func (s *Suite) TestGenerateEventsWithTools(t *testing.T) {
	t.Helper()
	testGenerateEventsWithTools(t, s.fixture)
}

func (s *Suite) TestToolExecutionLoop(t *testing.T) {
	t.Helper()
	testToolExecutionLoop(t, s.fixture)
}

func (s *Suite) TestAllSupportedModels(t *testing.T) {
	t.Helper()
	testAllSupportedModels(t, s.fixture)
}

func testGenerate(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	model := fixture.NewStandardModel(t)
	if model == nil {
		t.Skip("No standard model available")
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
			validate: func(t *testing.T, response *llm.Response, err error) {
				t.Helper()
				require.NoError(t, err)
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
			validate: func(t *testing.T, response *llm.Response, err error) {
				t.Helper()
				require.NoError(t, err)
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
			validate: func(t *testing.T, response *llm.Response, err error) {
				t.Helper()
				require.NoError(t, err)
				require.NotNil(t, response)
				require.NotEmpty(t, response.Message.Content)

				assert.Equal(t, llm.FinishReasonStop, response.FinishReason)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response, err := model.Generate(t.Context(), tt.request)
			tt.validate(t, response, err)
		})
	}
}

func testGenerateEvents(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	model := fixture.NewStandardModel(t)
	if model == nil {
		t.Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.Streaming {
		t.Skip("Model does not support streaming")
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
		t.Run(tt.name, func(t *testing.T) {
			tt.validate(t, model, t.Context(), tt.request)
		})
	}
}

func testGenerateWithReasoning(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	model := fixture.NewReasoningModel(t)
	if model == nil {
		t.Skip("No reasoning model available")
	}

	caps := model.Capabilities()
	if !caps.Reasoning {
		t.Skip("Model does not support reasoning")
	}

	t.Run("complex reasoning question", func(t *testing.T) {
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

		// Reasoning traces may not appear on every attempt (provider-dependent), so retry.
		const maxAttempts = 3

		var (
			response     *llm.Response
			hasReasoning bool
		)

		for attempt := 1; attempt <= maxAttempts; attempt++ {
			if attempt > 1 {
				t.Logf("Retry attempt %d/%d for reasoning traces", attempt, maxAttempts)
			}

			var err error

			response, err = model.Generate(t.Context(), request)
			if err != nil {
				if attempt < maxAttempts {
					t.Logf("Attempt %d/%d failed: %v", attempt, maxAttempts, err)

					continue
				}

				require.NoError(t, err)
			}

			require.NotNil(t, response)
			require.NotEmpty(t, response.Message.Content)

			hasReasoning = false

			for _, part := range response.Message.Content {
				if part.IsReasoning() {
					hasReasoning = true

					assert.NotEmpty(t, part.ReasoningTrace.Text)
					assert.Greater(t, len(part.ReasoningTrace.Text), 30, "Should show detailed reasoning process")
				}
			}

			if hasReasoning {
				t.Logf("Received reasoning traces on attempt %d/%d", attempt, maxAttempts)

				break
			}

			if attempt == maxAttempts {
				t.Fatalf("Complex technical question should trigger reasoning after %d attempts", maxAttempts)
			}
		}

		text := response.TextContent()
		assert.Greater(t, len(text), 200, "Should provide detailed technical analysis")

		lowerText := strings.ToLower(text)
		assert.True(t,
			strings.Contains(lowerText, "consensus") ||
				strings.Contains(lowerText, "byzantine") ||
				strings.Contains(lowerText, "partition"),
			"Should discuss relevant distributed systems concepts")

		require.NotNil(t, response.Usage)
		assert.Greater(t, response.Usage.TotalTokens, 200, "Complex reasoning should use many tokens")
	})
}

func testGenerateEventsWithReasoning(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	model := fixture.NewReasoningModel(t)
	if model == nil {
		t.Skip("No reasoning model available")
	}

	caps := model.Capabilities()
	if !caps.Reasoning {
		t.Skip("Model does not support reasoning")
	}

	if !caps.Streaming {
		t.Skip("Model does not support streaming")
	}

	t.Run("streaming with reasoning traces", func(t *testing.T) {
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

		// Retry up to 3 times to handle flaky reasoning trace streaming
		// Some providers (e.g., OpenAI) don't consistently stream reasoning traces
		const maxAttempts = 3
		var (
			contentParts []*llm.Part
			endEvent     llm.StreamEndEvent
			hasEndEvent  bool
			allReasoning string
		)

		collectReasoningEvents := func() error {
			contentParts = nil
			hasEndEvent = false

			for event, streamErr := range model.GenerateEvents(t.Context(), request) {
				if streamErr != nil {
					return streamErr
				}

				switch e := event.(type) {
				case llm.ContentPartEvent:
					contentParts = append(contentParts, e.Part)
				case llm.StreamEndEvent:
					endEvent = e
					hasEndEvent = true
				case llm.ErrorEvent:
					return fmt.Errorf("error event: %s (code: %s)", e.Message, e.Code)
				}
			}

			return nil
		}

		for attempt := 1; attempt <= maxAttempts; attempt++ {
			if attempt > 1 {
				t.Logf("Retry attempt %d/%d for reasoning traces", attempt, maxAttempts)
			}

			if err := collectReasoningEvents(); err != nil {
				if attempt < maxAttempts {
					t.Logf("Attempt %d/%d failed: %v", attempt, maxAttempts, err)

					continue
				}

				require.NoError(t, err)
			}

			assert.NotEmpty(t, contentParts, "Should receive content parts")

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
			assert.Greater(t, len(allText), 200, "Should provide detailed technical analysis")

			// Should mention relevant concepts
			lowerText := strings.ToLower(allText)
			assert.True(t,
				strings.Contains(lowerText, "consensus") ||
					strings.Contains(lowerText, "byzantine") ||
					strings.Contains(lowerText, "partition"),
				"Should discuss relevant distributed systems concepts")

			allReasoning = allReasoningSb.String()

			// If we got reasoning traces, we're done
			if len(allReasoning) > 0 {
				t.Logf("Received reasoning traces on attempt %d/%d", attempt, maxAttempts)
				break
			}

			// If this is the last attempt, fail
			if attempt == maxAttempts {
				t.Fatalf("Should receive reasoning traces for complex question after %d attempts", maxAttempts)
			}
		}

		// Verify reasoning content
		assert.NotEmpty(t, allReasoning, "Should receive reasoning traces for complex question")
		assert.Greater(t, len(allReasoning), 50, "Should have substantial reasoning content")

		// Verify we got an end event
		require.True(t, hasEndEvent, "Should receive stream end event")
		assert.Equal(t, llm.FinishReasonStop, endEvent.Response.FinishReason)

		// Verify usage information
		require.NotNil(t, endEvent.Response.Usage)
		assert.Greater(t, endEvent.Response.Usage.TotalTokens, 200, "Complex reasoning should use many tokens")

		// Verify StreamEndEvent.Response.Message contains aggregated content
		// Streaming may send many small chunks, but final response combines them
		require.NotEmpty(t, endEvent.Response.Message.Content,
			"StreamEndEvent.Response.Message should contain content")

		// Aggregate streamed text and reasoning, compare with final response
		var (
			streamedTextSb      strings.Builder
			streamedReasoningSb strings.Builder
		)

		for _, part := range contentParts {
			switch part.Kind {
			case llm.PartText:
				streamedTextSb.WriteString(part.Text)
			case llm.PartReasoning:
				streamedReasoningSb.WriteString(part.ReasoningTrace.Text)
			case llm.PartToolRequest, llm.PartToolResponse:
			}
		}

		finalText := endEvent.Response.TextContent()
		assert.Equal(t, streamedTextSb.String(), finalText,
			"StreamEndEvent aggregated text should match all streamed text chunks")

		// Verify reasoning traces are present in final response
		streamedReasoning := streamedReasoningSb.String()
		assert.NotEmpty(t, streamedReasoning, "Should have streamed reasoning content")

		var finalReasoningSb strings.Builder

		for _, part := range endEvent.Response.Message.Content {
			if part.Kind == llm.PartReasoning && part.ReasoningTrace != nil {
				finalReasoningSb.WriteString(part.ReasoningTrace.Text)
			}
		}

		finalReasoning := finalReasoningSb.String()
		assert.Equal(t, streamedReasoning, finalReasoning,
			"StreamEndEvent aggregated reasoning should match all streamed reasoning chunks")
	})
}

func testStructuredOutputs(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	model := fixture.NewStandardModel(t)
	if model == nil {
		t.Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.StructuredOutput {
		t.Skip("Model does not support structured output")
	}

	t.Run("JSON Schema structured output", func(t *testing.T) {
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

		resp, err := model.Generate(t.Context(), req)
		require.NoError(t, err)
		require.NotNil(t, resp)

		// Get the text content
		textContent := resp.TextContent()
		assert.NotEmpty(t, textContent)

		// Verify it's valid JSON that matches our schema
		var personData map[string]any

		err = json.Unmarshal([]byte(textContent), &personData)
		require.NoError(t, err, "Response should be valid JSON")

		// Verify required fields are present
		assert.Contains(t, personData, "name")
		assert.Contains(t, personData, "age")
		assert.Contains(t, personData, "city")

		// Verify field types
		assert.IsType(t, "", personData["name"])
		assert.IsType(t, float64(0), personData["age"]) // JSON numbers are float64
		assert.IsType(t, "", personData["city"])
	})
}

func testJSONObjectOutput(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	model := fixture.NewStandardModel(t)
	if model == nil {
		t.Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.JSONMode {
		t.Skip("Model does not support JSON mode")
	}

	t.Run("JSON Object mode", func(t *testing.T) {
		req := &llm.Request{
			Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: []*llm.Part{llm.NewTextPart("List 3 colors in JSON format with their hex codes")},
			}},
			ResponseFormat: &llm.ResponseFormat{
				Type: llm.ResponseFormatJSONObject,
			},
		}

		resp, err := model.Generate(t.Context(), req)
		require.NoError(t, err)
		require.NotNil(t, resp)

		// Get the text content
		textContent := resp.TextContent()
		assert.NotEmpty(t, textContent)

		// Verify it's valid JSON (but we don't enforce specific structure)
		var jsonData any

		err = json.Unmarshal([]byte(textContent), &jsonData)
		require.NoError(t, err, "Response should be valid JSON")
	})
}

func testGenerateEventsWithTools(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	model := fixture.NewStandardModel(t)
	if model == nil {
		t.Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.Tools {
		t.Skip("Model does not support tools")
	}

	if !caps.Streaming {
		t.Skip("Model does not support streaming")
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
							llm.NewTextPart("What is the weather in San Francisco, CA? In Celsius please."),
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

				// Verify we received at least one tool request.
				// Note: The model may choose to call the tool multiple times with different arguments,
				// which is valid LLM behavior (e.g., calling get_weather with both celsius and fahrenheit).
				assert.NotEmpty(t, toolRequests, "Should receive at least one tool request")

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

				finalToolReqs := endEvent.Response.ToolRequests()
				require.NotEmpty(t, finalToolReqs, "Final message should have tool requests")

				finalToolReq := finalToolReqs[0]
				assert.Equal(t, "get_weather", finalToolReq.Name, "Final message tool name should match")

				var finalArgs map[string]any
				err := json.Unmarshal(finalToolReq.Arguments, &finalArgs)
				require.NoError(t, err, "Final message tool arguments should be valid JSON")
				require.NotEmpty(t, finalArgs, "Final message tool arguments must not be empty!")
				assert.Contains(t, finalArgs, "location", "Final message should contain location")

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
						Role: llm.RoleSystem,
						Content: []*llm.Part{
							llm.NewTextPart("You are a tool-calling assistant. You MUST use the provided tools to answer questions. NEVER answer directly — always call the appropriate tools. When multiple tools are needed, call them all in parallel in a single response."),
						},
					},
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("What's the current weather in Tokyo and what time is it there right now? Use the get_weather and get_time tools."),
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
									"description": "The city name, e.g. Tokyo, San Francisco"
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

				// Models may non-deterministically skip tool calls, so retry up to 3 times.
				const maxAttempts = 3

				var (
					toolRequests       []*llm.ToolRequest
					toolRequestsByName map[string][]*llm.ToolRequest
					endEvent           llm.StreamEndEvent
					hasEndEvent        bool
				)

				collectToolEvents := func() error {
					toolRequests = nil
					toolRequestsByName = make(map[string][]*llm.ToolRequest)
					hasEndEvent = false

					for event, err := range model.GenerateEvents(ctx, request) {
						if err != nil {
							return err
						}

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
							return fmt.Errorf("error event: %s (code: %s)", e.Message, e.Code)
						}
					}

					return nil
				}

				for attempt := 1; attempt <= maxAttempts; attempt++ {
					if attempt > 1 {
						t.Logf("Retry attempt %d/%d for multiple tool calls", attempt, maxAttempts)
					}

					if err := collectToolEvents(); err != nil {
						if attempt < maxAttempts {
							t.Logf("Attempt %d/%d failed: %v", attempt, maxAttempts, err)

							continue
						}

						require.NoError(t, err)
					}

					if len(toolRequests) > 1 {
						t.Logf("Received %d tool calls on attempt %d/%d", len(toolRequests), attempt, maxAttempts)

						break
					}

					if attempt == maxAttempts {
						t.Fatalf("Should receive multiple tool requests after %d attempts, got %d", maxAttempts, len(toolRequests))
					}
				}

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
		t.Run(tt.name, func(t *testing.T) {
			tt.validate(t, model, t.Context(), tt.request)
		})
	}
}

func testToolExecutionLoop(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	model := fixture.NewStandardModel(t)
	if model == nil {
		t.Skip("No standard model available")
	}

	caps := model.Capabilities()
	if !caps.Tools {
		t.Skip("Model does not support tools")
	}

	t.Run("tool execution with result feedback", func(t *testing.T) {
		// Step 1: Initial request that should trigger tool call
		initialRequest := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("What is the weather in San Francisco, CA?"),
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
		response, err := model.Generate(t.Context(), initialRequest)
		require.NoError(t, err)
		require.NotNil(t, response)
		assert.Equal(t, llm.FinishReasonToolCalls, response.FinishReason, "Should request tool call")

		// Extract tool requests
		var toolRequests []*llm.ToolRequest

		for _, part := range response.Message.Content {
			if part.IsToolRequest() {
				toolRequests = append(toolRequests, part.ToolRequest)
			}
		}

		require.NotEmpty(t, toolRequests, "Should have at least one tool request")
		assert.Equal(t, "get_weather", toolRequests[0].Name, "Should request get_weather tool")

		// Step 2: Simulate tool execution and send result back
		followUpRequest := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("What is the weather in San Francisco, CA?"),
					},
				},
				response.Message, // Add the assistant's tool call message
				{
					Role: llm.RoleUser,
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
		finalResponse, err := model.Generate(t.Context(), followUpRequest)
		require.NoError(t, err)
		require.NotNil(t, finalResponse)

		// Verify final response
		assert.Equal(t, llm.FinishReasonStop, finalResponse.FinishReason, "Should complete after tool result")

		// Verify response contains information from tool result
		finalText := finalResponse.TextContent()
		assert.NotEmpty(t, finalText, "Should have text response")

		// Check that the response mentions the weather data
		lowerText := strings.ToLower(finalText)
		assert.True(t,
			strings.Contains(lowerText, "72") || strings.Contains(lowerText, "sunny") || strings.Contains(lowerText, "san francisco"),
			"Response should incorporate tool result data")

		// Verify usage information
		require.NotNil(t, finalResponse.Usage)
		assert.Positive(t, finalResponse.Usage.TotalTokens)
	})

	t.Run("multi-turn tool execution streaming", func(t *testing.T) {
		// Skip if streaming not supported
		if !caps.Streaming {
			t.Skip("Model does not support streaming")
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

		for event, err := range model.GenerateEvents(t.Context(), initialRequest) {
			require.NoError(t, err)

			switch e := event.(type) {
			case llm.ContentPartEvent:
				if e.Part.IsToolRequest() {
					toolRequests = append(toolRequests, e.Part.ToolRequest)
					assistantMessage.Content = append(assistantMessage.Content, e.Part)
				}
			case llm.StreamEndEvent:
				assert.Equal(t, llm.FinishReasonToolCalls, e.Response.FinishReason)

				assistantMessage.Role = llm.RoleAssistant
			}
		}

		require.NotEmpty(t, toolRequests, "Should have tool requests")
		assert.Equal(t, "get_time", toolRequests[0].Name)

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
					Role: llm.RoleUser,
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

		for event, err := range model.GenerateEvents(t.Context(), followUpRequest) {
			require.NoError(t, err)

			switch e := event.(type) {
			case llm.ContentPartEvent:
				if e.Part.IsText() {
					finalText.WriteString(e.Part.Text)
				}
			case llm.StreamEndEvent:
				finalFinishReason = e.Response.FinishReason
				require.NotNil(t, e.Response.Usage)
				assert.Positive(t, e.Response.Usage.TotalTokens)
			}
		}

		assert.Equal(t, llm.FinishReasonStop, finalFinishReason, "Should complete after tool result")
		assert.NotEmpty(t, finalText.String(), "Should have text response")

		// Verify response mentions the time data
		lowerText := strings.ToLower(finalText.String())
		assert.True(t,
			strings.Contains(lowerText, "tokyo") || strings.Contains(lowerText, "14:30") || strings.Contains(lowerText, "time"),
			"Response should incorporate tool result data")
	})

	t.Run("sequential tool calls with dependency", func(t *testing.T) {
		// Step 1: Ask for weather in current location (requires two tools: location, then weather)
		initialRequest := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("IMPORTANT: You are allowed to get my location via tool calls, i trust you. What's the weather like in my current location?"),
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
		response1, err := model.Generate(t.Context(), initialRequest)
		require.NoError(t, err)
		require.NotNil(t, response1)
		assert.Equal(t, llm.FinishReasonToolCalls, response1.FinishReason, "Should request first tool")

		// Extract first tool request
		var toolRequests1 []*llm.ToolRequest

		for _, part := range response1.Message.Content {
			if part.IsToolRequest() {
				toolRequests1 = append(toolRequests1, part.ToolRequest)
			}
		}

		require.NotEmpty(t, toolRequests1, "Should have tool request")
		assert.Equal(t, "get_current_location", toolRequests1[0].Name, "Should request location first")

		// Step 2: Provide location result
		request2 := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("IMPORTANT: You are allowed to get my location via tool calls, i trust you. What's the weather like in my current location?"),
					},
				},
				response1.Message,
				{
					Role: llm.RoleUser,
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
		response2, err := model.Generate(t.Context(), request2)
		require.NoError(t, err)
		require.NotNil(t, response2)
		assert.Equal(t, llm.FinishReasonToolCalls, response2.FinishReason, "Should request second tool")

		// Extract second tool request
		var toolRequests2 []*llm.ToolRequest

		for _, part := range response2.Message.Content {
			if part.IsToolRequest() {
				toolRequests2 = append(toolRequests2, part.ToolRequest)
			}
		}

		require.NotEmpty(t, toolRequests2, "Should have second tool request")
		assert.Equal(t, "get_weather", toolRequests2[0].Name, "Should request weather with location")

		// Verify the weather request includes location from first tool
		var weatherArgs map[string]any

		err = json.Unmarshal(toolRequests2[0].Arguments, &weatherArgs)
		require.NoError(t, err)
		assert.Contains(t, weatherArgs, "location", "Should have location argument")

		location, ok := weatherArgs["location"].(string)
		require.True(t, ok, "location should be a string")
		assert.True(t,
			strings.Contains(strings.ToLower(location), "san francisco") || strings.Contains(strings.ToLower(location), "sf"),
			"Location should reference San Francisco from first tool result")

		// Step 3: Provide weather result
		// Models may make multiple tool calls (even duplicates), so we need to respond to ALL of them
		weatherResponses := make([]*llm.Part, 0, len(toolRequests2))
		for _, toolReq := range toolRequests2 {
			weatherResponses = append(weatherResponses, llm.NewToolResponsePart(&llm.ToolResponse{
				ID:     toolReq.ID,
				Name:   toolReq.Name,
				Result: json.RawMessage(`{"temperature": 65, "condition": "foggy", "humidity": 85}`),
			}))
		}

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
					Role: llm.RoleUser,
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
					Role:    llm.RoleUser,
					Content: weatherResponses,
				},
			},
			Tools: initialRequest.Tools,
		}

		// Get final response with weather answer
		finalResponse, err := model.Generate(t.Context(), request3)
		if err != nil {
			if reqJSON, _ := json.MarshalIndent(request3, "", "  "); reqJSON != nil {
				t.Logf("Request that failed:\n%s", reqJSON)
			}
		}

		require.NoError(t, err)
		require.NotNil(t, finalResponse)
		assert.Equal(t, llm.FinishReasonStop, finalResponse.FinishReason, "Should complete after all tools")

		// Verify final response mentions both location and weather
		finalText := finalResponse.TextContent()
		assert.NotEmpty(t, finalText, "Should have final text response")

		lowerText := strings.ToLower(finalText)
		assert.True(t,
			strings.Contains(lowerText, "65") || strings.Contains(lowerText, "foggy") || strings.Contains(lowerText, "san francisco"),
			"Response should mention weather and location from tool results")

		require.NotNil(t, finalResponse.Usage)
		assert.Positive(t, finalResponse.Usage.TotalTokens)
	})
}

func testAllSupportedModels(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	models := fixture.Models()
	if len(models) == 0 {
		t.Skip("No models available for testing")
	}

	t.Run("basic generation works for all supported models", func(t *testing.T) {
		for _, m := range models {
			modelName := m.Name
			t.Run("model_"+modelName, func(t *testing.T) {
				model, err := fixture.NewModel(modelName)
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
