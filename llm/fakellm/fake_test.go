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

package fakellm_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	llmtesting "github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

func TestFakeModel_BasicTextResponse(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.UserMessageContains("hello")).
		ThenRespondText("Hello! How can I help you?")

	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role:    llm.RoleUser,
				Content: []*llm.Part{llm.NewTextPart("hello")},
			},
		},
	}

	resp, err := model.Generate(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Hello! How can I help you?", resp.TextContent())
	assert.Equal(t, llm.FinishReasonStop, resp.FinishReason)
	assert.NotNil(t, resp.Usage)
	assert.Positive(t, resp.Usage.OutputTokens)
}

func TestFakeModel_MultipleRules(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.UserMessageContains("weather")).
		ThenRespondText("It's sunny!").
		When(llmtesting.UserMessageContains("time")).
		ThenRespondText("It's 3 PM").
		When(llmtesting.Any()).
		ThenRespondText("I don't understand")

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"weather query", "What's the weather?", "It's sunny!"},
		{"time query", "What time is it?", "It's 3 PM"},
		{"unknown query", "Tell me a joke", "I don't understand"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			req := &llm.Request{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart(tt.input)}},
				},
			}

			resp, err := model.Generate(context.Background(), req)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, resp.TextContent())
		})
	}
}

func TestFakeModel_ToolCall(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.HasTool("get_weather")).
		ThenRespondWithToolCall("get_weather", map[string]any{
			"location": "San Francisco",
			"unit":     "fahrenheit",
		})

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("What's the weather?")}},
		},
		Tools: []llm.ToolDefinition{
			{Name: "get_weather", Description: "Get weather for a location"},
		},
	}

	resp, err := model.Generate(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, llm.FinishReasonToolCalls, resp.FinishReason)
	assert.True(t, resp.HasToolRequests())

	toolReqs := resp.ToolRequests()
	require.Len(t, toolReqs, 1)
	assert.Equal(t, "get_weather", toolReqs[0].Name)
}

func TestFakeModel_Streaming(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenStreamText("This is a streaming response", llmtesting.StreamConfig{
			ChunkSize: 5, // Small chunks for testing
		})

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	var (
		chunks      []string
		finishEvent llm.StreamEndEvent
	)

	for event, err := range model.GenerateEvents(context.Background(), req) {
		require.NoError(t, err)

		switch e := event.(type) {
		case llm.ContentPartEvent:
			if e.Part.IsText() {
				chunks = append(chunks, e.Part.Text)
			}
		case llm.StreamEndEvent:
			finishEvent = e
		}
	}

	// Should have multiple chunks
	assert.Greater(t, len(chunks), 1, "should have multiple chunks")

	// Reconstruct full text
	fullText := ""

	var fullTextSb141 strings.Builder
	for _, chunk := range chunks {
		fullTextSb141.WriteString(chunk)
	}

	fullText += fullTextSb141.String()

	assert.Equal(t, "This is a streaming response", fullText)
	assert.Equal(t, llm.FinishReasonStop, finishEvent.Response.FinishReason)
}

func TestFakeModel_RateLimitOnce(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		RateLimitOnce().
		When(llmtesting.Any()).
		ThenRespondText("Success!")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	// First call should fail
	_, err := model.Generate(context.Background(), req)
	require.ErrorIs(t, err, llm.ErrAPICall)

	// Second call should succeed
	resp, err := model.Generate(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Success!", resp.TextContent())
}

func TestFakeModel_ErrorPattern(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		ErrorPattern("EESS", llm.ErrAPICall).
		When(llmtesting.Any()).
		ThenRespondText("Success")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	// Call 1: Error
	_, err := model.Generate(context.Background(), req)
	require.ErrorIs(t, err, llm.ErrAPICall)

	// Call 2: Error
	_, err = model.Generate(context.Background(), req)
	require.ErrorIs(t, err, llm.ErrAPICall)

	// Call 3: Success
	resp, err := model.Generate(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Success", resp.TextContent())

	// Call 4: Success
	resp, err = model.Generate(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Success", resp.TextContent())
}

func TestFakeModel_ContextCancellation(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel(
		llmtesting.WithLatency(llmtesting.LatencyProfile{
			Base: 200 * time.Millisecond,
		}),
	)

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := model.Generate(ctx, req)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}

func TestFakeModel_Scenario_ToolCalling(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		Scenario("weather-lookup", func(s *llmtesting.ScenarioBuilder) {
			// Turn 0: Request tool call
			s.OnTurn(0).
				When(llmtesting.HasTool("get_weather")).
				ThenRespondWithToolCall("get_weather", map[string]any{
					"location": "San Francisco",
				})

			// Turn 1: After tool response, provide answer
			s.OnTurn(1).
				When(llmtesting.LastMessageHasToolResponse("get_weather")).
				ThenRespondText("The weather in San Francisco is 68°F and sunny.")
		})

	// Turn 0: Initial request
	req1 := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("What's the weather in SF?")}},
		},
		Tools: []llm.ToolDefinition{
			{Name: "get_weather"},
		},
		Metadata: map[string]string{"session_id": "test-session"},
	}

	resp1, err := model.Generate(context.Background(), req1)
	require.NoError(t, err)
	assert.Equal(t, llm.FinishReasonToolCalls, resp1.FinishReason)
	assert.True(t, resp1.HasToolRequests())

	// Turn 1: Tool response
	req2 := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("What's the weather in SF?")}},
			resp1.Message, // Include tool request
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewToolResponsePart(&llm.ToolResponse{
						ID:     resp1.ToolRequests()[0].ID,
						Name:   "get_weather",
						Result: []byte(`{"temperature": 68, "condition": "sunny"}`),
					}),
				},
			},
		},
		Tools:    []llm.ToolDefinition{{Name: "get_weather"}},
		Metadata: map[string]string{"session_id": "test-session"},
	}

	resp2, err := model.Generate(context.Background(), req2)
	require.NoError(t, err)
	assert.Contains(t, resp2.TextContent(), "68°F")
	assert.Contains(t, resp2.TextContent(), "sunny")
}

func TestFakeModel_MidStreamError(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenStreamText("This will be interrupted", llmtesting.StreamConfig{
			ChunkSize:        5,
			ErrorAfterChunks: 2,
			MidStreamError:   llm.ErrStreamClosed,
		})

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	chunkCount := 0

	var lastErr error

	for _, err := range model.GenerateEvents(context.Background(), req) {
		if err != nil {
			// Should error after 2 chunks
			lastErr = err
			break
		}

		chunkCount++
	}

	// Should receive exactly 2 chunks before the error
	assert.Equal(t, 2, chunkCount, "stream should emit 2 chunks before erroring")
	require.ErrorIs(t, lastErr, llm.ErrStreamClosed)
}

func TestFakeModel_FallbackBehavior(t *testing.T) {
	t.Parallel()

	// Model with no rules - should use fallback
	model := llmtesting.NewFakeModel()

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("Hello world")}},
		},
	}

	resp, err := model.Generate(context.Background(), req)
	require.NoError(t, err)
	// Fallback echoes the last user message
	assert.Equal(t, "Hello world", resp.TextContent())
	assert.Equal(t, llm.FinishReasonStop, resp.FinishReason)
}

func TestFakeModel_LatencySimulation(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel(
		llmtesting.WithLatency(llmtesting.LatencyProfile{
			Base:     50 * time.Millisecond,
			PerToken: 5 * time.Millisecond,
		}),
	)

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	start := time.Now()
	_, err := model.Generate(context.Background(), req)
	require.NoError(t, err)

	elapsed := time.Since(start)

	// Should take at least the base latency
	assert.GreaterOrEqual(t, elapsed, 50*time.Millisecond)
}

func TestFakeModel_ModelCapabilities(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel(
		llmtesting.WithModelName("test-model-pro"),
		llmtesting.WithCapabilities(llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			StructuredOutput: false,
			Vision:           false,
			MultiTurn:        true,
			SystemPrompts:    true,
		}),
	)

	assert.Equal(t, "test-model-pro", model.Name())

	caps := model.Capabilities()
	assert.True(t, caps.Streaming)
	assert.True(t, caps.Tools)
	assert.False(t, caps.StructuredOutput)
	assert.False(t, caps.Vision)
}

func TestFakeModel_ComplexMatchers(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(
			llmtesting.And(
				llmtesting.UserMessageContains("weather"),
				llmtesting.HasTool("get_weather"),
				llmtesting.TurnIs(0),
			),
		).
		ThenRespondWithToolCall("get_weather", map[string]any{"location": "SF"}).
		When(
			llmtesting.Or(
				llmtesting.UserMessageContains("hi"),
				llmtesting.UserMessageContains("hello"),
			),
		).
		ThenRespondText("Hello there!")

	// Test AND matcher
	req1 := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("What's the weather?")}},
		},
		Tools:    []llm.ToolDefinition{{Name: "get_weather"}},
		Metadata: map[string]string{"session_id": "test"},
	}

	resp1, err := model.Generate(context.Background(), req1)
	require.NoError(t, err)
	assert.True(t, resp1.HasToolRequests())

	// Test OR matcher
	req2 := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("hi")}},
		},
	}

	resp2, err := model.Generate(context.Background(), req2)
	require.NoError(t, err)
	assert.Equal(t, "Hello there!", resp2.TextContent())
}

func TestFakeModel_TokenCounting(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenRespondText("This is a test response")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("Hello")}},
		},
	}

	resp, err := model.Generate(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp.Usage)

	// Verify token counts are reasonable
	assert.Positive(t, resp.Usage.InputTokens)
	assert.Positive(t, resp.Usage.OutputTokens)
	assert.Equal(t, resp.Usage.InputTokens+resp.Usage.OutputTokens, resp.Usage.TotalTokens)
}

// TestFakeModel_ToolCallingLoop demonstrates an integration test with agent-like tool calling.
func TestFakeModel_ToolCallingLoop(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		Scenario("calculator", func(s *llmtesting.ScenarioBuilder) {
			// Turn 0: Request calculation
			s.OnTurn(0).
				When(llmtesting.HasTool("calculate")).
				ThenRespondWithToolCall("calculate", map[string]any{
					"expression": "2 + 2",
				})

			// Turn 1: Respond with result
			s.OnTurn(1).
				ThenRespondText("The answer is 4")
		})

	ctx := context.Background()

	// Turn 0: Model requests tool call
	req1 := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("What is 2 + 2?")}},
		},
		Tools:    []llm.ToolDefinition{{Name: "calculate"}},
		Metadata: map[string]string{"session_id": "calc-session"},
	}

	resp1, err := model.Generate(ctx, req1)
	require.NoError(t, err)
	require.True(t, resp1.HasToolRequests())

	// Turn 1: Provide tool result, model responds with final answer
	req2 := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("What is 2 + 2?")}},
			resp1.Message,
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewToolResponsePart(&llm.ToolResponse{
				ID:     resp1.ToolRequests()[0].ID,
				Name:   "calculate",
				Result: []byte(`{"result": 4}`),
			})}},
		},
		Tools:    []llm.ToolDefinition{{Name: "calculate"}},
		Metadata: map[string]string{"session_id": "calc-session"},
	}

	resp2, err := model.Generate(ctx, req2)
	require.NoError(t, err)
	assert.Equal(t, "The answer is 4", resp2.TextContent())
}
