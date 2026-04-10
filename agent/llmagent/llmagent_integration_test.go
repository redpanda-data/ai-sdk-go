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

package llmagent_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const (
	// Integration test timeout.
	integrationTestTimeout = 60 * time.Second
)

// calculatorTool is a simple test tool that adds two numbers.
type calculatorTool struct{}

func (*calculatorTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "add_numbers",
		Description: "Adds two numbers together and returns the result",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"a": {
					"type": "number",
					"description": "The first number to add"
				},
				"b": {
					"type": "number",
					"description": "The second number to add"
				}
			},
			"required": ["a", "b"]
		}`),
	}
}

func (*calculatorTool) Execute(_ context.Context, args json.RawMessage) (tool.Result, error) {
	// Parse arguments
	var params struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return tool.Result{}, err
	}

	// Calculate result
	result := params.A + params.B

	// Return result as JSON
	response := map[string]any{
		"result": result,
		"a":      params.A,
		"b":      params.B,
	}

	encoded, err := json.Marshal(response)
	if err != nil {
		return tool.Result{}, err
	}

	return tool.Result{Output: encoded}, nil
}

// TestLLMAgent_Integration_ToolCalling tests end-to-end tool calling with real OpenAI API.
func TestLLMAgent_Integration_ToolCalling(t *testing.T) {
	t.Parallel()

	// Get API key or skip test
	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	ctx, cancel := context.WithTimeout(context.Background(), integrationTestTimeout)
	defer cancel()

	// Create OpenAI provider
	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err, "failed to create OpenAI provider")

	// Create model - use a model that supports tool calling
	model, err := provider.NewModel(openaitest.TestModelName)
	require.NoError(t, err, "failed to create model")

	// Create tool registry and register calculator tool
	registry := tool.NewRegistry(tool.RegistryConfig{})
	calculator := &calculatorTool{}
	err = registry.Register(calculator)
	require.NoError(t, err, "failed to register calculator tool")

	// Create agent with tools
	ag, err := llmagent.New(
		"calculator-agent",
		"You are a calculator assistant. You MUST call the add_numbers tool and MUST NOT compute the answer yourself.",
		model,
		llmagent.WithTools(registry),
		llmagent.WithMaxTurns(5), // Allow multiple turns for tool calling
	)
	require.NoError(t, err, "failed to create agent")

	// Create session and invocation metadata
	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("What is 42 plus 58?"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Info{})

	// Execute
	events := collectEvents(t, ag.Run(ctx, inv))

	// Collect different event types
	toolCallEvents := filterEvents[agent.ToolRequestEvent](events)
	toolResultEvents := filterEvents[agent.ToolResponseEvent](events)
	messageEvents := filterEvents[agent.MessageEvent](events)
	endEvent := findInvocationEndEvent(events)

	// Assertions

	// 1. Verify tool was called
	require.NotEmpty(t, toolCallEvents, "agent should have called the calculator tool")
	assert.Equal(t, "add_numbers", toolCallEvents[0].Request.Name, "should call add_numbers tool")

	// 2. Verify tool arguments contain the numbers
	var toolArgs struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	err = json.Unmarshal(toolCallEvents[0].Request.Arguments, &toolArgs)
	require.NoError(t, err, "tool arguments should be valid JSON")
	assert.InDelta(t, 42.0, toolArgs.A, 0.0, "first argument should be 42")
	assert.InDelta(t, 58.0, toolArgs.B, 0.0, "second argument should be 58")

	// 3. Verify tool result was received
	require.NotEmpty(t, toolResultEvents, "should have received tool result")
	assert.Empty(t, toolResultEvents[0].Response.Error, "tool execution should succeed")

	// Parse and verify the result
	var toolResult struct {
		Result float64 `json:"result"`
		A      float64 `json:"a"`
		B      float64 `json:"b"`
	}
	err = json.Unmarshal(toolResultEvents[0].Response.Result, &toolResult)
	require.NoError(t, err, "tool result should be valid JSON")
	assert.InDelta(t, 100.0, toolResult.Result, 0.0, "calculator should return correct sum")

	// 4. Verify final response - assert on presence only, not content (model output is non-deterministic)
	require.NotNil(t, endEvent, "should receive InvocationEndEvent")
	assert.Equal(t, agent.FinishReasonStop, endEvent.FinishReason, "should finish normally")

	// Final message should exist with text content
	require.NotEmpty(t, messageEvents, "should have received at least one message event")
	lastMessage := messageEvents[len(messageEvents)-1]
	finalText := lastMessage.Response.TextContent()
	assert.NotEmpty(t, finalText, "should have final response text")

	// 5. Verify usage information is tracked
	require.NotNil(t, endEvent.Usage, "should track token usage")
	assert.Positive(t, endEvent.Usage.TotalTokens, "should have non-zero token usage")

	// 6. Verify turn count is reasonable
	finalTurn := endEvent.GetEnvelope().Turn
	assert.LessOrEqual(t, finalTurn, 4, "should complete within max turns (0-indexed)")
	assert.GreaterOrEqual(t, finalTurn, 1, "should take at least 2 turns (0 and 1)")

	t.Logf("Final response: %s", finalText)
	t.Logf("Tool calls: %d, Tool results: %d", len(toolCallEvents), len(toolResultEvents))
	t.Logf("Token usage: %d total tokens", endEvent.Usage.TotalTokens)
}
