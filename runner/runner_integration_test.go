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

package runner_test

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
	"github.com/redpanda-data/ai-sdk-go/runner"
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

func (*calculatorTool) IsAsynchronous() bool { return false }

func (*calculatorTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	// Parse arguments
	var params struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return nil, err
	}

	// Calculate result
	result := params.A + params.B

	// Return result as JSON
	response := map[string]any{
		"result": result,
		"a":      params.A,
		"b":      params.B,
	}

	return json.Marshal(response)
}

// TestRunner_Integration_WithTools tests runner with agent that uses tools.
func TestRunner_Integration_WithTools(t *testing.T) {
	t.Parallel()

	// Get API key or skip test
	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	ctx, cancel := context.WithTimeout(context.Background(), integrationTestTimeout)
	defer cancel()

	// Create OpenAI provider
	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName)
	require.NoError(t, err)

	// Create tool registry with calculator
	registry := tool.NewRegistry(tool.RegistryConfig{})
	err = registry.Register(&calculatorTool{})
	require.NoError(t, err)

	// Create agent with tools
	ag, err := llmagent.New(
		"calculator-agent",
		"You are a calculator assistant. You MUST call the add_numbers tool and MUST NOT compute the answer yourself.",
		model,
		llmagent.WithTools(registry),
		llmagent.WithMaxTurns(5),
	)
	require.NoError(t, err)

	// Create runner
	store := session.NewInMemoryStore()
	r, err := runner.New(ag, store)
	require.NoError(t, err)

	// Execute with tool calling
	t.Log("Executing with tool calling...")

	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart("Please calculate 10 + 20 for me."))
	events := collectEventsIntegration(t, r.Run(ctx, "", "test-session-tools", userMsg))

	// Verify tool was called
	toolCallEvents := filterEventsIntegration[agent.ToolRequestEvent](events)
	require.NotEmpty(t, toolCallEvents, "should have called the calculator tool")

	toolResultEvents := filterEventsIntegration[agent.ToolResponseEvent](events)
	require.NotEmpty(t, toolResultEvents, "should have tool result")

	// Verify completion
	endEvent := findInvocationEndEventIntegration(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonStop, endEvent.FinishReason)

	// Verify session was saved with tool messages
	sess, err := store.Load(ctx, "test-session-tools")
	require.NoError(t, err)
	assert.Greater(t, len(sess.Messages), 2, "should have user, assistant (with tool request), tool response, and final assistant message")

	t.Logf("Tool calling completed successfully with %d messages", len(sess.Messages))
}

// Helper functions

// collectEventsIntegration collects all events from an iterator.
func collectEventsIntegration(t *testing.T, iter func(func(agent.Event, error) bool)) []agent.Event {
	t.Helper()

	var events []agent.Event //nolint:prealloc // size unknown, depends on iterator

	for evt, err := range iter {
		require.NoError(t, err, "unexpected error in event stream")

		events = append(events, evt)
	}

	return events
}

// filterEventsIntegration returns all events of a specific type.
func filterEventsIntegration[T agent.Event](events []agent.Event) []T {
	var filtered []T

	for _, evt := range events {
		if typed, ok := evt.(T); ok {
			filtered = append(filtered, typed)
		}
	}

	return filtered
}

// findInvocationEndEventIntegration finds the InvocationEndEvent in events.
func findInvocationEndEventIntegration(events []agent.Event) *agent.InvocationEndEvent {
	for i := len(events) - 1; i >= 0; i-- {
		if endEvt, ok := events[i].(agent.InvocationEndEvent); ok {
			return &endEvt
		}
	}

	return nil
}
