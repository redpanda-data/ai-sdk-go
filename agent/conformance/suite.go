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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Suite provides generic conformance tests for agents with any provider.
// Provider packages should create fixtures and run this suite to verify
// their provider works correctly with the agent layer.
type Suite struct {
	fixture Fixture
}

// NewSuite creates a new agent conformance test suite with the given fixture.
func NewSuite(fixture Fixture) *Suite {
	return &Suite{
		fixture: fixture,
	}
}

// TestBasicToolCalling tests that an agent can make a single tool call with proper arguments.
func (s *Suite) TestBasicToolCalling(t *testing.T) {
	t.Helper()
	testBasicToolCalling(t, s.fixture)
}

// TestMultiTurnToolExecution tests that an agent can execute tools across multiple turns.
func (s *Suite) TestMultiTurnToolExecution(t *testing.T) {
	t.Helper()
	testMultiTurnToolExecution(t, s.fixture)
}

// testBasicToolCalling contains the shared implementation for TestBasicToolCalling.
func testBasicToolCalling(t *testing.T, fixture Fixture) {
	t.Helper()

	// Create tool registry with calculator
	registry := tool.NewRegistry(tool.RegistryConfig{})
	calculator := NewCalculatorTool()
	err := registry.Register(calculator)
	require.NoError(t, err)

	// Create agent
	ag, err := fixture.StandardAgent(registry)
	if ag == nil {
		t.Skip("No standard agent available")
	}

	require.NoError(t, err)

	// Create session and execute
	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("What is 42 plus 58?"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Info{Name: "test-agent", Description: "Test agent for conformance"})

	events := collectEvents(t, ag.Run(t.Context(), inv))

	// Extract event types
	toolCallEvents := filterEvents[agent.ToolRequestEvent](events)
	toolResultEvents := filterEvents[agent.ToolResponseEvent](events)
	messageEvents := filterEvents[agent.MessageEvent](events)
	endEvent := findInvocationEndEvent(events)

	// Verify tool was called
	require.NotEmpty(t, toolCallEvents, "agent should have called the calculator tool")
	assert.Equal(t, "add_numbers", toolCallEvents[0].Request.Name)

	// CRITICAL: Verify tool arguments are populated (not empty)
	var toolArgs struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	err = json.Unmarshal(toolCallEvents[0].Request.Arguments, &toolArgs)
	require.NoError(t, err, "tool arguments should be valid JSON")
	assert.NotZero(t, toolArgs.A, "tool argument 'a' must not be zero")
	assert.NotZero(t, toolArgs.B, "tool argument 'b' must not be zero")

	// Verify tool result received
	require.NotEmpty(t, toolResultEvents, "should receive tool result")
	assert.Empty(t, toolResultEvents[0].Response.Error, "tool execution should succeed")

	// Verify final response
	require.NotNil(t, endEvent, "should receive InvocationEndEvent")
	assert.Equal(t, agent.FinishReasonStop, endEvent.FinishReason)

	// Should have final message with text
	require.NotEmpty(t, messageEvents, "should have message events")
	lastMessage := messageEvents[len(messageEvents)-1]
	finalText := lastMessage.Response.TextContent()
	assert.NotEmpty(t, finalText, "should have final response text")

	// Verify usage tracking
	require.NotNil(t, endEvent.Usage)
	assert.Positive(t, endEvent.Usage.TotalTokens)
}

// testMultiTurnToolExecution contains the shared implementation for TestMultiTurnToolExecution.
func testMultiTurnToolExecution(t *testing.T, fixture Fixture) {
	t.Helper()

	// Create tool registry with calculator
	registry := tool.NewRegistry(tool.RegistryConfig{})
	calculator := NewCalculatorTool()
	err := registry.Register(calculator)
	require.NoError(t, err)

	// Create agent with limited turns
	ag, err := fixture.StandardAgent(registry)
	if ag == nil {
		t.Skip("No standard agent available")
	}

	require.NoError(t, err)

	// Create session and execute
	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("What is 15 plus 27?"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Info{Name: "test-agent", Description: "Test agent for conformance"})

	events := collectEvents(t, ag.Run(t.Context(), inv))

	// Extract events
	toolCallEvents := filterEvents[agent.ToolRequestEvent](events)
	endEvent := findInvocationEndEvent(events)

	// Should complete successfully
	require.NotNil(t, endEvent)
	assert.NotEqual(t, agent.FinishReasonMaxTurns, endEvent.FinishReason, "should not hit max turns")

	// Should have made at least one tool call
	assert.NotEmpty(t, toolCallEvents, "should have called calculator")

	// Turn count should be reasonable (at least 2 turns: initial + tool response)
	finalTurn := endEvent.GetEnvelope().Turn
	assert.GreaterOrEqual(t, finalTurn, 1, "should take at least 2 turns")
	assert.LessOrEqual(t, finalTurn, 5, "should complete within reasonable turns")
}

// collectEvents collects all events from an agent run iterator.
func collectEvents(t *testing.T, iter func(func(agent.Event, error) bool)) []agent.Event {
	t.Helper()

	events := make([]agent.Event, 0)

	for evt, err := range iter {
		require.NoError(t, err, "unexpected error in event stream")

		events = append(events, evt)
	}

	return events
}

// filterEvents returns all events of a specific type.
func filterEvents[T agent.Event](events []agent.Event) []T {
	var filtered []T

	for _, evt := range events {
		if typed, ok := evt.(T); ok {
			filtered = append(filtered, typed)
		}
	}

	return filtered
}

// findInvocationEndEvent returns the InvocationEndEvent from the event list.
func findInvocationEndEvent(events []agent.Event) *agent.InvocationEndEvent {
	for i := len(events) - 1; i >= 0; i-- {
		if endEvt, ok := events[i].(agent.InvocationEndEvent); ok {
			return &endEvt
		}
	}

	return nil
}
