package conformance

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Suite provides generic conformance tests for agents with any provider.
// Provider packages should create fixtures and run this suite to verify
// their provider works correctly with the agent layer.
type Suite struct {
	suite.Suite

	fixture Fixture
}

// NewSuite creates a new agent conformance test suite with the given fixture.
func NewSuite(fixture Fixture) *Suite {
	return &Suite{
		fixture: fixture,
	}
}

// TestBasicToolCalling tests that an agent can make a single tool call with proper arguments.
// This is the critical test that catches streaming bugs where tool arguments are lost.
func (s *Suite) TestBasicToolCalling() {
	// Create tool registry with calculator
	registry := tool.NewRegistry(tool.RegistryConfig{})
	calculator := NewCalculatorTool()
	err := registry.Register(calculator)
	s.Require().NoError(err)

	// Create agent
	ag, err := s.fixture.StandardAgent(registry)
	if ag == nil {
		s.T().Skip("No standard agent available")
	}

	s.Require().NoError(err)

	// Create session and execute
	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("What is 42 plus 58?"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Snapshot{Name: "test-agent", Description: "Test agent for conformance"})

	events := collectEvents(s.T(), ag.Run(s.T().Context(), inv))

	// Extract event types
	toolCallEvents := filterEvents[agent.ToolRequestEvent](events)
	toolResultEvents := filterEvents[agent.ToolResponseEvent](events)
	messageEvents := filterEvents[agent.MessageEvent](events)
	endEvent := findInvocationEndEvent(events)

	// Verify tool was called
	s.Require().NotEmpty(toolCallEvents, "agent should have called the calculator tool")
	s.Equal("add_numbers", toolCallEvents[0].Request.Name)

	// CRITICAL: Verify tool arguments are populated (not empty)
	var toolArgs struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	err = json.Unmarshal(toolCallEvents[0].Request.Arguments, &toolArgs)
	s.Require().NoError(err, "tool arguments should be valid JSON")
	s.NotZero(toolArgs.A, "tool argument 'a' must not be zero")
	s.NotZero(toolArgs.B, "tool argument 'b' must not be zero")

	// Verify tool result received
	s.Require().NotEmpty(toolResultEvents, "should receive tool result")
	s.Empty(toolResultEvents[0].Response.Error, "tool execution should succeed")

	// Verify final response
	s.Require().NotNil(endEvent, "should receive InvocationEndEvent")
	s.Equal(agent.FinishReasonStop, endEvent.FinishReason)

	// Should have final message with text
	s.Require().NotEmpty(messageEvents, "should have message events")
	lastMessage := messageEvents[len(messageEvents)-1]
	finalText := lastMessage.Response.TextContent()
	s.NotEmpty(finalText, "should have final response text")

	// Verify usage tracking
	s.Require().NotNil(endEvent.Usage)
	s.Positive(endEvent.Usage.TotalTokens)
}

// TestMultiTurnToolExecution tests that an agent can execute tools across multiple turns.
func (s *Suite) TestMultiTurnToolExecution() {
	// Create tool registry with calculator
	registry := tool.NewRegistry(tool.RegistryConfig{})
	calculator := NewCalculatorTool()
	err := registry.Register(calculator)
	s.Require().NoError(err)

	// Create agent with limited turns
	ag, err := s.fixture.StandardAgent(registry)
	if ag == nil {
		s.T().Skip("No standard agent available")
	}

	s.Require().NoError(err)

	// Create session and execute
	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("What is 15 plus 27?"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Snapshot{Name: "test-agent", Description: "Test agent for conformance"})

	events := collectEvents(s.T(), ag.Run(s.T().Context(), inv))

	// Extract events
	toolCallEvents := filterEvents[agent.ToolRequestEvent](events)
	endEvent := findInvocationEndEvent(events)

	// Should complete successfully
	s.Require().NotNil(endEvent)
	s.NotEqual(agent.FinishReasonMaxTurns, endEvent.FinishReason, "should not hit max turns")

	// Should have made at least one tool call
	s.NotEmpty(toolCallEvents, "should have called calculator")

	// Turn count should be reasonable (at least 2 turns: initial + tool response)
	finalTurn := endEvent.GetEnvelope().Turn
	s.GreaterOrEqual(finalTurn, 1, "should take at least 2 turns")
	s.LessOrEqual(finalTurn, 5, "should complete within reasonable turns")
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
