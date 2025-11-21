package llmagent_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// simpleModel is a minimal model for testing (no I/O).
type simpleModel struct{}

func (simpleModel) Name() string                        { return "test-model" }
func (simpleModel) Capabilities() llm.ModelCapabilities { return llm.ModelCapabilities{} }
func (simpleModel) Generate(context.Context, *llm.Request) (*llm.Response, error) {
	return nil, errors.New("not implemented")
}

func (simpleModel) GenerateEvents(context.Context, *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		yield(nil, errors.New("not implemented"))
	}
}

// TestNew_Validation tests configuration validation.
func TestNew_Validation(t *testing.T) {
	t.Parallel()

	model := simpleModel{}

	tests := []struct {
		name      string
		agentName string
		prompt    string
		model     llm.Model
		opts      []llmagent.Option
		wantErr   string
	}{
		{
			name:      "valid agent",
			agentName: "test",
			prompt:    "You are helpful",
			model:     model,
		},
		{
			name:      "missing name",
			agentName: "",
			prompt:    "You are helpful",
			model:     model,
			wantErr:   "name is required",
		},
		{
			name:      "missing prompt",
			agentName: "test",
			prompt:    "",
			model:     model,
			wantErr:   "system prompt is required",
		},
		{
			name:      "missing model",
			agentName: "test",
			prompt:    "You are helpful",
			model:     nil,
			wantErr:   "model is required",
		},
		{
			name:      "invalid MaxTurns zero",
			agentName: "test",
			prompt:    "You are helpful",
			model:     model,
			opts:      []llmagent.Option{llmagent.WithMaxTurns(0)},
			wantErr:   "maxTurns must be positive",
		},
		{
			name:      "invalid MaxTurns negative",
			agentName: "test",
			prompt:    "You are helpful",
			model:     model,
			opts:      []llmagent.Option{llmagent.WithMaxTurns(-1)},
			wantErr:   "maxTurns must be positive",
		},
		{
			name:      "invalid ToolConcurrency zero",
			agentName: "test",
			prompt:    "You are helpful",
			model:     model,
			opts:      []llmagent.Option{llmagent.WithToolConcurrency(0)},
			wantErr:   "toolConcurrency must be positive",
		},
		{
			name:      "invalid ToolConcurrency negative",
			agentName: "test",
			prompt:    "You are helpful",
			model:     model,
			opts:      []llmagent.Option{llmagent.WithToolConcurrency(-1)},
			wantErr:   "toolConcurrency must be positive",
		},
		{
			name:      "invalid interceptor - does not implement any interface",
			agentName: "test",
			prompt:    "You are helpful",
			model:     model,
			opts:      []llmagent.Option{llmagent.WithInterceptors(struct{}{})},
			wantErr:   "interceptor at index 0 does not implement any valid interface",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			ag, err := llmagent.New(tt.agentName, tt.prompt, tt.model, tt.opts...)

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				assert.Nil(t, ag)
			} else {
				require.NoError(t, err)
				require.NotNil(t, ag)
			}
		})
	}
}

// TestLLMAgent_Name verifies Name() returns configured name.
func TestLLMAgent_Name(t *testing.T) {
	t.Parallel()

	ag, err := llmagent.New("my-agent", "You are helpful", simpleModel{})
	require.NoError(t, err)
	assert.Equal(t, "my-agent", ag.Name())
}

// TestLLMAgent_Description verifies Description() returns configured description.
func TestLLMAgent_Description(t *testing.T) {
	t.Parallel()

	t.Run("with description", func(t *testing.T) {
		t.Parallel()

		ag, err := llmagent.New(
			"test-agent",
			"You are helpful",
			simpleModel{},
			llmagent.WithDescription("A test agent"),
		)
		require.NoError(t, err)
		assert.Equal(t, "A test agent", ag.Description())
	})

	t.Run("without description", func(t *testing.T) {
		t.Parallel()

		ag, err := llmagent.New("test-agent", "You are helpful", simpleModel{})
		require.NoError(t, err)
		assert.Empty(t, ag.Description())
	})
}

// TestLLMAgent_InputSchema verifies InputSchema returns valid schema.
func TestLLMAgent_InputSchema(t *testing.T) {
	t.Parallel()

	ag, err := llmagent.New("test-agent", "You are helpful", simpleModel{})
	require.NoError(t, err)

	schema := ag.InputSchema()
	require.NotNil(t, schema)
	assert.Equal(t, "object", schema["type"])

	// Verify required fields
	required, ok := schema["required"].([]string)
	require.True(t, ok)
	assert.Contains(t, required, "message")
}

// TestRun_SimpleSingleTurn tests basic single-turn execution.
func TestRun_SimpleSingleTurn(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		agentReply string
		wantFinish agent.FinishReason
	}{
		{
			name:       "basic response",
			agentReply: "Hi there! How can I help you today?",
			wantFinish: agent.FinishReasonStop,
		},
		{
			name:       "answer",
			agentReply: "2+2 equals 4",
			wantFinish: agent.FinishReasonStop,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Setup fake model
			model := fakellm.NewFakeModel()
			model.When(fakellm.Any()).
				ThenStreamText(tt.agentReply, fakellm.StreamConfig{})

			// Create agent
			ag, err := llmagent.New("test-agent", "You are a helpful assistant", model)
			require.NoError(t, err)

			// Create session and invocation context
			sess := &session.State{
				ID:       "test-session",
				Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello!"))},
			}
			invCtx := agent.NewInvocationContext(context.Background(), sess)

			// Execute
			events := collectEvents(t, ag.Run(invCtx))

			// Assert: Should have completion event
			endEvent := findInvocationEndEvent(events)
			require.NotNil(t, endEvent)
			assert.Equal(t, tt.wantFinish, endEvent.FinishReason)

			// Assert: Should have message event
			messageEvents := filterEvents[agent.MessageEvent](events)
			require.NotEmpty(t, messageEvents)
			assert.Equal(t, tt.agentReply, messageEvents[0].Response.TextContent())

			// Assert: Usage is tracked
			require.NotNil(t, endEvent.Usage)
			assert.Positive(t, endEvent.Usage.TotalTokens)
		})
	}
}

// TestRun_MultiTurnWithTools tests multi-turn execution with tool calling.
func TestRun_MultiTurnWithTools(t *testing.T) {
	t.Parallel()

	// Setup: Create a simple weather tool
	weatherTool := &mockTool{
		name: "get_weather",
		definition: llm.ToolDefinition{
			Name:        "get_weather",
			Description: "Get current weather for a location",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"location": {"type": "string"}
				},
				"required": ["location"]
			}`),
		},
		executeFn: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"temp": 72, "condition": "sunny"}`), nil
		},
	}

	registry := tool.NewRegistry(tool.RegistryConfig{})
	err := registry.Register(weatherTool)
	require.NoError(t, err)

	// Setup: Configure fake model for two-turn scenario
	model := fakellm.NewFakeModel()

	// Turn 0: Model requests tool
	model.When(fakellm.FirstTurn()).
		Times(1).
		ThenRespondWithToolCall("get_weather", map[string]any{
			"location": "San Francisco",
		})

	// Turn 1+: After tool response, model provides final answer
	model.When(fakellm.LastMessageHasToolResponse("get_weather")).
		ThenStreamText("The weather in San Francisco is 72°F and sunny.", fakellm.StreamConfig{})

	// Create agent with tools
	ag, err := llmagent.New(
		"weather-agent",
		"You are a weather assistant",
		model,
		llmagent.WithTools(registry),
	)
	require.NoError(t, err)

	// Create session and invocation context
	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("What's the weather?"))},
	}
	invCtx := agent.NewInvocationContext(context.Background(), sess)

	// Execute
	events := collectEvents(t, ag.Run(invCtx))

	// Assert: Should have exactly 2 turns (0 and 1)
	maxTurn := -1
	for _, evt := range events {
		if evt.GetEnvelope().Turn > maxTurn {
			maxTurn = evt.GetEnvelope().Turn
		}
	}

	assert.Equal(t, 1, maxTurn, "expected 2 turns (0 and 1)")

	// Assert: Should have tool call and tool result events
	toolCallEvents := filterEvents[agent.ToolRequestEvent](events)
	toolResultEvents := filterEvents[agent.ToolResponseEvent](events)

	assert.Len(t, toolCallEvents, 1)
	assert.Len(t, toolResultEvents, 1)

	// Assert: Last event should be InvocationEndEvent with stop reason
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonStop, endEvent.FinishReason)

	// Assert: Final message should contain weather info
	messageEvents := filterEvents[agent.MessageEvent](events)
	require.NotEmpty(t, messageEvents)
	lastMessage := messageEvents[len(messageEvents)-1]
	assert.Contains(t, lastMessage.Response.TextContent(), "72°F")
}

// TestRun_MaxTurnsLimit verifies agent stops when hitting max turns.
func TestRun_MaxTurnsLimit(t *testing.T) {
	t.Parallel()

	// Setup: Create a tool that always succeeds
	dummyTool := &mockTool{
		name: "dummy_tool",
		definition: llm.ToolDefinition{
			Name:        "dummy_tool",
			Description: "A test tool",
		},
		executeFn: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"status": "ok"}`), nil
		},
	}

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(dummyTool))

	// Setup: Configure model to always request tools (infinite loop without max turns)
	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWithToolCall("dummy_tool", map[string]any{})

	// Create agent with max turns = 3
	ag, err := llmagent.New(
		"test-agent",
		"You are a test assistant",
		model,
		llmagent.WithTools(registry),
		llmagent.WithMaxTurns(3),
	)
	require.NoError(t, err)

	// Create session and invocation context
	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Start"))},
	}
	invCtx := agent.NewInvocationContext(context.Background(), sess)

	// Execute
	events := collectEvents(t, ag.Run(invCtx))

	// Assert: Should hit max turns
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonMaxTurns, endEvent.FinishReason)

	// Verify model was called exactly 3 times (maxTurns)
	assert.Equal(t, 3, model.CallCount())
}

// TestExecuteTools_NoToolRegistry verifies error when tools requested but registry missing.
func TestExecuteTools_NoToolRegistry(t *testing.T) {
	t.Parallel()

	// Setup: Model that requests a tool
	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWithToolCall("some_tool", map[string]any{})

	// Create agent WITHOUT tools
	ag, err := llmagent.New("test-agent", "You are a helpful assistant", model)
	require.NoError(t, err)

	// Create session and invocation context
	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Test"))},
	}
	invCtx := agent.NewInvocationContext(context.Background(), sess)

	// Execute - expect TERMINAL error (system failure)
	var terminalErr error

	for _, err := range ag.Run(invCtx) {
		if err != nil {
			terminalErr = err
			break
		}
	}

	// Assert: Should have terminal error with ErrToolRegistry
	require.Error(t, terminalErr)
	require.ErrorIs(t, terminalErr, agent.ErrToolRegistry)
}

// TestExecuteTools_ToolError verifies individual tool errors are handled gracefully.
func TestExecuteTools_ToolError(t *testing.T) {
	t.Parallel()

	// Setup: Tool that returns an error
	failingTool := &mockTool{
		name: "failing_tool",
		definition: llm.ToolDefinition{
			Name:        "failing_tool",
			Description: "A tool that fails",
		},
		executeFn: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("tool execution failed")
		},
	}

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(failingTool))

	// Setup: Model scenario
	model := fakellm.NewFakeModel()

	// Turn 0: Request the failing tool
	model.When(fakellm.FirstTurn()).
		Times(1).
		ThenRespondWithToolCall("failing_tool", map[string]any{})

	// Turn 1+: Handle the error and provide response
	model.When(fakellm.LastMessageHasToolResponse("failing_tool")).
		ThenStreamText("I handled the tool error", fakellm.StreamConfig{})

	ag, err := llmagent.New(
		"test-agent",
		"You are a helpful assistant",
		model,
		llmagent.WithTools(registry),
	)
	require.NoError(t, err)

	// Create session and invocation context
	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Test"))},
	}
	invCtx := agent.NewInvocationContext(context.Background(), sess)

	// Execute
	events := collectEvents(t, ag.Run(invCtx))

	// Assert: Should have tool result with error
	toolResultEvents := filterEvents[agent.ToolResponseEvent](events)
	require.Len(t, toolResultEvents, 1)
	assert.NotEmpty(t, toolResultEvents[0].Response.Error)
	assert.Contains(t, toolResultEvents[0].Response.Error, "tool execution failed")
}

// TestRun_ContextCancellation verifies context cancellation is handled.
func TestRun_ContextCancellation(t *testing.T) {
	t.Parallel()

	t.Run("cancel during model call", func(t *testing.T) {
		t.Parallel()

		// Setup: Model with slow streaming
		model := fakellm.NewFakeModel()
		model.When(fakellm.Any()).
			ThenStreamText("Long response", fakellm.StreamConfig{
				ChunkSize:       5,
				InterChunkDelay: 100 * time.Millisecond,
			})

		ag, err := llmagent.New("test-agent", "You are helpful", model)
		require.NoError(t, err)

		// Create context that we'll cancel
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		sess := &session.State{
			ID:       "test-session",
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))},
		}
		invCtx := agent.NewInvocationContext(ctx, sess)

		// Start execution
		var events []agent.Event
		var gotError bool
		var cancelCalled bool
		eventIter := ag.Run(invCtx)

		// Collect first event, then cancel
		for evt, err := range eventIter {
			if err != nil {
				gotError = true
				break
			}

			events = append(events, evt)

			// Cancel after first event
			if len(events) == 1 && !cancelCalled {
				cancel()

				cancelCalled = true
			}

			// Check if we got terminal event
			if _, ok := evt.(agent.InvocationEndEvent); ok {
				break
			}
		}

		// Assert: Should either get error or InvocationEndEvent with canceled reason
		if !gotError {
			endEvent := findInvocationEndEvent(events)
			if endEvent != nil {
				assert.Equal(t, agent.FinishReasonInterrupted, endEvent.FinishReason)
			}
		}
	})

	t.Run("cancel before execution", func(t *testing.T) {
		t.Parallel()

		model := fakellm.NewFakeModel()
		model.When(fakellm.Any()).ThenStreamText("Response", fakellm.StreamConfig{})

		ag, err := llmagent.New("test-agent", "You are helpful", model)
		require.NoError(t, err)

		// Create canceled context
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		sess := &session.State{
			ID:       "test-session",
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))},
		}
		invCtx := agent.NewInvocationContext(ctx, sess)

		// Execute
		events := collectEvents(t, ag.Run(invCtx))

		// Assert: Should get canceled finish reason
		endEvent := findInvocationEndEvent(events)
		require.NotNil(t, endEvent)
		assert.Equal(t, agent.FinishReasonInterrupted, endEvent.FinishReason)
	})
}

// TestRun_EventEnvelope verifies event envelopes have correct fields.
func TestRun_EventEnvelope(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenStreamText("Response", fakellm.StreamConfig{})

	ag, err := llmagent.New("test-agent", "You are helpful", model)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))},
	}
	invCtx := agent.NewInvocationContext(context.Background(), sess)

	// Execute
	events := collectEvents(t, ag.Run(invCtx))

	// Assert: All events should have proper envelope
	require.NotEmpty(t, events)

	var invocationID string

	for i, evt := range events {
		envelope := evt.GetEnvelope()

		// InvocationID should be set and consistent
		if i == 0 {
			invocationID = envelope.InvocationID
			assert.NotEmpty(t, invocationID)
		} else {
			assert.Equal(t, invocationID, envelope.InvocationID)
		}

		// SessionID should match
		assert.Equal(t, "test-session", envelope.SessionID)

		// Turn should be >= 0
		assert.GreaterOrEqual(t, envelope.Turn, 0)

		// Timestamp should be set and in UTC
		assert.False(t, envelope.At.IsZero())
		assert.Equal(t, "UTC", envelope.At.Location().String())
	}
}

// TestRun_UsageTracking verifies token usage is tracked correctly.
func TestRun_UsageTracking(t *testing.T) {
	t.Parallel()

	t.Run("single turn", func(t *testing.T) {
		t.Parallel()

		model := fakellm.NewFakeModel()
		model.When(fakellm.Any()).ThenStreamText("Response", fakellm.StreamConfig{})

		ag, err := llmagent.New("test-agent", "You are helpful", model)
		require.NoError(t, err)

		sess := &session.State{
			ID:       "test-session",
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))},
		}
		invCtx := agent.NewInvocationContext(context.Background(), sess)

		events := collectEvents(t, ag.Run(invCtx))

		endEvent := findInvocationEndEvent(events)
		require.NotNil(t, endEvent)
		require.NotNil(t, endEvent.Usage)
		assert.Positive(t, endEvent.Usage.TotalTokens)
	})

	t.Run("multi turn", func(t *testing.T) {
		t.Parallel()

		// Setup tool
		dummyTool := &mockTool{
			name: "dummy_tool",
			definition: llm.ToolDefinition{
				Name: "dummy_tool",
			},
			executeFn: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"status": "ok"}`), nil
			},
		}

		registry := tool.NewRegistry(tool.RegistryConfig{})
		require.NoError(t, registry.Register(dummyTool))

		model := fakellm.NewFakeModel()
		// Turn 0: Request tool
		model.When(fakellm.FirstTurn()).Times(1).
			ThenRespondWithToolCall("dummy_tool", map[string]any{})
		// Turn 1: Final response
		model.When(fakellm.TurnGreaterThan(0)).
			ThenStreamText("Done", fakellm.StreamConfig{})

		ag, err := llmagent.New(
			"test-agent",
			"You are helpful",
			model,
			llmagent.WithTools(registry),
		)
		require.NoError(t, err)

		sess := &session.State{
			ID:       "test-session",
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))},
		}
		invCtx := agent.NewInvocationContext(context.Background(), sess)

		events := collectEvents(t, ag.Run(invCtx))

		// Verify usage accumulates across turns
		endEvent := findInvocationEndEvent(events)
		require.NotNil(t, endEvent)
		require.NotNil(t, endEvent.Usage)
		assert.Positive(t, endEvent.Usage.TotalTokens)

		// Should be more than single turn since we had 2 model calls
		assert.Equal(t, 2, model.CallCount())
	})
}

// TestRun_StreamingDeltas verifies streaming model emits deltas.
func TestRun_StreamingDeltas(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenStreamText("Hello world!", fakellm.StreamConfig{
			ChunkSize:       5, // Small chunks to get multiple deltas
			InterChunkDelay: time.Millisecond,
		})

	ag, err := llmagent.New("test-agent", "You are helpful", model)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))},
	}
	invCtx := agent.NewInvocationContext(context.Background(), sess)

	// Execute and collect delta events
	events := collectEvents(t, ag.Run(invCtx))

	deltaEvents := filterEvents[agent.AssistantDeltaEvent](events)
	assert.Greater(t, len(deltaEvents), 1, "should have multiple streaming deltas")

	// Reconstruct message from deltas
	var reconstructed string

	var reconstructedSb714 strings.Builder
	for _, delta := range deltaEvents {
		reconstructedSb714.WriteString(delta.Delta.Part.Text)
	}

	reconstructed += reconstructedSb714.String()

	assert.Equal(t, "Hello world!", reconstructed)
}

// TestRun_EventOrdering verifies canonical event sequences for both no-tool and tool paths.
func TestRun_EventOrdering(t *testing.T) {
	t.Parallel()

	t.Run("NoTools", func(t *testing.T) {
		t.Parallel()

		// Setup: Model that responds without tools
		model := fakellm.NewFakeModel()
		model.When(fakellm.Any()).ThenStreamText("Simple response", fakellm.StreamConfig{})

		ag, err := llmagent.New("test-agent", "You are helpful", model)
		require.NoError(t, err)

		sess := &session.State{
			ID:       "test-session",
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello"))},
		}
		invCtx := agent.NewInvocationContext(context.Background(), sess)

		// Execute and collect events
		events := collectEvents(t, ag.Run(invCtx))

		// Verify canonical sequence: StatusEvent → MessageEvent → InvocationEndEvent
		require.GreaterOrEqual(t, len(events), 3, "should have at least status, message, and end events")

		// Find event indices
		statusIdx, messageIdx, endIdx := -1, -1, -1

		for i, evt := range events {
			switch evt.(type) {
			case agent.StatusEvent:
				if statusIdx == -1 {
					statusIdx = i
				}
			case agent.MessageEvent:
				messageIdx = i // Keep last message
			case agent.InvocationEndEvent:
				endIdx = i
			}
		}

		// Assert ordering
		require.NotEqual(t, -1, statusIdx, "should have StatusEvent")
		require.NotEqual(t, -1, messageIdx, "should have MessageEvent")
		require.NotEqual(t, -1, endIdx, "should have InvocationEndEvent")

		assert.Less(t, statusIdx, messageIdx, "StatusEvent should come before MessageEvent")
		assert.Less(t, messageIdx, endIdx, "MessageEvent should come before InvocationEndEvent")
	})

	t.Run("WithTools", func(t *testing.T) {
		t.Parallel()

		// Setup: Create a simple tool
		weatherTool := &mockTool{
			name: "get_weather",
			definition: llm.ToolDefinition{
				Name:        "get_weather",
				Description: "Get the weather",
			},
			executeFn: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"temperature": "72°F"}`), nil
			},
		}

		registry := tool.NewRegistry(tool.RegistryConfig{})
		require.NoError(t, registry.Register(weatherTool))

		// Setup: Model that first calls tool, then responds
		model := fakellm.NewFakeModel()

		// Turn 0: Model requests tool
		model.When(fakellm.FirstTurn()).
			Times(1).
			ThenRespondWithToolCall("get_weather", map[string]any{})

		// Turn 1+: After tool response, model provides final answer
		model.When(fakellm.LastMessageHasToolResponse("get_weather")).
			ThenStreamText("The weather is nice!", fakellm.StreamConfig{})

		ag, err := llmagent.New(
			"test-agent",
			"You are a weather assistant",
			model,
			llmagent.WithTools(registry),
		)
		require.NoError(t, err)

		sess := &session.State{
			ID:       "test-session",
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("What's the weather?"))},
		}
		invCtx := agent.NewInvocationContext(context.Background(), sess)

		// Execute and collect events
		events := collectEvents(t, ag.Run(invCtx))

		// Verify canonical sequence:
		// StatusEvent → MessageEvent (tool calls) → ToolRequestEvent → ToolResponseEvent → MessageEvent (final) → InvocationEndEvent
		require.GreaterOrEqual(t, len(events), 6, "should have status, message, tool call, tool result, message, and end events")

		// Find event indices
		statusIdx, firstMessageIdx, toolCallIdx, toolResultIdx, finalMessageIdx, endIdx := -1, -1, -1, -1, -1, -1

		for i, evt := range events {
			switch evt.(type) {
			case agent.StatusEvent:
				if statusIdx == -1 {
					statusIdx = i
				}
			case agent.MessageEvent:
				if firstMessageIdx == -1 {
					firstMessageIdx = i
				}

				finalMessageIdx = i // Keep updating to get last message
			case agent.ToolRequestEvent:
				if toolCallIdx == -1 {
					toolCallIdx = i
				}
			case agent.ToolResponseEvent:
				if toolResultIdx == -1 {
					toolResultIdx = i
				}
			case agent.InvocationEndEvent:
				endIdx = i
			}
		}

		// Assert all events present
		require.NotEqual(t, -1, statusIdx, "should have StatusEvent")
		require.NotEqual(t, -1, firstMessageIdx, "should have first MessageEvent")
		require.NotEqual(t, -1, toolCallIdx, "should have ToolRequestEvent")
		require.NotEqual(t, -1, toolResultIdx, "should have ToolResponseEvent")
		require.NotEqual(t, -1, finalMessageIdx, "should have final MessageEvent")
		require.NotEqual(t, -1, endIdx, "should have InvocationEndEvent")

		// Assert canonical ordering
		assert.Less(t, statusIdx, firstMessageIdx, "StatusEvent should come before first MessageEvent")
		assert.Less(t, firstMessageIdx, toolCallIdx, "MessageEvent should come before ToolRequestEvent")
		assert.Less(t, toolCallIdx, toolResultIdx, "ToolRequestEvent should come before ToolResponseEvent")
		assert.Less(t, toolResultIdx, finalMessageIdx, "ToolResponseEvent should come before final MessageEvent")
		assert.Less(t, finalMessageIdx, endIdx, "Final MessageEvent should come before InvocationEndEvent")
	})
}

// Helper functions

// collectEvents collects all events from an iterator into a slice.
func collectEvents(t *testing.T, iter func(func(agent.Event, error) bool)) []agent.Event {
	t.Helper()

	var events []agent.Event //nolint:prealloc // size unknown, depends on iterator

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

// findInvocationEndEvent finds the InvocationEndEvent in events.
func findInvocationEndEvent(events []agent.Event) *agent.InvocationEndEvent {
	for i := len(events) - 1; i >= 0; i-- {
		if endEvt, ok := events[i].(agent.InvocationEndEvent); ok {
			return &endEvt
		}
	}

	return nil
}

// mockTool is a test implementation of the Tool interface.
type mockTool struct {
	name       string
	definition llm.ToolDefinition
	executeFn  func(context.Context, json.RawMessage) (json.RawMessage, error)
}

func (m *mockTool) Definition() llm.ToolDefinition {
	return m.definition
}

func (m *mockTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	if m.executeFn != nil {
		return m.executeFn(ctx, args)
	}

	return json.RawMessage(`{}`), nil
}
