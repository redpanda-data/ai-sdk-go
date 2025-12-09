package agenttool_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/agenttool"
)

// mockAgent is a simple test agent that returns a predefined response.
type mockAgent struct {
	name        string
	description string
	inputSchema map[string]any
	response    string
	shouldError bool
}

func (m *mockAgent) Name() string {
	return m.name
}

func (m *mockAgent) Description() string {
	return m.description
}

func (m *mockAgent) InputSchema() map[string]any {
	return m.inputSchema
}

func (m *mockAgent) Run(_ context.Context, _ *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		if m.shouldError {
			yield(nil, errors.New("mock agent error"))
			return
		}

		// Emit a message event
		msg := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(m.response))
		evt := agent.MessageEvent{
			Response: llm.Response{
				Message: msg,
			},
		}

		yield(evt, nil)

		// Emit end event
		endEvt := agent.InvocationEndEvent{
			FinishReason: agent.FinishReasonStop,
		}
		yield(endEvt, nil)
	}
}

// blockingMockAgent simulates an agent that blocks until context is cancelled.
type blockingMockAgent struct {
	mockAgent
}

func (m *blockingMockAgent) Run(ctx context.Context, _ *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		// Block until context is cancelled
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}

func TestNew(t *testing.T) {
	t.Parallel()

	mockAgent := &mockAgent{
		name:        "test-agent",
		description: "A test agent",
		response:    "test response",
	}

	agentTool := agenttool.New(mockAgent)

	require.NotNil(t, agentTool)
	assert.Implements(t, (*tool.Tool)(nil), agentTool)
}

func TestDefinition(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		agentName   string
		description string
		schema      map[string]any
	}{
		{
			name:        "basic agent",
			agentName:   "search-agent",
			description: "Searches for information",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "The search query",
					},
				},
			},
		},
		{
			name:        "nil schema",
			agentName:   "simple-agent",
			description: "Simple agent",
			schema:      nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			mockAgent := &mockAgent{
				name:        tt.agentName,
				description: tt.description,
				inputSchema: tt.schema,
				response:    "test",
			}

			agentTool := agenttool.New(mockAgent)
			def := agentTool.Definition()

			assert.Equal(t, tt.agentName, def.Name)
			assert.Equal(t, tt.description, def.Description)

			// Parameters is json.RawMessage, so unmarshal to compare
			var actualSchema map[string]any
			if def.Parameters != nil {
				err := json.Unmarshal(def.Parameters, &actualSchema)
				require.NoError(t, err)
			}

			assert.Equal(t, tt.schema, actualSchema)
		})
	}
}

func TestExecute(t *testing.T) {
	t.Parallel()

	t.Run("successful execution", func(t *testing.T) {
		t.Parallel()

		mockAgent := &mockAgent{
			name:     "test-agent",
			response: "This is the agent response",
		}

		agentTool := agenttool.New(mockAgent)

		args, _ := json.Marshal(map[string]string{"query": "test query"})
		result, err := agentTool.Execute(context.Background(), args)

		require.NoError(t, err)

		var output agenttool.Result
		err = json.Unmarshal(result, &output)
		require.NoError(t, err)
		assert.Equal(t, "This is the agent response", output.Result)
	})

	t.Run("empty args", func(t *testing.T) {
		t.Parallel()

		mockAgent := &mockAgent{
			name:     "test-agent",
			response: "Response without input",
		}

		agentTool := agenttool.New(mockAgent)

		result, err := agentTool.Execute(context.Background(), json.RawMessage("{}"))

		require.NoError(t, err)

		var output agenttool.Result
		err = json.Unmarshal(result, &output)
		require.NoError(t, err)
		assert.Equal(t, "Response without input", output.Result)
	})

	t.Run("agent error propagation", func(t *testing.T) {
		t.Parallel()

		mockAgent := &mockAgent{
			name:        "failing-agent",
			shouldError: true,
		}

		agentTool := agenttool.New(mockAgent)

		_, err := agentTool.Execute(context.Background(), json.RawMessage("{}"))

		require.Error(t, err)
		assert.Contains(t, err.Error(), "agent execution failed")
	})

	t.Run("context cancellation", func(t *testing.T) {
		t.Parallel()

		blockingAgent := &blockingMockAgent{
			mockAgent: mockAgent{name: "blocking-agent"},
		}

		agentTool := agenttool.New(blockingAgent)

		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		_, err := agentTool.Execute(ctx, json.RawMessage("{}"))

		require.Error(t, err)
		assert.Contains(t, err.Error(), "agent execution failed")
		assert.ErrorIs(t, err, context.DeadlineExceeded)
	})
}
