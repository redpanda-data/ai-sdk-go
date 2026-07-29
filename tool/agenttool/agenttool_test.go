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
	name          string
	description   string
	inputSchema   map[string]any
	response      string
	shouldError   bool
	finishReason  agent.FinishReason
	errorEventErr error // if set, emit a (non-terminal) ErrorEvent before the end event
}

func (m *mockAgent) Info() agent.Info {
	return agent.Info{
		Name:        m.name,
		Description: m.description,
	}
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

		if !yield(evt, nil) {
			return
		}

		// llmagent carries a fatal cause in a non-terminal ErrorEvent and reports
		// the terminal condition via the finish reason (both yielded with a nil
		// iterator error) — mirror that here when the test asks for it.
		if m.errorEventErr != nil {
			if !yield(agent.ErrorEvent{Err: m.errorEventErr, Message: m.errorEventErr.Error()}, nil) {
				return
			}
		}

		// Emit end event. Default to a natural stop unless the test overrides it.
		finishReason := m.finishReason
		if finishReason == "" {
			finishReason = agent.FinishReasonStop
		}

		endEvt := agent.InvocationEndEvent{
			FinishReason: finishReason,
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

	t.Run("truncated sub-agent turn is flagged", func(t *testing.T) {
		t.Parallel()

		mockAgent := &mockAgent{
			name:         "verbose-agent",
			response:     "partial answer that got cut off",
			finishReason: agent.FinishReasonLength,
		}

		agentTool := agenttool.New(mockAgent)

		result, err := agentTool.Execute(context.Background(), json.RawMessage("{}"))
		require.NoError(t, err)

		var output agenttool.Result
		require.NoError(t, json.Unmarshal(result, &output))

		assert.Equal(t, true, output.Metadata["truncated"],
			"a sub-agent turn that stopped at the output-token cap must be flagged truncated")
		assert.Contains(t, output.Result, "partial answer that got cut off",
			"the partial content the sub-agent produced must still be delivered")
	})

	t.Run("completed sub-agent turn is not flagged", func(t *testing.T) {
		t.Parallel()

		mockAgent := &mockAgent{
			name:     "test-agent",
			response: "complete answer",
			// finishReason defaults to FinishReasonStop
		}

		agentTool := agenttool.New(mockAgent)

		result, err := agentTool.Execute(context.Background(), json.RawMessage("{}"))
		require.NoError(t, err)

		var output agenttool.Result
		require.NoError(t, json.Unmarshal(result, &output))

		assert.Nil(t, output.Metadata, "a naturally completed turn must carry no markers")
		assert.Equal(t, "complete answer", output.Result)
	})

	// Terminal states other than output truncation must fail the tool call rather
	// than return a silent success, mirroring the A2A executor's mapping.
	terminalFailures := []struct {
		name         string
		finishReason agent.FinishReason
		wantContains string
	}{
		{"context overflow", agent.FinishReasonContextOverflow, "context window"},
		{"max turns", agent.FinishReasonMaxTurns, "maximum iterations"},
		{"input required", agent.FinishReasonInputRequired, "external input"},
	}

	for _, tt := range terminalFailures {
		t.Run(tt.name+" is a tool error", func(t *testing.T) {
			t.Parallel()

			mockAgent := &mockAgent{
				name:         "sub-agent",
				response:     "some partial content",
				finishReason: tt.finishReason,
			}

			agentTool := agenttool.New(mockAgent)

			_, err := agentTool.Execute(context.Background(), json.RawMessage("{}"))

			require.Error(t, err, "a %s sub-agent turn must not be returned as a success", tt.name)
			assert.Contains(t, err.Error(), tt.wantContains)
		})
	}

	t.Run("terminal error surfaces the underlying cause", func(t *testing.T) {
		t.Parallel()

		mockAgent := &mockAgent{
			name:          "sub-agent",
			response:      "some partial content",
			finishReason:  agent.FinishReasonError,
			errorEventErr: llm.ErrContentPolicyViolation,
		}

		agentTool := agenttool.New(mockAgent)

		_, err := agentTool.Execute(context.Background(), json.RawMessage("{}"))

		require.Error(t, err)
		assert.ErrorIs(t, err, llm.ErrContentPolicyViolation,
			"the sub-agent's underlying error must reach the parent, not be swallowed")
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
