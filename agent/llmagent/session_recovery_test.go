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
	_ "embed"
	"encoding/json"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/store/session/kvstore"
	llmpb "github.com/redpanda-data/ai-sdk-go/store/session/kvstore/proto/gen/go/redpanda/llm/v1"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

//go:embed testdata/session_recovery_single.json
var sessionRecoverySingleJSON []byte

//go:embed testdata/session_recovery_multiple.json
var sessionRecoveryMultipleJSON []byte

// TestSessionRecovery_AnswersWithoutExecuting loads persisted sessions with
// the shape [..., assistant(tool_request), user] and no tool result. Every
// incomplete call is answered with an "interrupted" error result inserted
// before the new user message, and no tool runs.
func TestSessionRecovery_AnswersWithoutExecuting(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		data      []byte
		wantCalls int
	}{
		{name: "single call", data: sessionRecoverySingleJSON, wantCalls: 1},
		{name: "parallel calls", data: sessionRecoveryMultipleJSON, wantCalls: 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			sess, incomplete := loadRecoverySession(t, tt.data)
			require.Len(t, incomplete, tt.wantCalls)

			var executed atomic.Int32

			registry := tool.NewRegistry(tool.RegistryConfig{})

			for _, name := range uniqueToolNames(incomplete) {
				require.NoError(t, registry.Register(&mockTool{
					definition: llm.ToolDefinition{
						Name:        name,
						Description: "Must not run during recovery",
						Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
					},
					executeFn: func(context.Context, json.RawMessage) (json.RawMessage, error) {
						executed.Add(1)

						return json.RawMessage(`{"status":"ran"}`), nil
					},
				}))
			}

			model := fakellm.NewFakeModel().When(fakellm.Any()).ThenRespondText("Noted.")

			ag, err := llmagent.New("test-agent", "You are a helpful assistant.", model, llmagent.WithTools(registry))
			require.NoError(t, err)

			lastUser := sess.Messages[len(sess.Messages)-1].TextContent()
			events := collectEvents(t, ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{})))

			endEvent := findInvocationEndEvent(events)
			require.NotNil(t, endEvent)
			assert.Equal(t, agent.FinishReasonStop, endEvent.FinishReason)
			assert.Zero(t, executed.Load(), "recovery must not execute tools")
			assert.Empty(t, filterEvents[agent.ToolRequestEvent](events), "nothing is dispatched during recovery")
			assert.Empty(t, filterEvents[agent.ToolResponseEvent](events), "nothing is dispatched during recovery")

			// The model's first request already carries the repaired tail.
			calls := model.Calls()
			require.Len(t, calls, 1)

			msgs := calls[0].Request.Messages
			require.GreaterOrEqual(t, len(msgs), 2)
			assert.Equal(t, lastUser, msgs[len(msgs)-1].TextContent(), "the new user message stays last")

			responses := msgs[len(msgs)-2].ToolResponses()
			require.Len(t, responses, len(incomplete))

			for i, resp := range responses {
				assert.Equal(t, incomplete[i].ID, resp.ID)
				assert.Equal(t, incomplete[i].Name, resp.Name)
				assert.True(t, resp.IsError)
				assert.Contains(t, string(resp.Result), `"interrupted"`)
			}
		})
	}
}

// An agent without a tool registry can still repair a session that carries
// an incomplete call, because nothing needs to run.
func TestSessionRecovery_NoToolRegistry(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel().When(fakellm.Any()).ThenRespondText("Noted.")

	ag, err := llmagent.New("test-agent", "You are a helpful assistant.", model)
	require.NoError(t, err)

	sess := &session.State{
		ID: "no-registry",
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Look it up.")),
			llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("call-1", "lookup", json.RawMessage(`{}`))),
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("thanks")),
		},
	}

	events := collectEvents(t, ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{})))

	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonStop, endEvent.FinishReason)
	require.Len(t, sess.Messages[2].ToolResponses(), 1)
	assert.Equal(t, "thanks", sess.Messages[3].TextContent())
}

// A consumer that stops iterating when it sees a ToolRequestEvent, for
// example because the user declined, leaves the call without a result. The
// next message must not run it.
func TestSessionRecovery_DeclinedCallNeverRuns(t *testing.T) {
	t.Parallel()

	refund := &countingTool{name: "issue_refund"}
	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(refund))

	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWithToolCall("issue_refund", map[string]any{"order": "9999"})
	model.When(fakellm.Any()).ThenRespondText("Okay, I won't refund it.")

	ag, err := llmagent.New("support-agent", "You are a support assistant.", model, llmagent.WithTools(registry))
	require.NoError(t, err)

	store := session.NewInMemoryStore()
	run, err := runner.New(ag, store)
	require.NoError(t, err)

	// First run: decline by stopping iteration at the request.
	var sawRequest bool

	for evt, err := range run.Run(t.Context(), "sess-1", llm.NewMessage(llm.RoleUser, llm.NewTextPart("Refund order 9999."))) {
		require.NoError(t, err)

		if _, ok := evt.(agent.ToolRequestEvent); ok {
			sawRequest = true

			break
		}
	}

	require.True(t, sawRequest)

	// Second run: an unrelated message.
	for _, err := range run.Run(t.Context(), "sess-1", llm.NewMessage(llm.RoleUser, llm.NewTextPart("never mind"))) {
		require.NoError(t, err)
	}

	assert.Zero(t, refund.executed.Load(), "a declined call must not run on the next message")

	saved, err := store.Load(t.Context(), "sess-1")
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(saved.Messages), 4)

	responses := saved.Messages[2].ToolResponses()
	require.Len(t, responses, 1)
	assert.True(t, responses[0].IsError)
	assert.Contains(t, string(responses[0].Result), `"interrupted"`)
	assert.Equal(t, "never mind", saved.Messages[3].TextContent())
}

// When the model still wants an interrupted call, it requests it again, and
// that request is an ordinary call: announced with a ToolRequestEvent before
// it runs.
func TestSessionRecovery_RetryIsAnnouncedBeforeRunning(t *testing.T) {
	t.Parallel()

	refund := &countingTool{name: "issue_refund"}
	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(refund))

	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWithToolCall("issue_refund", map[string]any{"order": "9999"})
	model.When(fakellm.Any()).ThenRespondText("Refunded.")

	ag, err := llmagent.New("support-agent", "You are a support assistant.", model, llmagent.WithTools(registry))
	require.NoError(t, err)

	sess := &session.State{
		ID: "retry",
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Refund order 9999.")),
			llm.NewMessage(llm.RoleAssistant,
				llm.NewToolRequestPart("call-old", "issue_refund", json.RawMessage(`{"order":"9999"}`))),
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("did it go through?")),
		},
	}

	events := collectEvents(t, ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{})))

	reqIdx, respIdx := -1, -1

	for i, evt := range events {
		switch e := evt.(type) {
		case agent.ToolRequestEvent:
			assert.NotEqual(t, "call-old", e.Request.ID, "the interrupted call is never dispatched")

			if reqIdx < 0 {
				reqIdx = i
			}
		case agent.ToolResponseEvent:
			if respIdx < 0 {
				respIdx = i
			}
		}
	}

	require.GreaterOrEqual(t, reqIdx, 0, "the retry is announced")
	require.GreaterOrEqual(t, respIdx, 0, "the retry produces a result")
	assert.Less(t, reqIdx, respIdx, "the retry is announced before it runs")
	assert.Equal(t, int32(1), refund.executed.Load(), "only the model's new call runs")
}

// countingTool records how often it runs.
type countingTool struct {
	name     string
	executed atomic.Int32
}

func (c *countingTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        c.name,
		Description: "Counts executions",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"order":{"type":"string"}}}`),
	}
}

func (c *countingTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	c.executed.Add(1)

	return json.RawMessage(`{"status":"ok"}`), nil
}

// loadRecoverySession parses a protojson session and extracts incomplete tool requests.
func loadRecoverySession(t *testing.T, data []byte) (*session.State, []*llm.ToolRequestPart) {
	t.Helper()

	var protoState llmpb.SessionState
	err := protojson.Unmarshal(data, &protoState)
	require.NoError(t, err, "failed to parse session JSON")

	sess, err := kvstore.FromProtoSessionState(&protoState)
	require.NoError(t, err, "failed to convert proto to session")

	// Find incomplete tool requests (in second-to-last message if pattern matches)
	if len(sess.Messages) < 2 {
		return sess, nil
	}

	prevMsg := sess.Messages[len(sess.Messages)-2]
	if prevMsg.Role == llm.RoleAssistant {
		return sess, prevMsg.ToolRequests()
	}

	return sess, nil
}

// uniqueToolNames returns the distinct tool names requested, in order.
func uniqueToolNames(reqs []*llm.ToolRequestPart) []string {
	seen := make(map[string]bool)
	names := make([]string, 0, len(reqs))

	for _, req := range reqs {
		if !seen[req.Name] {
			seen[req.Name] = true
			names = append(names, req.Name)
		}
	}

	return names
}
