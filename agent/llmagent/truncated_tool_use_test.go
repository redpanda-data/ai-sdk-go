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
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// TestRun_TruncatedTurnStopsBeforeToolExecution covers the end-to-end contract
// of the truncation fix. When a streaming turn is cut off mid-parallel-tool-use:
//
//   - the provider layer drops the partial tool_use block and surfaces
//     FinishReasonLength (see providers/anthropic/stream_partial_test.go for
//     the wire-level reproducer);
//   - the agent loop must recognise Length as terminal even when completed
//     tool_use blocks are present, terminate without executing any of them,
//     and emit a StatusStageTurnCompleted event so the caller knows the turn
//     was cut short.
//
// The bug that motivated this test: providers used to clamp FinishReason to
// ToolCalls whenever any tool_use block made it through, masking the Length
// signal. The agent loop then happily executed N-1 of N intended tool calls
// and the model had to guess why the Nth result was missing. This test locks
// in the post-fix behaviour so the mask can't come back.
func TestRun_TruncatedTurnStopsBeforeToolExecution(t *testing.T) {
	t.Parallel()

	// Two tool calls in the response simulate "N-1 of N completed" — the
	// partial Nth block is what the provider already dropped at stream
	// finalisation, so it never shows up here. FinishReasonLength is what
	// the fixed response mapper now lets through instead of clamping to
	// ToolCalls.
	toolCalls := []*llm.ToolRequest{
		{
			ID:        "call_1",
			Name:      "db_query",
			Arguments: json.RawMessage(`{"q":"SELECT 1"}`),
		},
		{
			ID:        "call_2",
			Name:      "db_query",
			Arguments: json.RawMessage(`{"q":"SELECT 2"}`),
		},
	}

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			content := make([]*llm.Part, 0, len(toolCalls))
			for _, tr := range toolCalls {
				content = append(content, llm.NewToolRequestPart(tr))
			}

			return &llm.Response{
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: content,
				},
				FinishReason: llm.FinishReasonLength,
			}, nil
		})

	ag, err := llmagent.New("test-agent", "You are a helpful assistant", model)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("analyse the db"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Info{})

	events := collectEvents(t, ag.Run(t.Context(), inv))

	// End-of-invocation must report Length so callers can distinguish "we
	// ran out of budget" from a normal completion.
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonLength, endEvent.FinishReason,
		"Length must propagate terminally; ToolCalls masquerading as Length would trigger tool execution")

	// No tools were executed. The whole point of the fix is that the loop
	// bails on Length instead of treating the completed tool_use blocks as a
	// normal parallel-tool-use turn.
	toolReqEvents := filterEvents[agent.ToolRequestEvent](events)
	assert.Empty(t, toolReqEvents, "no ToolRequestEvent should fire when the turn was truncated")

	toolRespEvents := filterEvents[agent.ToolResponseEvent](events)
	assert.Empty(t, toolRespEvents, "no ToolResponseEvent should fire when the turn was truncated")

	// The assistant message still gets persisted to the session — the caller
	// may want to inspect the completed tool_use blocks to decide whether to
	// retry with higher max_tokens, surface to the user, etc.
	require.Len(t, sess.Messages, 2, "assistant response must be appended to session")
	assistant := sess.Messages[1]
	assert.Equal(t, llm.RoleAssistant, assistant.Role)

	var persistedToolCalls []string

	for _, part := range assistant.Content {
		if part.IsToolRequest() {
			persistedToolCalls = append(persistedToolCalls, part.ToolRequest.ID)
		}
	}

	assert.Equal(t, []string{"call_1", "call_2"}, persistedToolCalls,
		"the tool_use blocks that did complete must still reach the session so the caller can inspect them")

	// A StatusStageTurnCompleted event with the length detail lets
	// observability catch the truncation without the caller having to
	// parse FinishReason.
	statusEvents := filterEvents[agent.StatusEvent](events)

	var sawLengthStatus bool

	for _, s := range statusEvents {
		if s.Stage == agent.StatusStageTurnCompleted && containsLengthMention(s.Details) {
			sawLengthStatus = true
			break
		}
	}

	assert.True(t, sawLengthStatus, "expected a StatusStageTurnCompleted with a length-limit detail, got %+v", statusEvents)
}

func containsLengthMention(s string) bool {
	// The current implementation emits "turn N completed - length limit".
	// Match loosely so the test isn't fragile to wording tweaks.
	lower := strings.ToLower(s)
	for _, needle := range []string{"length", "max_tokens", "truncat"} {
		if strings.Contains(lower, needle) {
			return true
		}
	}

	return false
}
