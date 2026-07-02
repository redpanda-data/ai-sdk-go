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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// TestRun_EmptyContentTurnIsNotPersisted is the defense-in-depth regression for
// the empty-content replay bug. When a max_tokens cut yields an assistant
// message with zero content parts (its only block was a partial tool_use the
// provider dropped), the agent loop must NOT append that turn to the session.
// Persisting it poisons the store: the next request replays a content-less
// assistant message that Anthropic rejects with "messages.N.content: Field
// required".
//
// The truncation signal must still surface — FinishReasonLength is read from
// the response, not from the (unpersisted) session message.
func TestRun_EmptyContentTurnIsNotPersisted(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			return &llm.Response{
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: []llm.Part{}, // dropped-partial-tool_use max_tokens cut
				},
				FinishReason: llm.FinishReasonLength,
			}, nil
		})

	ag, err := llmagent.New("test-agent", "You are a helpful assistant", model)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("do the thing"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Info{})

	events := collectEvents(t, ag.Run(t.Context(), inv))

	// The empty-content assistant turn must not enter the session.
	require.Len(t, sess.Messages, 1,
		"empty-content assistant turn must not be persisted; got %+v", sess.Messages)
	assert.Equal(t, llm.RoleUser, sess.Messages[0].Role)

	// The truncation signal still propagates terminally.
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonLength, endEvent.FinishReason)
}
