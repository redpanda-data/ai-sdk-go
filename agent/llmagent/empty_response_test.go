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

// TestRun_EmptyContentNotPersisted locks in the agent-loop guard against
// session poisoning by empty assistant messages.
//
// Sibling case to PR #116. Where #116 surfaced *truncated* tool args
// instead of corrupting state, this guards against *empty* content.
// When a provider returns a Response whose Message has zero content
// blocks (max_tokens before any block was emitted, the only block was
// a partial tool_use that stream finalisation correctly dropped, or
// refusal with no text emitted), Anthropic and most other providers
// reject any subsequent replay with `messages.N.content: Field
// required` — once the empty turn is in session state, every
// following call 400s and the conversation is wedged forever.
//
// The guard sits at the same place adk-go's AppendEvent honours its
// `Event.Partial` flag: the session-store boundary, the single
// chokepoint for "what gets persisted." The MessageEvent still fires
// (observers see what happened) and the FinishReason still propagates
// (terminal-reason handling fires below); only the persistence step
// is skipped.
func TestRun_EmptyContentNotPersisted(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			// max_tokens hit before any content block emitted: provider
			// returns a Response with empty content + FinishReasonLength.
			return &llm.Response{
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: nil,
				},
				FinishReason: llm.FinishReasonLength,
			}, nil
		})

	ag, err := llmagent.New("test-agent", "You are a helpful assistant", model)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Info{})

	events := collectEvents(t, ag.Run(t.Context(), inv))

	// FinishReason still propagates so callers can react.
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonLength, endEvent.FinishReason,
		"finish reason must propagate even when content is empty")

	// MessageEvent still fires so observers can see what happened.
	messageEvents := filterEvents[agent.MessageEvent](events)
	assert.NotEmpty(t, messageEvents, "MessageEvent must still fire so observers see the empty response")

	// Session must NOT contain the empty assistant message — that's the
	// whole point. Only the original user message stays.
	require.Len(t, sess.Messages, 1, "empty assistant message must not be persisted to session")
	assert.Equal(t, llm.RoleUser, sess.Messages[0].Role)

	// Sanity: every persisted message has non-empty content. This is
	// the invariant Anthropic et al. require — every message in a
	// replayed array must have at least one content block.
	for i, m := range sess.Messages {
		assert.NotEmpty(t, m.Content, "session.Messages[%d] must have non-empty content", i)
	}
}

// TestRun_NonEmptyContentStillPersisted is the negative control:
// the guard must not over-fire and skip messages that DO have content.
// If this regresses, normal turns stop being persisted and the agent
// loses all state.
func TestRun_NonEmptyContentStillPersisted(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			return &llm.Response{
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: []*llm.Part{llm.NewTextPart("hi back")},
				},
				FinishReason: llm.FinishReasonStop,
			}, nil
		})

	ag, err := llmagent.New("test-agent", "You are a helpful assistant", model)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Info{})

	collectEvents(t, ag.Run(t.Context(), inv))

	require.Len(t, sess.Messages, 2, "non-empty assistant message must be persisted")
	assert.Equal(t, llm.RoleAssistant, sess.Messages[1].Role)
	assert.NotEmpty(t, sess.Messages[1].Content)
}
