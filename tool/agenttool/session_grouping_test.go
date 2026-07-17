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
	"iter"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/agenttool"
)

// capturingAgent records the session and context it is invoked with, so tests
// can assert on the session id, messages, and conversation id that agenttool
// builds for the child.
type capturingAgent struct {
	mockAgent

	gotSession           *session.State
	gotCtxConversationID string
}

func (m *capturingAgent) Run(ctx context.Context, inv *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		m.gotSession = inv.Session()
		m.gotCtxConversationID = agent.ConversationIDFromContext(ctx)

		msg := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(m.response))
		if !yield(agent.MessageEvent{Response: llm.Response{Message: msg}}, nil) {
			return
		}

		yield(agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}, nil)
	}
}

func TestExecute_GroupsUnderParentConversation(t *testing.T) {
	t.Parallel()

	// A calling agent sets the conversation grouping id on ctx before the tool
	// call (llmagent does this from session.ConversationID).
	ctx := agent.ContextWithConversationID(context.Background(), "parent-sess-123")

	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{"query":"x"}`))
	require.NoError(t, err)

	// The sub-agent keeps its OWN unique storage id — it never reuses the
	// parent's, so it can never collide in a store.
	assert.True(t, strings.HasPrefix(child.gotSession.ID, "agent-tool-search-"),
		"got %q", child.gotSession.ID)
	assert.NotEqual(t, "parent-sess-123", child.gotSession.ID)

	// Conversation grouping is carried as the session's ConversationID
	// override, so observability groups the two under one conversation.
	assert.Equal(t, "parent-sess-123", child.gotSession.ConversationID)
	assert.Equal(t, "parent-sess-123", session.ConversationID(child.gotSession))
}

func TestExecute_ContextCarriesGroupingForChildRun(t *testing.T) {
	t.Parallel()

	ctx := agent.ContextWithConversationID(context.Background(), "parent-sess-123")
	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{}`))
	require.NoError(t, err)

	// The child run's ctx carries the resolved grouping id so nested sub-agents
	// group under the same root even if the child agent sets nothing itself.
	assert.Equal(t, "parent-sess-123", child.gotCtxConversationID)
}

// nestingAgent simulates a mid-level agent that itself delegates to an inner
// agenttool, passing along the ctx it was invoked with — the transitive case.
type nestingAgent struct {
	mockAgent

	inner tool.Tool
}

func (m *nestingAgent) Run(ctx context.Context, _ *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		if _, err := m.inner.Execute(ctx, json.RawMessage(`{}`)); err != nil {
			yield(nil, err)
			return
		}

		msg := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(m.response))
		if !yield(agent.MessageEvent{Response: llm.Response{Message: msg}}, nil) {
			return
		}

		yield(agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}, nil)
	}
}

func TestExecute_PropagatesRootConversationIDTransitively(t *testing.T) {
	t.Parallel()

	// root conversation → mid agenttool → leaf agenttool: the leaf must group
	// under the ROOT conversation, not under the mid sub-agent's unique
	// storage session id.
	leaf := &capturingAgent{mockAgent: mockAgent{name: "leaf", response: "ok"}}
	mid := &nestingAgent{
		mockAgent: mockAgent{name: "mid", response: "ok"},
		inner:     agenttool.New(leaf),
	}

	ctx := agent.ContextWithConversationID(context.Background(), "root-sess-1")

	_, err := agenttool.New(mid).Execute(ctx, json.RawMessage(`{}`))
	require.NoError(t, err)

	assert.Equal(t, "root-sess-1", leaf.gotSession.ConversationID)
	assert.True(t, strings.HasPrefix(leaf.gotSession.ID, "agent-tool-leaf-"),
		"got %q", leaf.gotSession.ID)
}

func TestExecute_ContextIsolatedDespiteGrouping(t *testing.T) {
	t.Parallel()

	ctx := agent.ContextWithConversationID(context.Background(), "parent-sess-123")

	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{"query":"x"}`))
	require.NoError(t, err)

	// The sub-agent must NOT observe the parent's history: exactly the args
	// message and nothing from the parent.
	require.Len(t, child.gotSession.Messages, 1)
	assert.JSONEq(t, `{"query":"x"}`, child.gotSession.Messages[0].TextContent())
}

func TestExecute_NoParentConversation_MintsUniqueID(t *testing.T) {
	t.Parallel()

	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(context.Background(), json.RawMessage(`{}`))
	require.NoError(t, err)

	// A freshly minted, unique id; no conversation override, so the session is
	// its own conversation and grouping falls back to the session id.
	assert.True(t, strings.HasPrefix(child.gotSession.ID, "agent-tool-search-"),
		"got %q", child.gotSession.ID)
	assert.Empty(t, child.gotSession.ConversationID)
	assert.Equal(t, child.gotSession.ID, session.ConversationID(child.gotSession))

	// The child's metadata map stays non-nil so child agents and interceptors
	// can write to it.
	assert.NotNil(t, child.gotSession.Metadata)
}

// idRecordingAgent reports the session id of every invocation on a channel,
// safe for concurrent Execute calls on the same AgentTool.
type idRecordingAgent struct {
	mockAgent

	ids chan string
}

func (m *idRecordingAgent) Run(_ context.Context, inv *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		m.ids <- inv.Session().ID

		msg := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(m.response))
		if !yield(agent.MessageEvent{Response: llm.Response{Message: msg}}, nil) {
			return
		}

		yield(agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}, nil)
	}
}

func TestExecute_ConcurrentCalls_UniqueSessionIDs(t *testing.T) {
	t.Parallel()

	const calls = 16

	child := &idRecordingAgent{
		mockAgent: mockAgent{name: "search", response: "ok"},
		ids:       make(chan string, calls),
	}
	at := agenttool.New(child)

	var wg sync.WaitGroup
	for range calls {
		wg.Go(func() {
			_, err := at.Execute(context.Background(), json.RawMessage(`{}`))
			assert.NoError(t, err)
		})
	}

	wg.Wait()
	close(child.ids)

	seen := make(map[string]bool, calls)
	for id := range child.ids {
		assert.False(t, seen[id], "duplicate session id %q", id)
		seen[id] = true
	}

	assert.Len(t, seen, calls)
}
