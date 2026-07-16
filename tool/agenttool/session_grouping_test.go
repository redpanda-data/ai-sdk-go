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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool/agenttool"
)

// capturingAgent records the session it is invoked with, so tests can assert on
// the session id, messages, and metadata that agenttool builds for the child.
type capturingAgent struct {
	mockAgent

	gotSessionID string
	gotMessages  []llm.Message
	gotMetadata  map[string]any

	gotContextInvocationPresent bool
	gotContextInvocationSame    bool
	gotContextSessionID         string
}

func (m *capturingAgent) Run(ctx context.Context, inv *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		s := inv.Session()
		m.gotSessionID = s.ID
		m.gotMessages = s.Messages
		m.gotMetadata = s.Metadata

		ctxInv, ok := agent.InvocationFromContext(ctx)
		m.gotContextInvocationPresent = ok
		m.gotContextInvocationSame = ctxInv == inv

		if ok && ctxInv.Session() != nil {
			m.gotContextSessionID = ctxInv.Session().ID
		}

		msg := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(m.response))
		if !yield(agent.MessageEvent{Response: llm.Response{Message: msg}}, nil) {
			return
		}

		yield(agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}, nil)
	}
}

func parentContext(id string, messages ...llm.Message) context.Context {
	parentSess := &session.State{ID: id, Messages: messages}
	parentInv := agent.NewInvocationMetadata(parentSess, agent.Info{Name: "root"})

	return agent.ContextWithInvocation(context.Background(), parentInv)
}

func TestExecute_GroupsUnderParentConversation(t *testing.T) {
	t.Parallel()

	ctx := parentContext("parent-sess-123",
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("parent secret")))

	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{"query":"x"}`))
	require.NoError(t, err)

	// The sub-agent keeps its OWN unique storage id — it never reuses the
	// parent's, so it can never collide in a store.
	assert.True(t, strings.HasPrefix(child.gotSessionID, "agent-tool-search-"),
		"got %q", child.gotSessionID)
	assert.NotEqual(t, "parent-sess-123", child.gotSessionID)

	// Conversation grouping is carried in metadata: the parent's conversation id,
	// so observability groups the two under one conversation.
	assert.Equal(t, "parent-sess-123", child.gotMetadata[session.MetadataConversationID])

	// And ConversationID resolves the sub-agent's session to the parent's id.
	assert.Equal(t, "parent-sess-123",
		session.ConversationID(&session.State{ID: child.gotSessionID, Metadata: child.gotMetadata}))
}

func TestExecute_ContextCarriesChildInvocation(t *testing.T) {
	t.Parallel()

	ctx := parentContext("parent-sess-123")
	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{}`))
	require.NoError(t, err)

	assert.True(t, child.gotContextInvocationPresent)
	assert.True(t, child.gotContextInvocationSame)
	assert.Equal(t, child.gotSessionID, child.gotContextSessionID)
	assert.NotEqual(t, "parent-sess-123", child.gotContextSessionID)
}

func TestExecute_PropagatesRootConversationID(t *testing.T) {
	t.Parallel()

	// A parent that is itself a sub-agent already carries a conversation id in
	// metadata (the root). A nested sub-agent must group under that ROOT, not
	// under the immediate parent's unique storage id.
	parentSess := &session.State{
		ID:       "agent-tool-mid-999",
		Metadata: map[string]any{session.MetadataConversationID: "root-sess-1"},
	}
	parentInv := agent.NewInvocationMetadata(parentSess, agent.Info{Name: "mid"})
	ctx := agent.ContextWithInvocation(context.Background(), parentInv)

	child := &capturingAgent{mockAgent: mockAgent{name: "leaf", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{}`))
	require.NoError(t, err)

	assert.Equal(t, "root-sess-1", child.gotMetadata[session.MetadataConversationID])
	assert.NotEqual(t, "agent-tool-mid-999", child.gotMetadata[session.MetadataConversationID])
}

func TestExecute_ContextIsolatedDespiteGrouping(t *testing.T) {
	t.Parallel()

	ctx := parentContext("parent-sess-123",
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("parent secret")))

	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{"query":"x"}`))
	require.NoError(t, err)

	// The sub-agent must NOT observe the parent's history: exactly the args
	// message and nothing from the parent.
	require.Len(t, child.gotMessages, 1)
	assert.JSONEq(t, `{"query":"x"}`, child.gotMessages[0].TextContent())
}

func TestExecute_NoParentInvocation_MintsUniqueID(t *testing.T) {
	t.Parallel()

	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(context.Background(), json.RawMessage(`{}`))
	require.NoError(t, err)

	// Falls back to the previous behavior: a freshly minted, unique id.
	assert.True(t, strings.HasPrefix(child.gotSessionID, "agent-tool-search-"),
		"got %q", child.gotSessionID)
	// No linkage metadata when there is no parent.
	assert.Empty(t, child.gotMetadata)
}

func TestExecute_ParentWithNilSession_MintsUniqueID(t *testing.T) {
	t.Parallel()

	// Invocation present in ctx, but it has no session.
	parentInv := agent.NewInvocationMetadata(nil, agent.Info{Name: "root"})
	ctx := agent.ContextWithInvocation(context.Background(), parentInv)

	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{}`))
	require.NoError(t, err)

	assert.True(t, strings.HasPrefix(child.gotSessionID, "agent-tool-search-"),
		"got %q", child.gotSessionID)
	assert.Empty(t, child.gotMetadata)
}

func TestExecute_ParentWithEmptySessionID_NoMetadata(t *testing.T) {
	t.Parallel()

	// Parent session exists but resolves to an empty conversation id; no junk
	// empty-string entry may be written into the child's metadata.
	ctx := parentContext("")

	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{}`))
	require.NoError(t, err)

	assert.NotContains(t, child.gotMetadata, session.MetadataConversationID)
}
