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
}

func (m *capturingAgent) Run(_ context.Context, inv *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		s := inv.Session()
		m.gotSessionID = s.ID
		m.gotMessages = s.Messages
		m.gotMetadata = s.Metadata

		msg := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(m.response))
		if !yield(agent.MessageEvent{Response: llm.Response{Message: msg}}, nil) {
			return
		}

		yield(agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}, nil)
	}
}

func parentContext(id string, messages ...llm.Message) (context.Context, *agent.InvocationMetadata) {
	parentSess := &session.State{ID: id, Messages: messages}
	parentInv := agent.NewInvocationMetadata(parentSess, agent.Info{Name: "root"})

	return agent.ContextWithInvocation(context.Background(), parentInv), parentInv
}

func TestExecute_SharesParentSessionID(t *testing.T) {
	t.Parallel()

	ctx, parentInv := parentContext("parent-sess-123",
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("parent secret")))

	child := &capturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{"query":"x"}`))
	require.NoError(t, err)

	// Shares the parent's session id for conversation grouping.
	assert.Equal(t, "parent-sess-123", child.gotSessionID)

	// Linkage metadata records the discriminator + back-reference.
	assert.Equal(t, true, child.gotMetadata[session.MetadataIsSidechain])
	assert.Equal(t, parentInv.InvocationID(), child.gotMetadata[session.MetadataParentInvocationID])
	assert.Equal(t, "search", child.gotMetadata[session.MetadataAgentPath])
}

func TestExecute_ContextIsolatedDespiteSharedID(t *testing.T) {
	t.Parallel()

	ctx, _ := parentContext("parent-sess-123",
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
