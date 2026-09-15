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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/agenttool"
)

// attrCapturingAgent records what agenttool passed down to its child.
type attrCapturingAgent struct {
	mockAgent

	gotAttributes    map[string]string
	gotCtxAttributes map[string]string
}

func (m *attrCapturingAgent) Run(
	ctx context.Context, inv *agent.InvocationMetadata,
) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		m.gotAttributes = inv.Attributes()
		m.gotCtxAttributes = agent.AttributesFromContext(ctx)

		msg := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(m.response))
		if !yield(agent.MessageEvent{Response: llm.Response{Message: msg}}, nil) {
			return
		}

		yield(agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}, nil)
	}
}

func TestExecute_SubAgentInheritsCallerAttributes(t *testing.T) {
	t.Parallel()

	// What a calling agent publishes before the tool call (see llmagent).
	ctx := agent.ContextWithAttributes(context.Background(), map[string]string{
		agent.AttrUserID: "alice@example.test",
		"user.tier":      "premium",
	})

	child := &attrCapturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{"query":"x"}`))
	require.NoError(t, err)

	assert.Equal(t, map[string]string{
		agent.AttrUserID: "alice@example.test",
		"user.tier":      "premium",
	}, child.gotAttributes)
}

func TestExecute_NoCallerAttributes_SubAgentAssertsNothing(t *testing.T) {
	t.Parallel()

	child := &attrCapturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(context.Background(), json.RawMessage(`{}`))
	require.NoError(t, err)

	assert.Empty(t, child.gotAttributes)
}

// attrNestingAgent is a mid-level agent that delegates to an inner agenttool,
// republishing its attributes the way llmagent does.
type attrNestingAgent struct {
	mockAgent

	inner tool.Tool
}

func (m *attrNestingAgent) Run(
	ctx context.Context, inv *agent.InvocationMetadata,
) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		ctx := agent.ContextWithAttributes(ctx, inv.Attributes())
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

func TestExecute_AttributesPropagateTransitively(t *testing.T) {
	t.Parallel()

	// parent -> mid -> inner: attribution must survive more than one hop.
	inner := &attrCapturingAgent{mockAgent: mockAgent{name: "inner", response: "deep"}}
	mid := &attrNestingAgent{
		mockAgent: mockAgent{name: "mid", response: "ok"},
		inner:     agenttool.New(inner),
	}

	ctx := agent.ContextWithAttributes(context.Background(), map[string]string{
		agent.AttrUserID: "alice@example.test",
	})

	_, err := agenttool.New(mid).Execute(ctx, json.RawMessage(`{}`))
	require.NoError(t, err)

	assert.Equal(t, "alice@example.test", inner.gotAttributes[agent.AttrUserID])
}

func TestExecute_SubAgentAttributesAreIsolatedFromParent(t *testing.T) {
	t.Parallel()

	// Inheritance is a copy: the sub-agent must not write back to the caller.
	parent := map[string]string{agent.AttrUserID: "alice@example.test"}
	ctx := agent.ContextWithAttributes(context.Background(), parent)

	child := &attrCapturingAgent{mockAgent: mockAgent{name: "search", response: "ok"}}
	at := agenttool.New(child)

	_, err := at.Execute(ctx, json.RawMessage(`{}`))
	require.NoError(t, err)

	child.gotAttributes["sub.only"] = "x"

	assert.Equal(t, map[string]string{agent.AttrUserID: "alice@example.test"},
		agent.AttributesFromContext(ctx))
	assert.Equal(t, map[string]string{agent.AttrUserID: "alice@example.test"}, parent)
}
