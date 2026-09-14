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

package llmagent

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/toolloading"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

//
// Fixtures shared by the tool-loading tests that drive a whole agent.
//

// stubTool is a registry tool with a canned result that counts its runs and
// keeps the arguments it received.
type stubTool struct {
	def    llm.ToolDefinition
	result json.RawMessage
	calls  atomic.Int32

	mu       sync.Mutex
	received []json.RawMessage
}

func (t *stubTool) Definition() llm.ToolDefinition { return t.def }

func (t *stubTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	t.calls.Add(1)

	t.mu.Lock()
	t.received = append(t.received, append(json.RawMessage(nil), args...))
	t.mu.Unlock()

	if t.result != nil {
		return t.result, nil
	}

	return json.RawMessage(`{"ok":true}`), nil
}

func (t *stubTool) arguments() []json.RawMessage {
	t.mu.Lock()
	defer t.mu.Unlock()

	return append([]json.RawMessage(nil), t.received...)
}

type fixtureTool struct {
	name        string
	group       string
	description string
	params      string
	deferred    bool
	groupInfo   llm.ToolGroup // optional description/instructions for the group
}

func newFixtureRegistry(tb testing.TB, tools ...fixtureTool) tool.Registry {
	tb.Helper()

	registry := tool.NewRegistry(tool.RegistryConfig{})

	for _, spec := range tools {
		params := spec.params
		if params == "" {
			params = `{"type":"object"}`
		}

		var opts []tool.Option
		if spec.deferred {
			opts = append(opts, tool.WithDeferred())
		}

		if spec.group != "" {
			group := spec.groupInfo
			group.Name = spec.group
			opts = append(opts, tool.WithGroup(group))
		}

		require.NoError(tb, registry.Register(&stubTool{def: llm.ToolDefinition{
			Name: spec.name, Description: spec.description, Parameters: json.RawMessage(params),
		}}, opts...))
	}

	return registry
}

func defNames(defs []llm.ToolDefinition) []string {
	names := make([]string, len(defs))
	for i, def := range defs {
		names[i] = def.Name
	}

	return names
}

// runAgent drives one invocation to completion and returns its events.
func runAgent(t *testing.T, ag *LLMAgent, sess *session.State) []agent.Event {
	t.Helper()

	events := make([]agent.Event, 0, 32)

	for evt, err := range ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{})) {
		require.NoError(t, err)

		events = append(events, evt)
	}

	return events
}

func finishReason(events []agent.Event) agent.FinishReason {
	for i := len(events) - 1; i >= 0; i-- {
		if end, ok := events[i].(agent.InvocationEndEvent); ok {
			return end.FinishReason
		}
	}

	return ""
}

func requestToolNames(req *llm.Request) []string {
	return defNames(req.Tools)
}

// toolResultsFor returns every result recorded for name, in order.
func toolResultsFor(msgs []llm.Message, name string) []string {
	var results []string

	for _, msg := range msgs {
		for _, resp := range msg.ToolResponses() {
			if resp.Name == name {
				results = append(results, string(resp.Result))
			}
		}
	}

	return results
}

func userMessage(text string) []llm.Message {
	return []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart(text))}
}

// TestToolSearchNameIsReserved: a registry tool of the same name would be
// declared twice and unreachable, so construction fails with a clear message.
func TestToolSearchNameIsReserved(t *testing.T) {
	t.Parallel()

	registry := newFixtureRegistry(t,
		fixtureTool{name: toolloading.SearchToolName, description: "an operator's own tool of the same name"},
		fixtureTool{name: "svc__thing", group: "svc", deferred: true, description: "A thing."},
	)

	_, err := New("agent", "prompt", fakellm.NewFakeModel(), WithTools(registry))
	require.ErrorContains(t, err, toolloading.SearchToolName)

	_, err = New("agent", "prompt", fakellm.NewFakeModel(), WithToolLoadingConfig(ToolLoadingConfig{MaxLoadTokens: -1}))
	require.ErrorContains(t, err, "MaxLoadTokens")

	// prepare never double-declares even if a registry gains the name later.
	defs := toolloading.New(registry, fakellm.NewFakeModel(), toolloading.Config{}).Prepare(registry.List(), &session.State{ID: "s"}).Tools
	assert.Equal(t, 1, strings.Count(strings.Join(defNames(defs), ","), toolloading.SearchToolName))
}
