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
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

type nativeSearchModel struct {
	*fakellm.FakeModel

	provider string
}

func (m nativeSearchModel) Provider() string { return m.provider }

func defByName(t *testing.T, defs []llm.ToolDefinition, name string) llm.ToolDefinition {
	t.Helper()

	for _, def := range defs {
		if def.Name == name {
			return def
		}
	}

	t.Fatalf("tool %q not in %v", name, defNames(defs))

	return llm.ToolDefinition{}
}

func TestNativeDiscoveryAndRestoration(t *testing.T) {
	t.Parallel()

	for _, provider := range []string{"openai", "anthropic"} {
		t.Run(provider, func(t *testing.T) {
			t.Parallel()
			registry := newFixtureRegistry(t,
				fixtureTool{name: "fetch", group: "svc", deferred: true, groupInfo: llm.ToolGroup{Description: "Service records", Instructions: "Use the service carefully."}},
				fixtureTool{name: "other", group: "svc", deferred: true},
				fixtureTool{name: "solo", deferred: true},
				fixtureTool{name: "eager", group: "always"},
			)
			fake := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{Tools: true, Streaming: true, ToolSearch: true}))
			fake.When(fakellm.FirstCall()).ThenRespondWith(respondWith(
				&llm.ToolSearchPart{Provider: provider, Data: json.RawMessage(`{"type":"native_search_result"}`), Tools: []string{"fetch"}},
				llm.NewToolRequestPart("call_1", "fetch", json.RawMessage(`{}`)),
			))
			fake.When(fakellm.Any()).ThenRespondText("Done.")
			model := nativeSearchModel{FakeModel: fake, provider: provider}
			ag, err := New("native", "prompt", model, WithTools(registry))
			require.NoError(t, err)

			sess := &session.State{ID: "s", Messages: userMessage("fetch it")}
			events := runAgent(t, ag, sess)
			require.Equal(t, agent.FinishReasonStop, finishReason(events))

			calls := fake.CallsMatching(fakellm.Any())
			require.Len(t, calls, 2, "hosted discovery does not require another client round trip")
			require.True(t, calls[0].Request.ToolSearch)
			require.Equal(t, calls[0].Request.Tools, calls[1].Request.Tools, "stable catalog prefix")
			require.Equal(t, calls[0].Request.Messages[0], calls[1].Request.Messages[0], "stable group instructions")

			prompt := calls[0].Request.Messages[0].TextContent()
			assert.Contains(t, prompt, "## Searchable tools", "every native provider gets the group directory")
			assert.Contains(t, prompt, "- svc - Service records")
			assert.Contains(t, prompt, "- Other - 1 ungrouped tools")
			assert.NotContains(t, prompt, "- always", "groups without deferred tools are not searchable")
			assert.NotContains(t, prompt, "`fetch`", "the directory names groups, not tools")
			require.Len(t, toolResultsFor(sess.Messages, "fetch"), 1)
			require.JSONEq(t, `{"ok":true}`, toolResultsFor(sess.Messages, "fetch")[0])
			require.Empty(t, toolResultsFor(sess.Messages, "tool_search"))
			require.True(t, loadedToolSet(sess)["fetch"])
			require.True(t, defByName(t, calls[1].Request.Tools, "fetch").Deferred, "history references preserve native deferral")

			// Persist and restore both native history and the loaded set.
			stored, err := json.Marshal(sess)
			require.NoError(t, err)
			var resumed session.State
			require.NoError(t, json.Unmarshal(stored, &resumed))
			replayed, _ := ag.prepareTools(registry.List(), &resumed, true)
			require.Equal(t, calls[1].Request.Tools, replayed)

			// Simulate compaction dropping native references while metadata survives.
			sess.Messages = userMessage("continue")
			restored, section := ag.prepareTools(registry.List(), sess, true)
			require.False(t, defByName(t, restored, "fetch").Deferred, "a previously loaded tool is now eager")
			require.True(t, defByName(t, restored, "other").Deferred, "undiscovered tools remain deferred")
			assert.Contains(t, section, "Use the service carefully.")
			assert.Contains(t, section, "- svc - Service records", "the directory follows the registry, not the restored set")
			runAgent(t, ag, sess)
			require.False(t, defByName(t, fake.CallsMatching(fakellm.Any())[2].Request.Tools, "fetch").Deferred)
		})
	}
}

func TestNativeSearchCapabilityFallback(t *testing.T) {
	t.Parallel()

	for _, provider := range []string{"google", "openaicompat", "bedrock", "openai"} {
		t.Run(provider, func(t *testing.T) {
			t.Parallel()
			// Even permissive compatibility capabilities must not enable Responses APIs.
			fake := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{ToolSearch: provider != "openai", Tools: true, Streaming: true}))
			fake.When(fakellm.Any()).ThenRespondText("Done.")
			model := nativeSearchModel{FakeModel: fake, provider: provider}
			registry := newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true})
			ag, err := New("fallback", "prompt", model, WithTools(registry))
			require.NoError(t, err)
			runAgent(t, ag, &session.State{ID: "s", Messages: userMessage("hi")})

			req := fake.CallsMatching(fakellm.Any())[0].Request
			require.False(t, req.ToolSearch)
			require.Equal(t, []string{"tool_search"}, defNames(req.Tools))
		})
	}
}

func TestForcedLocalDiscovery(t *testing.T) {
	t.Parallel()

	for _, provider := range []string{"openai", "anthropic"} {
		t.Run(provider, func(t *testing.T) {
			t.Parallel()
			registry := newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true})
			fake := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{Tools: true, Streaming: true, ToolSearch: true}))
			fake.When(fakellm.FirstCall()).ThenRespondWith(respondWith(
				toolCall("search_1", toolSearchName, `{"query":"select:fetch"}`),
			))
			fake.When(fakellm.CallNumber(2)).ThenRespondWith(respondWith(toolCall("call_1", "fetch", `{}`)))
			fake.When(fakellm.Any()).ThenRespondText("Done.")
			model := nativeSearchModel{FakeModel: fake, provider: provider}
			ag, err := New("local", "prompt", model, WithTools(registry), WithToolLoadingConfig(ToolLoadingConfig{ForceLocal: true}))
			require.NoError(t, err)

			sess := &session.State{ID: "s", Messages: userMessage("fetch it")}
			require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))

			calls := fake.CallsMatching(fakellm.Any())
			require.Len(t, calls, 3, "local discovery needs a client round trip before execution")

			for _, call := range calls {
				assert.False(t, call.Request.ToolSearch)
			}

			assert.Equal(t, []string{toolSearchName}, defNames(calls[0].Request.Tools))
			assert.Equal(t, []string{"fetch", toolSearchName}, defNames(calls[1].Request.Tools))
			require.Len(t, toolResultsFor(sess.Messages, toolSearchName), 1)
			require.Len(t, toolResultsFor(sess.Messages, "fetch"), 1)
			assert.JSONEq(t, `{"ok":true}`, toolResultsFor(sess.Messages, "fetch")[0])
			assert.True(t, loadedToolSet(sess)["fetch"])
			assert.Empty(t, nativeLoadedTools(sess.Messages, provider))
		})
	}
}

func TestNativeDiscoverySurvivesProactiveCompaction(t *testing.T) {
	t.Parallel()
	registry := newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true})
	fake := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{Tools: true, Streaming: true, ToolSearch: true}), fakellm.WithContextWindow(20_000))
	fake.When(fakellm.Any()).ThenRespondText("Done.")
	model := nativeSearchModel{FakeModel: fake, provider: "anthropic"}
	ag, err := New("native", "prompt", model, WithTools(registry), WithCompaction(CompactionConfig{}))
	require.NoError(t, err)

	sess := &session.State{ID: "s", Messages: userMessage("old request")}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleAssistant, &llm.ToolSearchPart{Provider: "anthropic", Data: json.RawMessage(`{"type":"tool_search_tool_result"}`), Tools: []string{"fetch"}}, llm.NewTextPart(strings.Repeat("history ", 12_000))))
	for range 4 {
		sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")), llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("okay")))
	}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue now")))
	ag.loader.commit(sess, []string{"fetch"})
	before, _ := ag.prepareTools(registry.List(), sess, true)
	require.True(t, before[0].Deferred, "the search reference is still present before compaction")
	events := runAgent(t, ag, sess)
	require.Equal(t, agent.FinishReasonStop, finishReason(events))

	calls := fake.CallsMatching(fakellm.Any())
	require.Len(t, calls, 1)
	require.Empty(t, nativeLoadedTools(calls[0].Request.Messages, "anthropic"), "compaction removed the search reference")
	require.False(t, calls[0].Request.Tools[0].Deferred, "restoration is re-evaluated after compaction and before the request")
}
