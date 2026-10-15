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
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/toolloading"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

type nativeSearchModel struct {
	*fakellm.FakeModel

	provider llm.ProviderID
}

func (m nativeSearchModel) Provider() llm.ProviderID { return m.provider }

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

// nativeSearchCases pairs the provider a model reports with the hosted search
// protocol it speaks. A Claude model on Vertex AI reports gcp.vertex but
// speaks, and tags its search results with, anthropic.
var nativeSearchCases = []struct {
	provider llm.ProviderID
	protocol llm.ToolSearchProtocol
}{
	{provider: "openai", protocol: "openai"},
	{provider: "anthropic", protocol: "anthropic"},
	{provider: "gcp.vertex", protocol: "anthropic"},
}

func TestNativeDiscoveryAndRestoration(t *testing.T) {
	t.Parallel()

	for _, tc := range nativeSearchCases {
		t.Run(string(tc.provider), func(t *testing.T) {
			t.Parallel()
			registry := newFixtureRegistry(t,
				fixtureTool{name: "fetch", group: "svc", deferred: true, groupInfo: llm.ToolGroup{Description: "Service records", Instructions: "Use the service carefully."}},
				fixtureTool{name: "other", group: "svc", deferred: true},
				fixtureTool{name: "solo", deferred: true},
				fixtureTool{name: "eager", group: "always"},
			)
			fake := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{Tools: true, Streaming: true, ToolSearchProtocol: tc.protocol}))
			fake.When(fakellm.FirstCall()).ThenRespondWith(respondWith(
				&llm.ToolSearchPart{Provider: llm.ProviderID(tc.protocol), Data: json.RawMessage(`{"type":"native_search_result"}`), Tools: []string{"fetch"}},
				llm.NewToolRequestPart("call_1", "fetch", json.RawMessage(`{}`)),
			))
			fake.When(fakellm.Any()).ThenRespondText("Done.")
			model := nativeSearchModel{FakeModel: fake, provider: tc.provider}
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
			require.True(t, toolloading.LoadedToolSet(sess)["fetch"])
			require.True(t, defByName(t, calls[1].Request.Tools, "fetch").Deferred, "history references preserve native deferral")

			// Persist and restore both native history and the loaded set.
			stored, err := json.Marshal(sess)
			require.NoError(t, err)
			var resumed session.State
			require.NoError(t, json.Unmarshal(stored, &resumed))
			replayed := ag.loader.Prepare(registry.List(), &resumed).Tools
			require.Equal(t, calls[1].Request.Tools, replayed)

			// Simulate compaction dropping native references while metadata survives.
			sess.Messages = userMessage("continue")
			plan := ag.loader.Prepare(registry.List(), sess)
			restored, section := plan.Tools, plan.Prompt
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

	for _, provider := range []llm.ProviderID{"google", "openaicompat", "bedrock", "openai"} {
		t.Run(string(provider), func(t *testing.T) {
			t.Parallel()
			// A protocol the agent does not speak, or none, keeps local discovery.
			protocol := llm.ToolSearchProtocol(provider)
			if provider == "openai" {
				protocol = ""
			}

			fake := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{ToolSearchProtocol: protocol, Tools: true, Streaming: true}))
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

	for _, tc := range nativeSearchCases {
		t.Run(string(tc.provider), func(t *testing.T) {
			t.Parallel()
			registry := newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true})
			fake := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{Tools: true, Streaming: true, ToolSearchProtocol: tc.protocol}))
			fake.When(fakellm.FirstCall()).ThenRespondWith(respondWith(
				toolCall("search_1", toolloading.SearchToolName, `{"query":"select:fetch"}`),
			))
			fake.When(fakellm.CallNumber(2)).ThenRespondWith(respondWith(toolCall("call_1", "fetch", `{}`)))
			fake.When(fakellm.Any()).ThenRespondText("Done.")
			model := nativeSearchModel{FakeModel: fake, provider: tc.provider}
			ag, err := New("local", "prompt", model, WithTools(registry), WithToolLoadingConfig(ToolLoadingConfig{ForceLocal: true}))
			require.NoError(t, err)

			sess := &session.State{ID: "s", Messages: userMessage("fetch it")}
			require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))

			calls := fake.CallsMatching(fakellm.Any())
			require.Len(t, calls, 3, "local discovery needs a client round trip before execution")

			for _, call := range calls {
				assert.False(t, call.Request.ToolSearch)
			}

			assert.Equal(t, []string{toolloading.SearchToolName}, defNames(calls[0].Request.Tools))
			assert.Equal(t, []string{"fetch", toolloading.SearchToolName}, defNames(calls[1].Request.Tools))
			require.Len(t, toolResultsFor(sess.Messages, toolloading.SearchToolName), 1)
			require.Len(t, toolResultsFor(sess.Messages, "fetch"), 1)
			assert.JSONEq(t, `{"ok":true}`, toolResultsFor(sess.Messages, "fetch")[0])
			assert.True(t, toolloading.LoadedToolSet(sess)["fetch"])
			assert.Empty(t, toolloading.NativeLoadedTools(sess.Messages, tc.protocol))
		})
	}
}

func TestNativeDiscoverySurvivesProactiveCompaction(t *testing.T) {
	t.Parallel()
	registry := newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true})
	fake := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{Tools: true, Streaming: true, ToolSearchProtocol: "anthropic"}), fakellm.WithContextWindow(20_000))
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
	ag.loader.Commit(sess, []string{"fetch"})
	before := ag.loader.Prepare(registry.List(), sess).Tools
	require.True(t, before[0].Deferred, "the search reference is still present before compaction")
	events := runAgent(t, ag, sess)
	require.Equal(t, agent.FinishReasonStop, finishReason(events))

	calls := fake.CallsMatching(fakellm.Any())
	require.Len(t, calls, 1)
	require.Empty(t, toolloading.NativeLoadedTools(calls[0].Request.Messages, "anthropic"), "compaction removed the search reference")
	require.False(t, calls[0].Request.Tools[0].Deferred, "restoration is re-evaluated after compaction and before the request")
}

// TestVertexClaudeUsesNativeToolSearch checks a Vertex Claude model, which
// speaks the Anthropic Messages protocol and whose offering names it as
// ToolSearchProtocol, gets hosted tool search like a direct Anthropic model.
func TestVertexClaudeUsesNativeToolSearch(t *testing.T) {
	t.Parallel()

	var body []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, claudeDoneStream)
	}))
	t.Cleanup(srv.Close)

	p, err := vertex.NewProvider(context.Background(),
		vertex.WithProject("my-project"), vertex.WithLocation("global"),
		vertex.WithBaseURL(srv.URL), vertex.WithHTTPClient(srv.Client()))
	require.NoError(t, err)

	model, err := p.NewModel(vertex.ModelClaudeSonnet46)
	require.NoError(t, err)
	require.Equal(t, llm.ToolSearchProtocolAnthropic, model.Capabilities().ToolSearchProtocol)

	ag, err := New("vertex", "prompt", model, WithTools(newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true})))
	require.NoError(t, err)
	runAgent(t, ag, &session.State{ID: "s", Messages: userMessage("hi")})

	require.Equal(t, "tool_search_tool_bm25_20251119", gjson.GetBytes(body, `tools.#(type%"tool_search*").type`).String(), "tools sent: %s", gjson.GetBytes(body, "tools.#.name"))
	require.True(t, gjson.GetBytes(body, `tools.#(name=="fetch").defer_loading`).Bool(), "tools sent: %s", gjson.GetBytes(body, "tools"))
}

func TestRetryWrappedVertexClaudeUsesNativeToolSearch(t *testing.T) {
	t.Parallel()

	var body []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, claudeDoneStream)
	}))
	t.Cleanup(srv.Close)

	p, err := vertex.NewProvider(context.Background(),
		vertex.WithProject("my-project"), vertex.WithLocation("global"),
		vertex.WithBaseURL(srv.URL), vertex.WithHTTPClient(srv.Client()))
	require.NoError(t, err)

	model, err := p.NewModel(vertex.ModelClaudeSonnet46)
	require.NoError(t, err)

	ag, err := New("vertex", "prompt", retry.WrapModel(model), WithTools(newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true})))
	require.NoError(t, err)
	runAgent(t, ag, &session.State{ID: "s", Messages: userMessage("hi")})

	require.Equal(t, "tool_search_tool_bm25_20251119", gjson.GetBytes(body, `tools.#(type%"tool_search*").type`).String(), "tools sent: %s", gjson.GetBytes(body, "tools.#.name"))
}

// TestVertexGeminiUsesLocalToolSearch checks a Gemini model on Vertex AI,
// whose offering names no ToolSearchProtocol, keeps local discovery: it
// sends the local tool_search function, not Anthropic's search tool or
// defer_loading.
func TestVertexGeminiUsesLocalToolSearch(t *testing.T) {
	t.Parallel()

	model, bodies := newFakeVertexModel(t, vertex.ModelGemini38Flash, geminiDoneStream)
	require.Empty(t, model.Capabilities().ToolSearchProtocol)

	ag, err := New("vertex", "prompt", model, WithTools(newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true})))
	require.NoError(t, err)
	runAgent(t, ag, &session.State{ID: "s", Messages: userMessage("hi")})

	require.Len(t, bodies(), 1)
	body := bodies()[0]
	assert.Equal(t, `["tool_search"]`, gjson.GetBytes(body, "tools.0.functionDeclarations.#.name").Raw, "tools sent: %s", gjson.GetBytes(body, "tools"))
	assert.NotContains(t, string(body), "tool_search_tool_bm25")
	assert.NotContains(t, string(body), "defer_loading")
}

// TestVertexClaudeToolSearchLoadsAcrossTurns checks a tool that Vertex AI's
// hosted search finds is loaded in the session: it is callable in the same
// response, callable on the next turn without another search, and still
// loaded once compaction drops the search result from the history.
func TestVertexClaudeToolSearchLoadsAcrossTurns(t *testing.T) {
	t.Parallel()

	model, bodies := newFakeVertexModel(t, vertex.ModelClaudeSonnet46,
		claudeSearchStream, claudeDoneStream, // turn 1: search, call fetch, answer
		claudeFetchStream, claudeDoneStream, // turn 2: call fetch, answer
		claudeDoneStream, // turn 3, after compaction
	)
	ag, err := New("vertex", "prompt", model, WithTools(newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true})))
	require.NoError(t, err)

	sess := &session.State{ID: "s", Messages: userMessage("fetch it")}
	require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))
	require.True(t, toolloading.LoadedToolSet(sess)["fetch"])
	require.Len(t, toolResultsFor(sess.Messages, "fetch"), 1)
	// A call to a tool the search did not load also loads it, but answers
	// tool_not_loaded, so only a real result proves the search loaded fetch.
	require.JSONEq(t, `{"ok":true}`, toolResultsFor(sess.Messages, "fetch")[0])

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("again")))
	require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))

	results := toolResultsFor(sess.Messages, "fetch")
	require.Len(t, results, 2, "fetch stays callable on the next turn without another search")
	require.JSONEq(t, `{"ok":true}`, results[1])

	// Simulate compaction dropping the search result: only the loaded set
	// remembers fetch now, so it goes out eager.
	sess.Messages = userMessage("continue")
	require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))

	sent := bodies()
	require.Len(t, sent, 5)

	turn2Request, turn3Request := sent[2], sent[4]
	assert.Contains(t, string(turn2Request), `"tool_search_tool_result"`, "the search result is replayed on the next turn")
	assert.True(t, gjson.GetBytes(turn2Request, `tools.#(name=="fetch").defer_loading`).Bool(), "the replayed search result keeps fetch deferred: %s", gjson.GetBytes(turn2Request, "tools"))
	require.True(t, gjson.GetBytes(turn3Request, `tools.#(name=="fetch")`).Exists(), "fetch is still sent: %s", gjson.GetBytes(turn3Request, "tools"))
	assert.False(t, gjson.GetBytes(turn3Request, `tools.#(name=="fetch").defer_loading`).Bool(), "a loaded fetch is eager once its search result is gone: %s", gjson.GetBytes(turn3Request, "tools"))
}

// newFakeVertexModel returns a Vertex AI model whose requests reach a test
// server that answers the nth request with the nth response, and a func that
// returns the request bodies it received.
func newFakeVertexModel(t *testing.T, name string, responses ...string) (llm.Model, func() [][]byte) {
	t.Helper()

	var (
		mu     sync.Mutex
		bodies [][]byte
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)

		mu.Lock()

		n := len(bodies)
		bodies = append(bodies, body)

		mu.Unlock()

		if n >= len(responses) {
			http.Error(w, "unexpected request", http.StatusBadRequest)

			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, responses[n])
	}))
	t.Cleanup(srv.Close)

	p, err := vertex.NewProvider(context.Background(),
		vertex.WithProject("my-project"), vertex.WithLocation("global"),
		vertex.WithBaseURL(srv.URL), vertex.WithHTTPClient(srv.Client()))
	require.NoError(t, err)

	model, err := p.NewModel(name)
	require.NoError(t, err)

	return model, func() [][]byte {
		mu.Lock()
		defer mu.Unlock()

		return slices.Clone(bodies)
	}
}

const geminiDoneStream = "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"Done."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}` + "\n\n"

// claudeSearchStream follows the block sequence Vertex AI streamed in a probe
// on 2026-10-07 - a bm25 search, its result, then a call to the found tool -
// with fetch as that tool.
const claudeSearchStream = `event: message_start
data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv_1","name":"tool_search_tool_bm25","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"fetch\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_search_tool_result","tool_use_id":"srv_1","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"fetch"}]}}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call_1","name":"fetch","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`

const claudeFetchStream = `event: message_start
data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_2","name":"fetch","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`

const claudeDoneStream = `event: message_start
data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Done."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`
