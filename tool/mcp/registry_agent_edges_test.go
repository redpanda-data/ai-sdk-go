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

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const (
	registryAgentRoutedTool = "remote__routed"
	registryAgentSharedTool = "remote__shared"
	registryAgentAddedTool  = "remote__added"
	registryAgentSearchTool = "tool_search"
	registryAgentCallMethod = "tools/call"
)

type registryAgentWireCall struct {
	name      string
	headers   http.Header
	arguments json.RawMessage
}

func registryAgentCaptureCalls(peer *registryPeer) <-chan registryAgentWireCall {
	calls := make(chan registryAgentWireCall, 16)

	peer.servers["alice"].AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
		return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
			if method == registryAgentCallMethod {
				call, ok := req.(*sdkmcp.CallToolRequest)
				if !ok {
					return nil, errors.New("unexpected MCP call request type")
				}

				calls <- registryAgentWireCall{name: call.Params.Name, headers: call.Extra.Header.Clone(), arguments: append(json.RawMessage(nil), call.Params.Arguments...)}
			}

			return next(ctx, method, req)
		}
	})

	return calls
}

func registryAgentReadCalls(t *testing.T, calls <-chan registryAgentWireCall, count int) []registryAgentWireCall {
	t.Helper()

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	result := make([]registryAgentWireCall, 0, count)
	for range count {
		select {
		case call := <-calls:
			result = append(result, call)
		case <-ctx.Done():
			t.Fatal("expected MCP calls did not reach the real server")
		}
	}

	assert.Empty(t, calls, "there must not be an extra remote call after the invocation ends")

	return result
}

func registryAgentRoutedDefinition(t *testing.T, req *llm.Request) llm.ToolDefinition {
	t.Helper()

	for _, definition := range req.Tools {
		if definition.Name == registryAgentRoutedTool {
			return definition
		}
	}

	t.Fatalf("tool %q absent from model request %v", registryAgentRoutedTool, registryNames(req.Tools))

	return llm.ToolDefinition{}
}

func registryAgentRespond(parts ...llm.Part) (*llm.Response, error) {
	return &llm.Response{Message: llm.NewMessage(llm.RoleAssistant, parts...), FinishReason: llm.FinishReasonToolCalls}, nil
}

func registryAgentRun(ctx context.Context, t *testing.T, ag *llmagent.LLMAgent, state *session.State) []agent.Event {
	t.Helper()
	var events []agent.Event

	for event, err := range ag.Run(ctx, agent.NewInvocationMetadata(state, ag.Info())) {
		require.NoError(t, err)

		events = append(events, event)
	}
	var reason agent.FinishReason

	for _, event := range events {
		if end, ok := event.(agent.InvocationEndEvent); ok {
			reason = end.FinishReason
		}
	}

	assert.Equal(t, agent.FinishReasonStop, reason)

	return events
}

func registryAgentResults(state *session.State, name string) []*llm.ToolResponsePart {
	var results []*llm.ToolResponsePart

	for _, message := range state.Messages {
		for _, result := range message.ToolResponses() {
			if result.Name == name {
				results = append(results, result)
			}
		}
	}

	return results
}

func registryAgentAddRouted(peer *registryPeer, description, property string) {
	peer.servers["alice"].AddTool(&sdkmcp.Tool{Name: "routed", Description: description, InputSchema: map[string]any{"type": "object", "properties": map[string]any{property: map[string]any{"type": "string"}}}},
		func(_ context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
			return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: req.Extra.Header.Get("Authorization")}}}, nil
		})
}

func registryAgentCredentials(token *atomic.Value) CredentialsProvider {
	return func(ctx context.Context) (Credentials, error) {
		credentials, err := registryContextCredentials(ctx)
		if err != nil {
			return Credentials{}, err
		}

		bearer, ok := token.Load().(string)
		if !ok {
			return Credentials{}, errors.New("missing test bearer")
		}

		credentials.Headers = http.Header{"Authorization": {"Bearer " + bearer}}

		return credentials, nil
	}
}

func TestRegistryAgentNotificationDuringModelStep(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()
			peer := newRegistryPeer(t, legacy)
			registryAgentAddRouted(peer, "Original route.", "old")
			sent := registryAdvertiseToolTTL(peer, time.Hour, false)
			wire := registryAgentCaptureCalls(peer)
			var token atomic.Value
			token.Store("alice:one")
			registry := newContextRegistry(t, peer, WithCredentials(registryAgentCredentials(&token)), WithRegistryMaxToolCacheTTL(time.Hour))
			model := fakellm.NewFakeModel()
			model.When(fakellm.FirstCall()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				assert.Len(t, peer.requests(registryParameterListMethod), 1)
				waitForChange := registryToolChangeBarrier(t, registry, "alice", sent)

				registryAgentAddRouted(peer, "Updated route.", "new")
				peer.servers["alice"].RemoveTools("shared")
				peer.addTool("alice", "added", "New capability.", "new")
				waitForChange()
				token.Store("alice:two")

				assert.Len(t, peer.requests(registryParameterListMethod), 1, "notification must invalidate without discovering during the model call")
				assert.Contains(t, registryNames(req.Tools), registryAgentSharedTool)
				assert.NotContains(t, registryNames(req.Tools), registryAgentAddedTool)
				assert.Equal(t, "Original route.", registryAgentRoutedDefinition(t, req).Description)
				assert.Contains(t, string(registryAgentRoutedDefinition(t, req).Parameters), "old")

				return registryAgentRespond(
					llm.NewToolRequestPart("old-route", registryAgentRoutedTool, json.RawMessage(`{"old":"first-route"}`)),
					llm.NewToolRequestPart("old-shared", registryAgentSharedTool, json.RawMessage(`{}`)),
				)
			})
			model.When(fakellm.CallNumber(2)).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				assert.Len(t, peer.requests(registryParameterListMethod), 2)
				assert.NotContains(t, registryNames(req.Tools), registryAgentSharedTool)
				assert.Contains(t, registryNames(req.Tools), registryAgentAddedTool)
				assert.Equal(t, "Updated route.", registryAgentRoutedDefinition(t, req).Description)
				assert.Contains(t, string(registryAgentRoutedDefinition(t, req).Parameters), "new")

				return registryAgentRespond(
					llm.NewToolRequestPart("new-route", registryAgentRoutedTool, json.RawMessage(`{"new":"second-route"}`)),
					llm.NewToolRequestPart("stale-shared", registryAgentSharedTool, json.RawMessage(`{}`)),
					llm.NewToolRequestPart("new-tool", registryAgentAddedTool, json.RawMessage(`{"new":"value"}`)),
				)
			})
			model.When(fakellm.Any()).ThenRespondText("done")
			ag, err := llmagent.New("notification-agent", "Use the caller's remote tools.", model, llmagent.WithTools(registry))
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 15*time.Second)
			defer cancel()

			state := &session.State{ID: "notification-agent", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("route twice"))}}
			registryAgentRun(ctx, t, ag, state)

			calls := registryAgentReadCalls(t, wire, 4)

			names := make([]string, 0, len(calls))
			routed := make([]json.RawMessage, 0, 2)

			for _, call := range calls {
				names = append(names, call.name)
				assert.Equal(t, "Bearer alice:two", call.headers.Get("Authorization"), "execution must re-resolve current credentials without changing the partition")

				if call.name == "routed" {
					routed = append(routed, call.arguments)
				}
			}

			require.Len(t, routed, 2)
			assert.JSONEq(t, `{"old":"first-route"}`, string(routed[0]))
			assert.JSONEq(t, `{"new":"second-route"}`, string(routed[1]))
			assert.ElementsMatch(t, []string{"routed", "shared", "routed", "added"}, names)
			assert.Len(t, peer.requests(registryAgentCallMethod), 4, "removed tool is dispatched once from the old step but never from the refreshed step")
			assert.Len(t, peer.requests(registryParameterListMethod), 2, "current-step dispatch must not refetch the invalidated catalog")

			results := registryAgentResults(state, registryAgentSharedTool)
			require.Len(t, results, 2)
			assert.True(t, results[0].IsError, "the server has removed the tool, although the step still owns its route")
			assert.True(t, results[1].IsError)

			for _, name := range []string{registryAgentRoutedTool, registryAgentAddedTool} {
				for _, result := range registryAgentResults(state, name) {
					assert.False(t, result.IsError, string(result.Result))
				}
			}

			if legacy {
				assert.Len(t, peer.requests("initialize"), 1)
			} else {
				assert.Len(t, peer.requests("server/discover"), 1)
			}
		})
	}
}

func TestRegistryAgentRestoredDeferredNamesCannotResurrectRemovedTool(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()
			peer := newRegistryPeer(t, legacy)
			sent := registryAdvertiseToolTTL(peer, time.Hour, false)
			wire := registryAgentCaptureCalls(peer)
			registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(time.Hour), WithRegistryToolOptions(tool.WithDeferred()))
			model := fakellm.NewFakeModel()
			model.When(fakellm.FirstCall()).ThenRespondWithToolCall(registryAgentSearchTool, map[string]any{"query": "select:" + registryAgentSharedTool})
			model.When(fakellm.CallNumber(2)).ThenRespondText("loaded; resume later")
			model.When(fakellm.CallNumber(3)).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				assert.Contains(t, registryNames(req.Tools), registryAgentSharedTool, "session restoration must really restore the previously loaded name")
				assert.Len(t, peer.requests(registryParameterListMethod), 1)
				waitForChange := registryToolChangeBarrier(t, registry, "alice", sent)

				peer.servers["alice"].RemoveTools("shared")
				peer.addTool("alice", "alice_only", "Refreshed caller schema.", "new")
				peer.addTool("alice", "added", "New searchable capability.", "new")
				waitForChange()

				return registryAgentRespond(llm.NewToolRequestPart("old-step", registryAgentSharedTool, json.RawMessage(`{}`)))
			})
			model.When(fakellm.CallNumber(4)).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				assert.Equal(t, []string{registryAgentSearchTool}, registryNames(req.Tools))
				assert.NotContains(t, req.Messages[0].TextContent(), "`"+registryAgentSharedTool+"`")
				assert.Contains(t, req.Messages[0].TextContent(), "Refreshed caller schema")
				assert.Contains(t, req.Messages[0].TextContent(), "New searchable capability")
				assert.Len(t, peer.requests(registryParameterListMethod), 2)

				return registryAgentRespond(
					llm.NewToolRequestPart("search-removed", registryAgentSearchTool, json.RawMessage(`{"query":"select:remote__shared"}`)),
					llm.NewToolRequestPart("stale-loaded", registryAgentSharedTool, json.RawMessage(`{}`)),
				)
			})
			model.When(fakellm.CallNumber(5)).ThenRespondWithToolCall(registryAgentSearchTool, map[string]any{"query": "select:" + registryAgentAddedTool})
			model.When(fakellm.CallNumber(6)).ThenRespondWithToolCall(registryAgentAddedTool, map[string]any{"new": "value"})
			model.When(fakellm.Any()).ThenRespondText("done")
			ag, err := llmagent.New("restored-search", "Use the searchable caller tools.", model, llmagent.WithTools(registry), llmagent.WithToolLoadingConfig(llmagent.ToolLoadingConfig{ForceLocal: true}))
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 15*time.Second)
			defer cancel()

			state := &session.State{ID: "restored-search", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("load shared"))}}
			registryAgentRun(ctx, t, ag, state)
			stored, err := json.Marshal(state)
			require.NoError(t, err)
			var restored session.State
			require.NoError(t, json.Unmarshal(stored, &restored))
			assert.Contains(t, restored.Metadata["ai-sdk-go/loaded_tools"], registryAgentSharedTool)
			restored.Messages = append(restored.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")))
			registryAgentRun(ctx, t, ag, &restored)

			calls := registryAgentReadCalls(t, wire, 2)
			assert.Equal(t, "shared", calls[0].name)
			assert.Equal(t, "added", calls[1].name)
			assert.Len(t, peer.requests(registryParameterListMethod), 2)
			assert.Len(t, peer.requests(registryAgentCallMethod), 2)

			results := registryAgentResults(&restored, registryAgentSharedTool)
			require.Len(t, results, 2)
			assert.True(t, results[0].IsError)
			assert.True(t, results[1].IsError)

			searches := registryAgentResults(&restored, registryAgentSearchTool)
			require.Len(t, searches, 3)
			assert.Contains(t, string(searches[1].Result), `"not_found":["remote__shared"]`)

			added := registryAgentResults(&restored, registryAgentAddedTool)
			require.Len(t, added, 1)
			assert.False(t, added[0].IsError, string(added[0].Result))
			assert.Contains(t, restored.Metadata["ai-sdk-go/loaded_tools"], registryAgentSharedTool, "retained load metadata must not grant availability")

			requests := model.CallsMatching(fakellm.Any())
			require.Len(t, requests, 7)
			assert.Equal(t, []string{registryAgentAddedTool, registryAgentSearchTool}, registryNames(requests[5].Request.Tools))
			assert.NotContains(t, requests[5].Request.Messages[0].TextContent(), "`"+registryAgentSharedTool+"`")
		})
	}
}

func TestRegistryAgentReactiveRetryKeepsNotifiedSnapshot(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()
			peer := newRegistryPeer(t, legacy)
			registryAgentAddRouted(peer, "Retry original route.", "old")
			sent := registryAdvertiseToolTTL(peer, time.Hour, false)
			wire := registryAgentCaptureCalls(peer)
			registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(time.Hour))
			model := fakellm.NewFakeModel(fakellm.WithContextWindow(200_000))
			model.When(fakellm.FirstCall()).ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				waitForChange := registryToolChangeBarrier(t, registry, "alice", sent)

				registryAgentAddRouted(peer, "Retry refreshed route.", "new")
				peer.servers["alice"].RemoveTools("shared")
				peer.addTool("alice", "added", "Available after retry.", "new")
				waitForChange()
				assert.Len(t, peer.requests(registryParameterListMethod), 1)

				return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, &llm.ProviderError{Base: llm.ErrContextOverflow, Code: "invalid_request_error", Message: "prompt is too long"})
			})
			model.When(fakellm.CallNumber(2)).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				assert.Len(t, peer.requests(registryParameterListMethod), 1, "reactive retry must not discover an already-invalidated catalog")
				assert.Contains(t, registryNames(req.Tools), registryAgentSharedTool)
				assert.NotContains(t, registryNames(req.Tools), registryAgentAddedTool)
				assert.Equal(t, "Retry original route.", registryAgentRoutedDefinition(t, req).Description)
				assert.Contains(t, string(registryAgentRoutedDefinition(t, req).Parameters), "old")

				return registryAgentRespond(llm.NewToolRequestPart("retry-route", registryAgentRoutedTool, json.RawMessage(`{"old":"retry-route"}`)))
			})
			model.When(fakellm.CallNumber(3)).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				assert.Len(t, peer.requests(registryParameterListMethod), 2)
				assert.NotContains(t, registryNames(req.Tools), registryAgentSharedTool)
				assert.Contains(t, registryNames(req.Tools), registryAgentAddedTool)
				assert.Equal(t, "Retry refreshed route.", registryAgentRoutedDefinition(t, req).Description)

				return registryAgentRespond(llm.NewToolRequestPart("next-route", registryAgentRoutedTool, json.RawMessage(`{"new":"next-route"}`)))
			})
			model.When(fakellm.Any()).ThenRespondText("done")
			ag, err := llmagent.New("reactive-notification", "Use remote tools.", model, llmagent.WithTools(registry), llmagent.WithCompaction(llmagent.CompactionConfig{}))
			require.NoError(t, err)

			state := &session.State{ID: "reactive-notification", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("start"))}}

			for i := range 4 {
				id := fmt.Sprintf("old-%d", i)
				state.Messages = append(state.Messages,
					llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart(id, registryAgentSharedTool, json.RawMessage(`{}`))),
					llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(id, registryAgentSharedTool, fmt.Appendf(nil, `{"data":%q}`, strings.Repeat("x", 16_000)), false)),
				)
			}

			state.Messages = append(state.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")))

			ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 15*time.Second)
			defer cancel()

			events := registryAgentRun(ctx, t, ag, state)

			requests := model.CallsMatching(fakellm.Any())
			require.Len(t, requests, 4)
			assert.Equal(t, requests[0].Request.Tools, requests[1].Request.Tools)
			assert.Equal(t, requests[0].Request.Messages[0], requests[1].Request.Messages[0])
			assert.Less(t, requests[1].InputTokens, requests[0].InputTokens)
			assert.NotEqual(t, requests[1].Request.Tools, requests[2].Request.Tools)
			assert.Len(t, peer.requests(registryParameterListMethod), 2)
			calls := registryAgentReadCalls(t, wire, 2)
			assert.Equal(t, "routed", calls[0].name)
			assert.JSONEq(t, `{"old":"retry-route"}`, string(calls[0].arguments))
			assert.Equal(t, "routed", calls[1].name)
			assert.JSONEq(t, `{"new":"next-route"}`, string(calls[1].arguments))

			for _, call := range calls {
				assert.Equal(t, "Bearer alice:one", call.headers.Get("Authorization"))
			}
			var reactive bool

			for _, event := range events {
				if compacted, ok := event.(agent.CompactionEvent); ok && compacted.Report.Phase == agent.CompactionPhaseReactive {
					reactive = true
				}
			}

			assert.True(t, reactive, "the provider overflow must really trigger the reactive retry")

			results := registryAgentResults(state, registryAgentRoutedTool)
			require.Len(t, results, 2)

			for _, result := range results {
				assert.False(t, result.IsError, string(result.Result))
			}
		})
	}
}
