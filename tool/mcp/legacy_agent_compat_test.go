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
	"log/slog"
	"slices"
	"sync"
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
	legacyCompatRouted = "old__routed"
	legacyCompatShared = "old__shared"
	legacyCompatAdded  = "old__added"
	legacyCompatOnly   = "old__alice_only"
	legacyCompatStatic = "local_lookup"
	legacyCompatAuth   = "Bearer alice:one"

	legacyCompatSharedRaw = "shared"
)

type legacyCompatStaticTool struct{ calls atomic.Int32 }

func (*legacyCompatStaticTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{Name: legacyCompatStatic, Description: "Local lookup.", Parameters: json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}}}`)}
}

func (s *legacyCompatStaticTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	s.calls.Add(1)

	return json.RawMessage(`{"source":"static"}`), nil
}

// legacyCompatSyncLog wakes a waiter after every completed SyncTools, including
// one driven by a tools/list_changed notification.
type legacyCompatSyncLog struct{ synced chan struct{} }

func (*legacyCompatSyncLog) Enabled(context.Context, slog.Level) bool { return true }

func (h *legacyCompatSyncLog) Handle(_ context.Context, record slog.Record) error {
	if record.Message == "tool sync completed" {
		select {
		case h.synced <- struct{}{}:
		default:
		}
	}

	return nil
}

func (h *legacyCompatSyncLog) WithAttrs([]slog.Attr) slog.Handler { return h }

func (h *legacyCompatSyncLog) WithGroup(string) slog.Handler { return h }

type legacyCompatFixture struct {
	client   Client
	registry tool.Registry
	static   *legacyCompatStaticTool
	synced   chan struct{}
	wire     <-chan registryAgentWireCall
}

// legacyCompatStart connects the pre-registry consumer path: a legacy Client
// populating a native registry with static headers. Peer tools and middleware
// must be installed before it is called.
func legacyCompatStart(t *testing.T, peer *registryPeer, opts ...ClientOption) *legacyCompatFixture {
	t.Helper()

	fx := &legacyCompatFixture{
		registry: tool.NewRegistry(tool.RegistryConfig{}),
		static:   &legacyCompatStaticTool{},
		synced:   make(chan struct{}, 1),
		wire:     registryAgentCaptureCalls(peer),
	}
	require.NoError(t, fx.registry.Register(fx.static))

	client, err := NewClient("old", NewStreamableTransport(peer.url, WithHTTPHeaders(map[string]string{"Authorization": legacyCompatAuth})),
		append([]ClientOption{WithRegistry(fx.registry), WithLogger(slog.New(&legacyCompatSyncLog{synced: fx.synced}))}, opts...)...)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, client.Close()) })
	require.NoError(t, client.Start(t.Context()))
	fx.client = client

	return fx
}

func (fx *legacyCompatFixture) awaitSync(ctx context.Context, done func() (bool, error)) error {
	watchdog := time.NewTimer(10 * time.Second)
	defer watchdog.Stop()

	for {
		ok, err := done()
		if err != nil || ok {
			return err
		}

		select {
		case <-fx.synced:
		case <-watchdog.C:
			return errors.New("legacy client never applied the pushed tool change")
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func legacyCompatAddRouted(peer *registryPeer, version, description, property string) {
	peer.servers["alice"].AddTool(&sdkmcp.Tool{Name: "routed", Description: description, InputSchema: map[string]any{"type": "object", "properties": map[string]any{property: map[string]any{"type": "string"}}}},
		func(_ context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
			return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: version + " " + string(req.Params.Arguments)}}}, nil
		})
}

func legacyCompatDefinition(defs []llm.ToolDefinition, name string) (llm.ToolDefinition, bool) {
	for _, def := range defs {
		if def.Name == name {
			return def, true
		}
	}

	return llm.ToolDefinition{}, false
}

func legacyCompatError(t *testing.T, part *llm.ToolResponsePart) string {
	t.Helper()
	require.True(t, part.IsError, string(part.Result))

	var payload map[string]string
	require.NoError(t, json.Unmarshal(part.Result, &payload), string(part.Result))

	return payload["error"]
}

type legacyCompatInterceptor struct {
	mu   sync.Mutex
	seen map[string]*llm.ToolDefinition
}

func (i *legacyCompatInterceptor) InterceptToolExecution(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
	var definition *llm.ToolDefinition

	if info.Definition != nil {
		copied := *info.Definition
		copied.Parameters = append(json.RawMessage(nil), copied.Parameters...)
		definition = &copied
	}

	i.mu.Lock()
	i.seen[info.Req.ID] = definition
	i.mu.Unlock()

	return next(ctx, info)
}

func (i *legacyCompatInterceptor) definition(id string) (*llm.ToolDefinition, bool) {
	i.mu.Lock()
	defer i.mu.Unlock()

	definition, ok := i.seen[id]

	return definition, ok
}

func TestLegacyClientAgentHotReloadKeepsCapturedStep(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		for _, notify := range []bool{false, true} {
			t.Run(fmt.Sprintf("legacy=%t/list_changed=%t", legacy, notify), func(t *testing.T) {
				t.Parallel()
				peer := newRegistryPeer(t, legacy)
				legacyCompatAddRouted(peer, "v1", "Original route.", "old")
				// Without notifications, a positive server TTL leaves explicit
				// SyncTools as the only way the change can reach the registry.
				registryAdvertiseToolTTL(peer, time.Hour, !notify)

				var opts []ClientOption
				if notify {
					opts = append(opts, WithAutoSync(time.Hour))
				}

				fx := legacyCompatStart(t, peer, opts...)
				interceptor := &legacyCompatInterceptor{seen: make(map[string]*llm.ToolDefinition)}

				// Another caller's identity in the context must not reach the
				// legacy transport, which only knows its static headers.
				ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "bob/tenant-b", "bob:one"), 15*time.Second)
				defer cancel()

				reloaded := func() (bool, error) {
					defs, err := fx.registry.List(ctx)
					if err != nil {
						return false, err
					}

					routed, ok := legacyCompatDefinition(defs, legacyCompatRouted)
					names := registryNames(defs)

					return ok && routed.Description == "Updated route." && slices.Contains(names, legacyCompatAdded) && !slices.Contains(names, legacyCompatShared), nil
				}

				var listsAfterReload int

				model := fakellm.NewFakeModel()
				model.When(fakellm.FirstCall()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
					legacyCompatAddRouted(peer, "v2", "Updated route.", "new")
					peer.servers["alice"].RemoveTools(legacyCompatSharedRaw)
					peer.addTool("alice", "added", "New capability.", "new")

					if notify {
						if err := fx.awaitSync(ctx, reloaded); err != nil {
							return nil, err
						}
					} else if err := fx.client.SyncTools(ctx); err != nil {
						return nil, err
					}

					listsAfterReload = len(peer.requests("tools/list"))

					ok, err := reloaded()
					if err != nil {
						return nil, err
					}

					assert.True(t, ok, "the live registry must already hold the reloaded catalog")

					live, err := fx.registry.Get(ctx, legacyCompatRouted)
					if err != nil {
						return nil, err
					}

					assert.Equal(t, "Updated route.", live.Definition().Description, "the registered wrapper is updated in place")

					assert.Equal(t, []string{legacyCompatStatic, legacyCompatOnly, legacyCompatRouted, legacyCompatShared}, registryNames(req.Tools))
					routed, _ := legacyCompatDefinition(req.Tools, legacyCompatRouted)
					assert.Equal(t, "Original route.", routed.Description)
					assert.Contains(t, string(routed.Parameters), `"old"`)

					return registryAgentRespond(
						llm.NewToolRequestPart("step1-routed", legacyCompatRouted, json.RawMessage(`{"old":"first"}`)),
						llm.NewToolRequestPart("step1-shared", legacyCompatShared, json.RawMessage(`{"value":"old-step"}`)),
						llm.NewToolRequestPart("step1-static", legacyCompatStatic, json.RawMessage(`{"key":"a"}`)),
					)
				})
				model.When(fakellm.CallNumber(2)).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
					assert.Equal(t, []string{legacyCompatStatic, legacyCompatAdded, legacyCompatOnly, legacyCompatRouted}, registryNames(req.Tools))
					routed, _ := legacyCompatDefinition(req.Tools, legacyCompatRouted)
					assert.Equal(t, "Updated route.", routed.Description)
					assert.Contains(t, string(routed.Parameters), `"new"`)

					return registryAgentRespond(
						llm.NewToolRequestPart("step2-routed", legacyCompatRouted, json.RawMessage(`{"new":"second"}`)),
						llm.NewToolRequestPart("step2-added", legacyCompatAdded, json.RawMessage(`{"new":"value"}`)),
						llm.NewToolRequestPart("step2-shared", legacyCompatShared, json.RawMessage(`{"value":"refreshed-step"}`)),
						llm.NewToolRequestPart("step2-static", legacyCompatStatic, json.RawMessage(`{"key":"b"}`)),
					)
				})
				model.When(fakellm.Any()).ThenRespondText("done")

				ag, err := llmagent.New("legacy-reload", "Use local and remote tools.", model, llmagent.WithTools(fx.registry), llmagent.WithInterceptors(interceptor))
				require.NoError(t, err)

				state := &session.State{ID: "legacy-reload", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("route twice"))}}
				registryAgentRun(ctx, t, ag, state)
				require.Len(t, model.CallsMatching(fakellm.Any()), 3)

				calls := registryAgentReadCalls(t, fx.wire, 4)
				names := make([]string, 0, len(calls))
				var routed []json.RawMessage

				for _, call := range calls {
					names = append(names, call.name)
					assert.Equal(t, legacyCompatAuth, call.headers.Get("Authorization"))

					switch call.name {
					case "routed":
						routed = append(routed, call.arguments)
					case legacyCompatSharedRaw:
						assert.JSONEq(t, `{"value":"old-step"}`, string(call.arguments), "only the captured step may dispatch the removed tool")
					}
				}

				assert.ElementsMatch(t, []string{"routed", legacyCompatSharedRaw, "routed", "added"}, names)
				require.Len(t, routed, 2)
				assert.JSONEq(t, `{"old":"first"}`, string(routed[0]))
				assert.JSONEq(t, `{"new":"second"}`, string(routed[1]))

				for _, req := range peer.requests("") {
					assert.Equal(t, legacyCompatAuth, req.auth, "method %q", req.method)
				}

				if notify {
					assert.GreaterOrEqual(t, listsAfterReload, 2)
				} else {
					assert.Equal(t, 2, listsAfterReload, "Start and the explicit SyncTools")
					assert.Len(t, peer.requests("tools/list"), 2, "agent steps must read the local registry without discovering")
				}

				// The captured route executes against the live server: a snapshot
				// freezes local membership and schema, not the remote implementation.
				results := registryAgentResults(state, legacyCompatRouted)
				require.Len(t, results, 2)
				assert.False(t, results[0].IsError, string(results[0].Result))
				assert.JSONEq(t, `[{"type":"text","text":"v2 {\"old\":\"first\"}"}]`, string(results[0].Result))
				assert.False(t, results[1].IsError, string(results[1].Result))
				assert.JSONEq(t, `[{"type":"text","text":"v2 {\"new\":\"second\"}"}]`, string(results[1].Result))

				shared := registryAgentResults(state, legacyCompatShared)
				require.Len(t, shared, 2)
				assert.Contains(t, legacyCompatError(t, shared[0]), `unknown tool "shared"`, "the server has removed the tool the captured step still routes")
				assert.Contains(t, legacyCompatError(t, shared[1]), tool.ErrToolNotFound.Error(), "the refreshed step must reject the removed tool locally")

				added := registryAgentResults(state, legacyCompatAdded)
				require.Len(t, added, 1)
				assert.False(t, added[0].IsError, string(added[0].Result))
				assert.JSONEq(t, `[{"type":"text","text":"Bearer alice:one added"}]`, string(added[0].Result))

				static := registryAgentResults(state, legacyCompatStatic)
				require.Len(t, static, 2)

				for _, result := range static {
					assert.False(t, result.IsError, string(result.Result))
					assert.JSONEq(t, `{"source":"static"}`, string(result.Result))
				}

				assert.Equal(t, int32(2), fx.static.calls.Load())

				first, ok := interceptor.definition("step1-routed")
				require.True(t, ok)
				require.NotNil(t, first)
				assert.Equal(t, "Original route.", first.Description, "interceptors must see the step's captured definition")
				assert.Contains(t, string(first.Parameters), `"old"`)

				removed, ok := interceptor.definition("step1-shared")
				require.True(t, ok)
				require.NotNil(t, removed)
				assert.Equal(t, legacyCompatSharedRaw, removed.Description)

				second, ok := interceptor.definition("step2-routed")
				require.True(t, ok)
				require.NotNil(t, second)
				assert.Equal(t, "Updated route.", second.Description)
				assert.Contains(t, string(second.Parameters), `"new"`)

				stale, ok := interceptor.definition("step2-shared")
				require.True(t, ok)
				assert.Nil(t, stale, "a removed tool must not regain a definition in a refreshed step")

				for _, id := range []string{"step1-static", "step2-static"} {
					local, ok := interceptor.definition(id)
					require.True(t, ok)
					require.NotNil(t, local)
					assert.Equal(t, "Local lookup.", local.Description)
				}
			})
		}
	}
}

func TestLegacyClientAgentResultsStayContentArrays(t *testing.T) {
	t.Parallel()

	// Neither type name is valid JSON Schema; the legacy client forwards the
	// schema unchanged.
	const looseSchema = `{"type":"object","properties":{"mode":{"type":"text"},"limit":{"type":""}}}`

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()
			peer := newRegistryPeer(t, legacy)
			alice := peer.servers["alice"]
			object := map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}}
			alice.AddTool(&sdkmcp.Tool{Name: "rich", Description: "Rich result.", InputSchema: object},
				func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
					return &sdkmcp.CallToolResult{
						Meta:              sdkmcp.Meta{"trace": "t-1"},
						Content:           []sdkmcp.Content{&sdkmcp.TextContent{Text: "first"}, &sdkmcp.ImageContent{Data: []byte("\x89PNG"), MIMEType: "image/png"}},
						StructuredContent: map[string]any{"answer": 42},
					}, nil
				})
			alice.AddTool(&sdkmcp.Tool{Name: "denied", Description: "Tool error.", InputSchema: object},
				func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
					return &sdkmcp.CallToolResult{
						Content:           []sdkmcp.Content{&sdkmcp.TextContent{Text: "denied: read-only"}},
						StructuredContent: map[string]any{"code": 403},
						IsError:           true,
					}, nil
				})

			var loose map[string]any
			require.NoError(t, json.Unmarshal([]byte(looseSchema), &loose))
			alice.AddTool(&sdkmcp.Tool{Name: "loose", Description: "Loose schema.", InputSchema: loose},
				func(_ context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
					return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: string(req.Params.Arguments)}}}, nil
				})

			fx := legacyCompatStart(t, peer)
			model := fakellm.NewFakeModel()
			model.When(fakellm.FirstCall()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				def, ok := legacyCompatDefinition(req.Tools, "old__loose")
				assert.True(t, ok, "the legacy path must not drop a schema with invalid type names: %v", registryNames(req.Tools))
				assert.JSONEq(t, looseSchema, string(def.Parameters))

				return registryAgentRespond(
					llm.NewToolRequestPart("rich", "old__rich", json.RawMessage(`{}`)),
					llm.NewToolRequestPart("denied", "old__denied", json.RawMessage(`{"path":"/etc"}`)),
					llm.NewToolRequestPart("loose", "old__loose", json.RawMessage(`{"mode":"fast","limit":"x"}`)),
				)
			})
			model.When(fakellm.Any()).ThenRespondText("done")

			ag, err := llmagent.New("legacy-results", "Use remote tools.", model, llmagent.WithTools(fx.registry))
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
			defer cancel()

			state := &session.State{ID: "legacy-results", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("call all"))}}
			registryAgentRun(ctx, t, ag, state)

			calls := registryAgentReadCalls(t, fx.wire, 3)
			byName := make(map[string]json.RawMessage, len(calls))

			for _, call := range calls {
				byName[call.name] = call.arguments
			}

			assert.JSONEq(t, `{}`, string(byName["rich"]))
			assert.JSONEq(t, `{"path":"/etc"}`, string(byName["denied"]))
			assert.JSONEq(t, `{"mode":"fast","limit":"x"}`, string(byName["loose"]))

			// Each result is the legacy content array, not the contextual registry's
			// complete CallToolResult with structuredContent, _meta and isError.
			rich := registryAgentResults(state, "old__rich")
			require.Len(t, rich, 1)
			assert.False(t, rich[0].IsError, string(rich[0].Result))
			assert.JSONEq(t, `[{"type":"text","text":"first"},{"type":"image","mimeType":"image/png","data":"iVBORw=="}]`, string(rich[0].Result))

			denied := registryAgentResults(state, "old__denied")
			require.Len(t, denied, 1)
			assert.True(t, denied[0].IsError, "MCP isError must reach the model as a tool error")
			assert.JSONEq(t, `[{"type":"text","text":"denied: read-only"}]`, string(denied[0].Result))

			looseResults := registryAgentResults(state, "old__loose")
			require.Len(t, looseResults, 1)
			assert.False(t, looseResults[0].IsError, string(looseResults[0].Result))
			assert.JSONEq(t, `[{"type":"text","text":"{\"limit\":\"x\",\"mode\":\"fast\"}"}]`, string(looseResults[0].Result))

			requests := model.CallsMatching(fakellm.Any())
			require.Len(t, requests, 2)

			var delivered []*llm.ToolResponsePart
			for _, message := range requests[1].Request.Messages {
				delivered = append(delivered, message.ToolResponses()...)
			}

			require.Len(t, delivered, 3)

			for _, part := range delivered {
				assert.NotContains(t, string(part.Result), "structuredContent")
				assert.NotContains(t, string(part.Result), "_meta")
			}
		})
	}
}

func TestLegacyClientAgentCapturedToolsFailClosedAfterShutdown(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()
			peer := newRegistryPeer(t, legacy)
			fx := legacyCompatStart(t, peer)

			ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
			defer cancel()

			model := fakellm.NewFakeModel()
			model.When(fakellm.FirstCall()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				assert.Equal(t, []string{legacyCompatStatic, legacyCompatOnly, legacyCompatShared}, registryNames(req.Tools))

				if err := fx.client.Close(); err != nil {
					return nil, err
				}

				live, err := fx.registry.List(ctx)
				if err != nil {
					return nil, err
				}

				assert.Equal(t, []string{legacyCompatStatic}, registryNames(live), "shutdown unregisters the client's tools")

				return registryAgentRespond(
					llm.NewToolRequestPart("closed-shared", legacyCompatShared, json.RawMessage(`{"value":"after-close"}`)),
					llm.NewToolRequestPart("closed-static", legacyCompatStatic, json.RawMessage(`{"key":"a"}`)),
				)
			})
			model.When(fakellm.CallNumber(2)).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				assert.Equal(t, []string{legacyCompatStatic}, registryNames(req.Tools))

				return &llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("done")), FinishReason: llm.FinishReasonStop}, nil
			})

			ag, err := llmagent.New("legacy-closed", "Use local and remote tools.", model, llmagent.WithTools(fx.registry))
			require.NoError(t, err)

			state := &session.State{ID: "legacy-closed", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("call both"))}}
			registryAgentRun(ctx, t, ag, state)
			require.Len(t, model.CallsMatching(fakellm.Any()), 2)

			shared := registryAgentResults(state, legacyCompatShared)
			require.Len(t, shared, 1)
			assert.Contains(t, legacyCompatError(t, shared[0]), ErrClosed.Error(), "a captured tool of a closed client must fail closed")

			static := registryAgentResults(state, legacyCompatStatic)
			require.Len(t, static, 1)
			assert.False(t, static[0].IsError, string(static[0].Result))
			assert.Equal(t, int32(1), fx.static.calls.Load())

			assert.Empty(t, peer.requests("tools/call"), "no call may reach the server after shutdown")
			assert.Empty(t, fx.wire)
		})
	}
}
