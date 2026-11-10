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
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/toolloading"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

func toolsetRunError(ctx context.Context, ag *LLMAgent, inv *agent.InvocationMetadata) error {
	for _, err := range ag.Run(ctx, inv) {
		if err != nil {
			return err
		}
	}

	return nil
}

func toolsetResults(sess *session.State, name string) []*llm.ToolResponsePart {
	var results []*llm.ToolResponsePart

	for _, msg := range sess.Messages {
		for _, result := range msg.ToolResponses() {
			if result.Name == name {
				results = append(results, result)
			}
		}
	}

	return results
}

func TestToolsetsRefreshCatalogAndDispatchEachTurn(t *testing.T) {
	t.Parallel()

	old := &stubTool{def: llm.ToolDefinition{Name: "fetch", Description: "old fetch", Parameters: json.RawMessage(`{"type":"object","properties":{"old":{"type":"string"}}}`)}, result: json.RawMessage(`{"version":"old"}`)}
	updated := &stubTool{def: llm.ToolDefinition{Name: "fetch", Description: "new fetch", Parameters: json.RawMessage(`{"type":"object","properties":{"new":{"type":"number"}}}`)}, result: json.RawMessage(`{"version":"new"}`)}
	removed := &stubTool{def: llm.ToolDefinition{Name: "removed", Parameters: json.RawMessage(`{"type":"object"}`)}}
	added := &stubTool{def: llm.ToolDefinition{Name: "added", Parameters: json.RawMessage(`{"type":"object"}`)}}
	first := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, first.Register(old))
	require.NoError(t, first.Register(removed))

	second := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, second.Register(updated))
	require.NoError(t, second.Register(added))
	registries := []tool.Registry{first, second, nil}
	var turns []int
	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWithToolCall("fetch", map[string]any{"old": "value"})
	model.When(fakellm.CallNumber(2)).ThenRespondWith(respondWith(
		toolCall("new", "fetch", `{"new":42}`),
		toolCall("removed", "removed", `{}`),
		toolCall("added", "added", `{}`),
	))
	model.When(fakellm.Any()).ThenRespondText("done")
	ag, err := New("dynamic", "prompt", model, WithToolsets(func(_ context.Context, inv *agent.InvocationMetadata) (tool.Registry, error) {
		turns = append(turns, inv.Turn())
		require.Less(t, inv.Turn(), len(registries))

		return registries[inv.Turn()], nil
	}))
	require.NoError(t, err)

	sess := &session.State{ID: "refresh", Messages: userMessage("fetch")}
	assert.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))

	assert.Equal(t, []int{0, 1, 2}, turns)

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 3)
	assert.Equal(t, []string{"fetch", "removed"}, requestToolNames(calls[0].Request))
	assert.Equal(t, []string{"added", "fetch"}, requestToolNames(calls[1].Request))
	assert.Empty(t, calls[2].Request.Tools)
	assert.Equal(t, "new fetch", defByName(t, calls[1].Request.Tools, "fetch").Description)
	assert.JSONEq(t, string(updated.def.Parameters), string(defByName(t, calls[1].Request.Tools, "fetch").Parameters))
	assert.Equal(t, int32(1), old.calls.Load())
	assert.Equal(t, int32(1), updated.calls.Load())
	assert.Equal(t, int32(1), added.calls.Load())
	assert.Zero(t, removed.calls.Load())
	require.Len(t, toolsetResults(sess, "removed"), 1)
	assert.True(t, toolsetResults(sess, "removed")[0].IsError)
	assert.Contains(t, toolResultsFor(sess.Messages, "fetch")[1], `"version":"new"`)
}

func TestToolsetsComposeStaticAndAppendOptions(t *testing.T) {
	t.Parallel()

	static := newFixtureRegistry(t, fixtureTool{name: "static"})
	one := newFixtureRegistry(t, fixtureTool{name: "one", group: "shared", groupInfo: llm.ToolGroup{Instructions: "Shared group instructions."}})
	two := newFixtureRegistry(t, fixtureTool{name: "two", group: "shared"})
	var order []string
	set := func(name string, registry tool.Registry) Toolset {
		return func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
			order = append(order, name)

			return registry, nil
		}
	}
	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWith(respondWith(toolCall("s", "static", `{}`), toolCall("1", "one", `{}`), toolCall("2", "two", `{}`)))
	model.When(fakellm.Any()).ThenRespondText("done")
	ag, err := New("composed", "prompt", model,
		WithToolsets(set("one", one)),
		WithTools(static),
		WithToolsets(set("empty", nil), set("two", two)),
	)
	require.NoError(t, err)

	sess := &session.State{ID: "composition", Messages: userMessage("run all")}
	runAgent(t, ag, sess)

	assert.Equal(t, []string{"one", "empty", "two", "one", "empty", "two"}, order)

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 2)

	for _, call := range calls {
		assert.Equal(t, []string{"one", "static", "two"}, requestToolNames(call.Request))
		assert.Contains(t, call.Request.Messages[0].TextContent(), "Shared group instructions.")
		assert.Equal(t, defByName(t, call.Request.Tools, "one").Group, defByName(t, call.Request.Tools, "two").Group)
	}

	for _, name := range []string{"static", "one", "two"} {
		require.Len(t, toolsetResults(sess, name), 1)
		assert.False(t, toolsetResults(sess, name)[0].IsError)
	}
}

func TestToolsetsIsolateConcurrentCallerIdentities(t *testing.T) {
	t.Parallel()

	identities := []string{"alice", "bob"}
	registries := make(map[string]tool.Registry, len(identities))

	tools := make(map[string]*stubTool, len(identities))
	for _, identity := range identities {
		registry := tool.NewRegistry(tool.RegistryConfig{})
		probe := &stubTool{def: llm.ToolDefinition{Name: "whoami", Description: identity, Parameters: json.RawMessage(`{"type":"object"}`)}, result: fmt.Appendf(nil, `{"identity":%q}`, identity)}
		require.NoError(t, registry.Register(probe))
		registries[identity], tools[identity] = registry, probe
	}

	arrived := make(chan string, len(identities))
	release := make(chan struct{})

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.LastMessageHasToolResponse("whoami")).ThenRespondText("done")
	model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		identity := req.Messages[len(req.Messages)-1].TextContent()
		assert.Equal(t, identity, defByName(t, req.Tools, "whoami").Description)

		arrived <- identity

		select {
		case <-release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}

		return respondWith(toolCall(identity, "whoami", `{}`))(req, nil)
	})
	var resolutions atomic.Int32
	ag, err := New("shared-agent", "prompt", model, WithToolsets(func(_ context.Context, inv *agent.InvocationMetadata) (tool.Registry, error) {
		resolutions.Add(1)

		return registries[inv.Attribute(agent.AttrUserID)], nil
	}))
	require.NoError(t, err)

	var wg sync.WaitGroup
	sessions := make([]*session.State, len(identities))

	errs := make([]error, len(identities))
	for i, identity := range identities {
		sessions[i] = &session.State{ID: identity, Messages: userMessage(identity)}

		wg.Go(func() {
			inv := agent.NewInvocationMetadata(sessions[i], ag.Info(), agent.WithAttributes(map[string]string{agent.AttrUserID: identity}))
			errs[i] = toolsetRunError(ctx, ag, inv)
		})
	}

	seen := make([]string, 0, len(identities))
	for range identities {
		select {
		case identity := <-arrived:
			seen = append(seen, identity)
		case <-ctx.Done():
			t.Error("both invocations must reach the model before either dispatches tools")
		}
	}

	close(release)
	wg.Wait()
	assert.ElementsMatch(t, identities, seen)
	assert.Equal(t, int32(4), resolutions.Load())

	for i, identity := range identities {
		require.NoError(t, errs[i])
		assert.Equal(t, int32(1), tools[identity].calls.Load())
		require.Len(t, toolResultsFor(sessions[i].Messages, "whoami"), 1)
		assert.JSONEq(t, fmt.Sprintf(`{"identity":%q}`, identity), toolResultsFor(sessions[i].Messages, "whoami")[0])
	}
}

func TestToolsetsSnapshotDispatchSurvivesSourceReplacement(t *testing.T) {
	t.Parallel()

	original := &stubTool{def: llm.ToolDefinition{Name: "fetch", Description: "original", Parameters: json.RawMessage(`{"type":"object","properties":{"old":{"type":"string"}}}`)}, result: json.RawMessage(`{"source":"original"}`)}
	replacement := &stubTool{def: llm.ToolDefinition{Name: "fetch", Description: "replacement", Parameters: json.RawMessage(`{"type":"object","properties":{"new":{"type":"string"}}}`)}, result: json.RawMessage(`{"source":"replacement"}`)}
	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(original))
	originalSchema := string(original.def.Parameters)
	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWith(func(req *llm.Request, cc *fakellm.CallContext) (*llm.Response, error) {
		require.NoError(t, registry.Unregister("fetch"))
		require.NoError(t, registry.Register(replacement))
		copy(original.def.Parameters, replacement.def.Parameters)
		original.def.Description = "mutated after planning"
		assert.Equal(t, "original", defByName(t, req.Tools, "fetch").Description)
		assert.JSONEq(t, originalSchema, string(defByName(t, req.Tools, "fetch").Parameters))

		return respondWith(toolCall("old", "fetch", `{"old":"value"}`))(req, cc)
	})
	model.When(fakellm.CallNumber(2)).ThenRespondWithToolCall("fetch", map[string]any{"new": "value"})
	model.When(fakellm.Any()).ThenRespondText("done")
	ag, err := New("snapshot", "prompt", model, WithToolsets(func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
		return registry, nil
	}))
	require.NoError(t, err)

	sess := &session.State{ID: "snapshot", Messages: userMessage("fetch twice")}
	runAgent(t, ag, sess)

	assert.Equal(t, int32(1), original.calls.Load())
	assert.Equal(t, int32(1), replacement.calls.Load())
	assert.Equal(t, []string{`{"source":"original"}`, `{"source":"replacement"}`}, toolResultsFor(sess.Messages, "fetch"))

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 3)
	assert.Equal(t, "replacement", defByName(t, calls[1].Request.Tools, "fetch").Description)
	assert.JSONEq(t, string(replacement.def.Parameters), string(defByName(t, calls[1].Request.Tools, "fetch").Parameters))
}

type toolsetExecutionProbe struct {
	def     llm.ToolDefinition
	execute func(context.Context) (json.RawMessage, error)
}

func (p toolsetExecutionProbe) Definition() llm.ToolDefinition { return p.def }

func (p toolsetExecutionProbe) Execute(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
	return p.execute(ctx)
}

func TestToolsetsPreserveRegistrationExecutionPolicies(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(toolsetExecutionProbe{
		def: llm.ToolDefinition{Name: "slow", Parameters: json.RawMessage(`{"type":"object"}`)},
		execute: func(ctx context.Context) (json.RawMessage, error) {
			deadline, ok := ctx.Deadline()
			if !ok || time.Until(deadline) > time.Second {
				return nil, errors.New("per-tool timeout was lost")
			}

			<-ctx.Done()

			return nil, ctx.Err()
		},
	}, tool.WithTimeout(10*time.Millisecond)))
	require.NoError(t, registry.Register(&stubTool{
		def: llm.ToolDefinition{Name: "large", Parameters: json.RawMessage(`{"type":"object"}`)}, result: fmt.Appendf(nil, `{"data":%q}`, strings.Repeat("large result ", 1000)),
	}, tool.WithMaxResponseTokens(10), tool.WithResponseTooLargeMessage("Use a smaller page.")))

	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWith(respondWith(toolCall("s", "slow", `{}`), toolCall("l", "large", `{}`)))
	model.When(fakellm.Any()).ThenRespondText("done")
	ag, err := New("policies", "prompt", model, WithToolsets(func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
		return registry, nil
	}))
	require.NoError(t, err)

	sess := &session.State{ID: "policies", Messages: userMessage("run")}
	runAgent(t, ag, sess)

	require.Len(t, toolsetResults(sess, "slow"), 1)
	assert.True(t, toolsetResults(sess, "slow")[0].IsError)
	assert.Contains(t, string(toolsetResults(sess, "slow")[0].Result), tool.ErrToolExecutionTimeout.Error())
	assert.Contains(t, string(toolsetResults(sess, "slow")[0].Result), "10ms")
	require.Len(t, toolResultsFor(sess.Messages, "large"), 1)
	assert.Contains(t, toolResultsFor(sess.Messages, "large")[0], "Use a smaller page.")
	assert.NotContains(t, toolResultsFor(sess.Messages, "large")[0], strings.Repeat("large result ", 10))
}

func TestToolsetsRejectInvalidCompositionBeforeModel(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		static     []fixtureTool
		sets       [][]fixtureTool
		wantErr    error
		wantDetail string
	}{
		{name: "static duplicate", static: []fixtureTool{{name: "duplicate"}}, sets: [][]fixtureTool{{{name: "duplicate"}}}, wantErr: tool.ErrToolAlreadyRegistered},
		{name: "dynamic duplicate", sets: [][]fixtureTool{{{name: "duplicate"}}, {{name: "duplicate"}}}, wantErr: tool.ErrToolAlreadyRegistered},
		{name: "group descriptions", sets: [][]fixtureTool{{{name: "one", group: "svc", groupInfo: llm.ToolGroup{Description: "first"}}}, {{name: "two", group: "svc", groupInfo: llm.ToolGroup{Description: "second"}}}}, wantErr: tool.ErrInvalidToolConfig},
		{name: "group instructions", static: []fixtureTool{{name: "one", group: "svc", groupInfo: llm.ToolGroup{Instructions: "first"}}}, sets: [][]fixtureTool{{{name: "two", group: "svc", groupInfo: llm.ToolGroup{Instructions: "second"}}}}, wantErr: tool.ErrInvalidToolConfig},
		{name: "reserved search", sets: [][]fixtureTool{{{name: toolloading.SearchToolName}}, {{name: "deferred", deferred: true}}}, wantDetail: toolloading.SearchToolName},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			model := fakellm.NewFakeModel()
			model.When(fakellm.Any()).ThenRespondText("must not run")

			opts := []Option{WithCompaction(CompactionConfig{})}
			if tc.static != nil {
				opts = append(opts, WithTools(newFixtureRegistry(t, tc.static...)))
			}

			for _, specs := range tc.sets {
				registry := newFixtureRegistry(t, specs...)

				opts = append(opts, WithToolsets(func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
					return registry, nil
				}))
			}

			ag, err := New("invalid", "prompt", model, opts...)
			require.NoError(t, err, "dynamic collisions are checked when a turn resolves its tools")

			err = toolsetRunError(t.Context(), ag, agent.NewInvocationMetadata(&session.State{ID: "invalid", Messages: userMessage("run")}, ag.Info()))
			if tc.wantErr != nil {
				require.ErrorIs(t, err, tc.wantErr)
			} else {
				require.ErrorContains(t, err, tc.wantDetail)
			}

			assert.Empty(t, model.CallsMatching(fakellm.Any()))
		})
	}
}

func TestToolsetsErrorsAndCancellation(t *testing.T) {
	t.Parallel()

	failure := errors.New("caller catalog unavailable")
	for _, tc := range []struct {
		name string
		err  error
	}{
		{name: "source error", err: failure},
		{name: "canceled", err: context.Canceled},
		{name: "deadline", err: context.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			model := fakellm.NewFakeModel()
			var nextCalls int
			ag, err := New("errors", "prompt", model,
				WithToolsets(func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
					return nil, fmt.Errorf("resolve caller: %w", tc.err)
				}, func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
					nextCalls++

					return nil, nil //nolint:nilnil // A nil registry denotes an empty toolset.
				}),
			)
			require.NoError(t, err)
			err = toolsetRunError(t.Context(), ag, agent.NewInvocationMetadata(&session.State{ID: tc.name, Messages: userMessage("run")}, ag.Info()))
			require.ErrorIs(t, err, tc.err)
			assert.Zero(t, nextCalls)
			assert.Empty(t, model.CallsMatching(fakellm.Any()))
		})
	}

	t.Run("canceled during resolution", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()

		model := fakellm.NewFakeModel()
		ag, err := New("cancel", "prompt", model, WithToolsets(func(ctx context.Context, _ *agent.InvocationMetadata) (tool.Registry, error) {
			cancel()
			<-ctx.Done()

			return nil, ctx.Err()
		}))
		require.NoError(t, err)
		err = toolsetRunError(ctx, ag, agent.NewInvocationMetadata(&session.State{ID: "cancel", Messages: userMessage("run")}, ag.Info()))
		require.ErrorIs(t, err, context.Canceled)
		assert.Empty(t, model.CallsMatching(fakellm.Any()))
	})
}

type toolsetTurnInterceptor func(context.Context, *agent.TurnInfo, agent.TurnNext) (agent.FinishReason, error)

func (f toolsetTurnInterceptor) InterceptTurn(ctx context.Context, info *agent.TurnInfo, next agent.TurnNext) (agent.FinishReason, error) {
	return f(ctx, info, next)
}

func TestToolsetsReceiveTurnContextAndInvocation(t *testing.T) {
	t.Parallel()

	type contextKey struct{}
	sess := &session.State{ID: "metadata", Messages: userMessage("hello")}
	inv := agent.NewInvocationMetadata(sess, agent.Info{}, agent.WithAttributes(map[string]string{agent.AttrUserID: "alice"}))
	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondText("done")
	var resolutions int
	ag, err := New("metadata", "prompt", model,
		WithInterceptors(toolsetTurnInterceptor(func(ctx context.Context, info *agent.TurnInfo, next agent.TurnNext) (agent.FinishReason, error) {
			info.Inv.SetMetadata("turn_marker", "from interceptor")

			return next(context.WithValue(ctx, contextKey{}, "turn context"), info)
		})),
		WithToolsets(func(ctx context.Context, actual *agent.InvocationMetadata) (tool.Registry, error) {
			resolutions++

			assert.Same(t, inv, actual)
			assert.Same(t, sess, actual.Session())
			assert.Equal(t, "alice", actual.Attribute(agent.AttrUserID))
			assert.Equal(t, "from interceptor", actual.GetMetadata("turn_marker"))
			assert.Equal(t, "turn context", ctx.Value(contextKey{}))

			return nil, nil //nolint:nilnil // A nil registry denotes an empty toolset.
		}),
	)
	require.NoError(t, err)
	require.NoError(t, toolsetRunError(t.Context(), ag, inv))
	assert.Equal(t, 1, resolutions)
}

type toolsetUnsnapshotableRegistry struct {
	tool.Registry
}

func TestToolsetsRejectUnsnapshotableRegistryBeforeModel(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	ag, err := New("unsupported", "prompt", model, WithToolsets(func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
		return toolsetUnsnapshotableRegistry{Registry: newFixtureRegistry(t, fixtureTool{name: "fetch"})}, nil
	}))
	require.NoError(t, err)
	err = toolsetRunError(t.Context(), ag, agent.NewInvocationMetadata(&session.State{ID: "unsupported", Messages: userMessage("hello")}, ag.Info()))
	require.ErrorIs(t, err, tool.ErrRegistrySnapshotUnsupported)
	assert.Empty(t, model.CallsMatching(fakellm.Any()))
}

func TestToolsetsEmptyAndNilOptions(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		opts []Option
	}{
		{name: "no sets", opts: []Option{WithToolsets()}},
		{name: "nil registry", opts: []Option{WithToolsets(func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
			return nil, nil //nolint:nilnil // A nil registry denotes an empty toolset.
		})}},
		{name: "empty registry", opts: []Option{WithToolsets(func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
			return tool.NewRegistry(tool.RegistryConfig{}), nil
		})}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			model := fakellm.NewFakeModel()
			model.When(fakellm.Any()).ThenRespondText("done")
			ag, err := New("empty", "prompt", model, tc.opts...)
			require.NoError(t, err)
			runAgent(t, ag, &session.State{ID: "empty", Messages: userMessage("hello")})

			calls := model.CallsMatching(fakellm.Any())
			require.Len(t, calls, 1)
			assert.Empty(t, calls[0].Request.Tools)
			assert.False(t, calls[0].Request.ToolSearch)
		})
	}

	t.Run("nil toolset", func(t *testing.T) {
		t.Parallel()

		_, err := New("nil", "prompt", fakellm.NewFakeModel(), WithToolsets(func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
			return nil, nil //nolint:nilnil // A nil registry denotes an empty toolset.
		}), WithToolsets(nil))
		require.ErrorContains(t, err, "toolset")
		require.ErrorContains(t, err, "nil")
	})
}

func TestToolsetsLocalDiscoveryTracksUpdatesAndRevocation(t *testing.T) {
	t.Parallel()

	first := newFixtureRegistry(t, fixtureTool{name: "fetch", deferred: true, description: "old catalog entry", group: "svc", groupInfo: llm.ToolGroup{Instructions: "Old instructions."}})
	second := newFixtureRegistry(t,
		fixtureTool{name: "fetch", deferred: true, description: "updated catalog entry", params: `{"type":"object","properties":{"version":{"type":"number"}}}`, group: "svc", groupInfo: llm.ToolGroup{Instructions: "New instructions."}},
		fixtureTool{name: "introduced", deferred: true, description: "newly available", group: "svc"},
	)
	third := newFixtureRegistry(t, fixtureTool{name: "introduced", deferred: true, description: "newly available", group: "svc", groupInfo: llm.ToolGroup{Instructions: "New instructions."}})
	registries := []tool.Registry{first, second, third, nil}
	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWithToolCall(toolloading.SearchToolName, map[string]any{"query": "select:fetch"})
	model.When(fakellm.CallNumber(2)).ThenRespondWith(respondWith(
		toolCall("fetch", "fetch", `{"version":2}`),
		toolCall("search", toolloading.SearchToolName, `{"query":"select:introduced"}`),
	))
	model.When(fakellm.CallNumber(3)).ThenRespondWith(respondWith(toolCall("stale", "fetch", `{}`), toolCall("new", "introduced", `{}`)))
	model.When(fakellm.Any()).ThenRespondText("done")
	ag, err := New("lazy-dynamic", "prompt", model, WithToolsets(func(_ context.Context, inv *agent.InvocationMetadata) (tool.Registry, error) {
		require.Less(t, inv.Turn(), len(registries))

		return registries[inv.Turn()], nil
	}))
	require.NoError(t, err)

	sess := &session.State{ID: "lazy-dynamic", Messages: userMessage("fetch")}
	runAgent(t, ag, sess)

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 4)
	assert.Equal(t, []string{toolloading.SearchToolName}, requestToolNames(calls[0].Request))
	assert.Contains(t, calls[0].Request.Messages[0].TextContent(), "old catalog entry")
	assert.NotContains(t, calls[0].Request.Messages[0].TextContent(), "Old instructions.")
	assert.Equal(t, []string{"fetch", toolloading.SearchToolName}, requestToolNames(calls[1].Request))
	assert.Contains(t, calls[1].Request.Messages[0].TextContent(), "updated catalog entry")
	assert.Contains(t, calls[1].Request.Messages[0].TextContent(), "New instructions.")
	assert.NotContains(t, calls[1].Request.Messages[0].TextContent(), "old catalog entry")
	assert.NotContains(t, calls[1].Request.Messages[0].TextContent(), "Old instructions.")
	assert.JSONEq(t, `{"type":"object","properties":{"version":{"type":"number"}}}`, string(defByName(t, calls[1].Request.Tools, "fetch").Parameters))
	assert.Equal(t, []string{"introduced", toolloading.SearchToolName}, requestToolNames(calls[2].Request))
	assert.NotContains(t, calls[2].Request.Messages[0].TextContent(), "`fetch`")
	assert.Empty(t, calls[3].Request.Tools)
	assert.NotContains(t, calls[3].Request.Messages[0].TextContent(), "New instructions.")
	assert.NotContains(t, calls[3].Request.Messages[0].TextContent(), "Additional tools")

	results := toolsetResults(sess, "fetch")
	require.Len(t, results, 2)
	assert.False(t, results[0].IsError)
	assert.True(t, results[1].IsError, "loaded-session metadata must not authorize a revoked tool")
	require.Len(t, toolsetResults(sess, "introduced"), 1)
	assert.False(t, toolsetResults(sess, "introduced")[0].IsError)
	searches := toolResultsFor(sess.Messages, toolloading.SearchToolName)
	require.Len(t, searches, 2)
	assert.Contains(t, searches[1], `"introduced"`)
}

func TestToolsetsNativeDiscoveryTracksRevocation(t *testing.T) {
	t.Parallel()

	for _, tc := range nativeSearchCases {
		t.Run(string(tc.provider), func(t *testing.T) {
			t.Parallel()

			first := newFixtureRegistry(t, fixtureTool{name: "fetch", group: "old", deferred: true, groupInfo: llm.ToolGroup{Description: "Old directory", Instructions: "Old native instructions."}})
			second := newFixtureRegistry(t, fixtureTool{name: "introduced", group: "new", deferred: true, groupInfo: llm.ToolGroup{Description: "New directory", Instructions: "New native instructions."}})
			fake := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{Tools: true, Streaming: true, ToolSearchProtocol: tc.protocol}))
			fake.When(fakellm.FirstCall()).ThenRespondWith(respondWith(
				&llm.ToolSearchPart{Provider: llm.ProviderID(tc.protocol), Data: json.RawMessage(`{"type":"native_search_result"}`), Tools: []string{"fetch"}},
				toolCall("first", "fetch", `{}`),
			))
			fake.When(fakellm.CallNumber(2)).ThenRespondWith(respondWith(
				&llm.ToolSearchPart{Provider: llm.ProviderID(tc.protocol), Data: json.RawMessage(`{"type":"native_search_result"}`), Tools: []string{"introduced"}},
				toolCall("new", "introduced", `{}`),
				toolCall("stale", "fetch", `{}`),
			))
			fake.When(fakellm.Any()).ThenRespondText("done")

			registries := []tool.Registry{first, second, nil}
			ag, err := New("native-dynamic", "prompt", nativeSearchModel{FakeModel: fake, provider: tc.provider}, WithToolsets(func(_ context.Context, inv *agent.InvocationMetadata) (tool.Registry, error) {
				require.Less(t, inv.Turn(), len(registries))

				return registries[inv.Turn()], nil
			}))
			require.NoError(t, err)

			sess := &session.State{ID: "native-dynamic", Messages: userMessage("fetch")}
			runAgent(t, ag, sess)

			calls := fake.CallsMatching(fakellm.Any())
			require.Len(t, calls, 3)
			assert.True(t, calls[0].Request.ToolSearch)
			assert.True(t, calls[1].Request.ToolSearch)
			assert.Equal(t, []string{"fetch"}, requestToolNames(calls[0].Request))
			assert.Equal(t, []string{"introduced"}, requestToolNames(calls[1].Request))
			assert.True(t, calls[0].Request.Tools[0].Deferred)
			assert.True(t, calls[1].Request.Tools[0].Deferred)
			assert.Contains(t, calls[1].Request.Messages[0].TextContent(), "New directory")
			assert.Contains(t, calls[1].Request.Messages[0].TextContent(), "New native instructions.")
			assert.NotContains(t, calls[1].Request.Messages[0].TextContent(), "Old directory")
			assert.NotContains(t, calls[1].Request.Messages[0].TextContent(), "Old native instructions.")
			assert.False(t, calls[2].Request.ToolSearch)
			assert.Empty(t, calls[2].Request.Tools)

			results := toolsetResults(sess, "fetch")
			require.Len(t, results, 2)
			assert.False(t, results[0].IsError)
			assert.True(t, results[1].IsError)
			require.Len(t, toolsetResults(sess, "introduced"), 1)
			assert.False(t, toolsetResults(sess, "introduced")[0].IsError)
			assert.Empty(t, toolResultsFor(sess.Messages, toolloading.SearchToolName))
		})
	}
}

func TestToolsetsSchemasParticipateInPrecallBudget(t *testing.T) {
	t.Parallel()

	for _, deferred := range []bool{false, true} {
		t.Run(fmt.Sprintf("deferred=%t", deferred), func(t *testing.T) {
			t.Parallel()

			registry := newFixtureRegistry(t, fixtureTool{name: "huge", deferred: deferred, description: "brief catalog entry", params: fmt.Sprintf(`{"type":"object","description":%q}`, strings.Repeat("schema detail ", 10_000))})
			model := fakellm.NewFakeModel(fakellm.WithContextWindow(16_000))
			model.When(fakellm.Any()).ThenRespondText("done")
			var resolutions int
			ag, err := New("budget", "prompt", model, WithCompaction(CompactionConfig{}), WithToolsets(func(context.Context, *agent.InvocationMetadata) (tool.Registry, error) {
				resolutions++

				return registry, nil
			}))
			require.NoError(t, err)
			err = toolsetRunError(t.Context(), ag, agent.NewInvocationMetadata(&session.State{ID: "budget", Messages: userMessage("hello")}, ag.Info()))
			assert.Equal(t, 1, resolutions)

			if !deferred {
				require.ErrorIs(t, err, llm.ErrContextOverflow)
				assert.Empty(t, model.CallsMatching(fakellm.Any()), "dynamic schemas must be budgeted before sending")
			} else {
				require.NoError(t, err)

				calls := model.CallsMatching(fakellm.Any())
				require.Len(t, calls, 1)
				assert.Equal(t, []string{toolloading.SearchToolName}, requestToolNames(calls[0].Request))
			}
		})
	}
}

func TestToolsetsReactiveRetryReusesSnapshot(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	original := &stubTool{def: llm.ToolDefinition{Name: "original", Parameters: json.RawMessage(`{"type":"object"}`)}}
	replacement := &stubTool{def: llm.ToolDefinition{Name: "replacement", Parameters: json.RawMessage(`{"type":"object"}`)}}

	require.NoError(t, registry.Register(original))

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(200_000))
	model.When(fakellm.FirstCall()).ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		require.NoError(t, registry.Unregister("original"))
		require.NoError(t, registry.Register(replacement))

		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, &llm.ProviderError{Base: llm.ErrContextOverflow, Code: "invalid_request_error", Message: "prompt is too long"})
	})
	model.When(fakellm.CallNumber(2)).ThenRespondWithToolCall("original", map[string]any{})
	model.When(fakellm.Any()).ThenRespondText("done")
	var turns []int
	ag, err := New("retry", "prompt", model, WithCompaction(CompactionConfig{}), WithToolsets(func(_ context.Context, inv *agent.InvocationMetadata) (tool.Registry, error) {
		turns = append(turns, inv.Turn())

		return registry, nil
	}))
	require.NoError(t, err)

	sess := &session.State{ID: "retry", Messages: userMessage("start")}

	for i := range 4 {
		id := fmt.Sprintf("old_%d", i)
		sess.Messages = append(sess.Messages,
			llm.NewMessage(llm.RoleAssistant, toolCall(id, "original", `{}`)),
			llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(id, "original", fmt.Appendf(nil, `{"data":%q}`, strings.Repeat("x", 16_000)), false)),
		)
	}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")))
	events := runAgent(t, ag, sess)

	assert.Equal(t, []int{0, 1}, turns, "retry must not resolve a new catalog")

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 3)
	assert.Equal(t, []string{"original"}, requestToolNames(calls[0].Request))
	assert.Equal(t, calls[0].Request.Tools, calls[1].Request.Tools)
	assert.Equal(t, []string{"replacement"}, requestToolNames(calls[2].Request))
	assert.Less(t, calls[1].InputTokens, calls[0].InputTokens)
	assert.Equal(t, int32(1), original.calls.Load())
	assert.Zero(t, replacement.calls.Load())
	var reactive bool

	for _, event := range events {
		if compacted, ok := event.(agent.CompactionEvent); ok && compacted.Report.Phase == agent.CompactionPhaseReactive {
			reactive = true
		}
	}

	assert.True(t, reactive)
}
