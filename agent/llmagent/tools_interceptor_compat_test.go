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
	"fmt"
	"slices"
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

const compatPrefix = "usermcp__"

type compatCallerKey struct{}

func compatCaller(ctx context.Context) string {
	caller, _ := ctx.Value(compatCallerKey{}).(string)

	return caller
}

type compatToolCall struct {
	caller     string
	session    string
	turn       int
	name       string
	routed     bool
	definition *llm.ToolDefinition
}

// compatUserTools reproduces the public contract of a per-user tool
// interceptor layered on WithTools: one value implements both halves, appends
// the caller's tools after the static definitions without wrapping the
// handler, routes only its own prefix, and falls through for everything else.
type compatUserTools struct {
	discover func(ctx context.Context, caller string, attempt int) []llm.ToolDefinition
	execute  func(ctx context.Context, caller string, info *agent.ToolCallInfo) (json.RawMessage, error)

	attempts atomic.Int32

	mu    sync.Mutex
	calls []compatToolCall
}

func (u *compatUserTools) InterceptModel(ctx context.Context, info *agent.ModelCallInfo, next agent.ModelCallHandler) agent.ModelCallHandler {
	attempt := int(u.attempts.Add(1))

	caller := compatCaller(ctx)
	if caller == "" {
		return next
	}

	info.Req.Tools = append(info.Req.Tools, u.discover(ctx, caller, attempt)...)

	return next
}

func (u *compatUserTools) InterceptToolExecution(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
	caller := compatCaller(ctx)
	routed := caller != "" && strings.HasPrefix(info.Req.Name, compatPrefix)
	u.record(caller, info, routed)

	if !routed {
		return next(ctx, info)
	}

	result, err := u.execute(ctx, caller, info)
	if err != nil {
		return nil, fmt.Errorf("tool %q failed: %w", info.Req.Name, err)
	}

	return &llm.ToolResponsePart{ID: info.Req.ID, Name: info.Req.Name, Result: result}, nil
}

func (u *compatUserTools) record(caller string, info *agent.ToolCallInfo, routed bool) {
	u.mu.Lock()
	defer u.mu.Unlock()

	u.calls = append(u.calls, compatToolCall{
		caller: caller, session: info.Inv.Session().ID, turn: info.Inv.Turn(), name: info.Req.Name, routed: routed,
		definition: compatCopyDefinition(info.Definition),
	})
}

func (u *compatUserTools) recorded() []compatToolCall {
	u.mu.Lock()
	defer u.mu.Unlock()

	return slices.Clone(u.calls)
}

func compatCopyDefinition(def *llm.ToolDefinition) *llm.ToolDefinition {
	if def == nil {
		return nil
	}

	return &cloneToolDefinitions([]llm.ToolDefinition{*def})[0]
}

func compatDef(name, owner string) llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        name,
		Description: fmt.Sprintf("%s for %s", name, owner),
		Parameters:  fmt.Appendf(nil, `{"type":"object","properties":{"q":{"type":"string","description":%q}}}`, owner),
		Type:        "function",
		Metadata:    map[string]any{"owner": owner, "server": "crm"},
	}
}

func compatFind(defs []llm.ToolDefinition, name string) (llm.ToolDefinition, int) {
	var (
		found llm.ToolDefinition
		count int
	)

	for _, def := range defs {
		if def.Name == name {
			found, count = def, count+1
		}
	}

	return found, count
}

func compatRun(ctx context.Context, ag *LLMAgent, inv *agent.InvocationMetadata) ([]agent.Event, error) {
	var events []agent.Event

	for event, err := range ag.Run(ctx, inv) {
		if err != nil {
			return events, err
		}

		events = append(events, event)
	}

	return events, nil
}

func compatAwait(ctx context.Context, t *testing.T, arrived <-chan string, want []string, msg string) {
	t.Helper()

	seen := make([]string, 0, len(want))
	for range want {
		select {
		case caller := <-arrived:
			seen = append(seen, caller)
		case <-ctx.Done():
			t.Errorf("%s: saw %v before the watchdog expired", msg, seen)

			return
		}
	}

	assert.ElementsMatch(t, want, seen, msg)
}

func TestInterceptorCompatInjectedToolsDispatchBesideStaticSnapshot(t *testing.T) {
	t.Parallel()

	alpha := &stubTool{def: llm.ToolDefinition{Name: "alpha", Description: "static alpha", Parameters: json.RawMessage(`{"type":"object"}`)}, result: json.RawMessage(`{"static":"alpha"}`)}
	zeta := &stubTool{def: llm.ToolDefinition{Name: "zeta", Description: "static zeta", Parameters: json.RawMessage(`{"type":"object"}`)}, result: json.RawMessage(`{"static":"zeta"}`)}
	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(zeta))
	require.NoError(t, registry.Register(alpha))

	search, create := compatDef(compatPrefix+"search", "alice"), compatDef(compatPrefix+"create", "alice")

	var discoveries atomic.Int32
	user := &compatUserTools{
		discover: func(_ context.Context, caller string, _ int) []llm.ToolDefinition {
			discoveries.Add(1)
			assert.Equal(t, "alice", caller)

			return []llm.ToolDefinition{search, create}
		},
		execute: func(_ context.Context, caller string, info *agent.ToolCallInfo) (json.RawMessage, error) {
			return fmt.Appendf(nil, `{"caller":%q,"tool":%q,"args":%s}`, caller, info.Req.Name, info.Req.Arguments), nil
		},
	}

	var (
		mu       sync.Mutex
		outerLog []compatToolCall
	)

	outer := toolInterceptorFunc(func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
		mu.Lock()

		outerLog = append(outerLog, compatToolCall{turn: info.Inv.Turn(), name: info.Req.Name, definition: compatCopyDefinition(info.Definition)})
		mu.Unlock()

		return next(ctx, info)
	})

	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWith(respondWith(
		toolCall("s1", "alpha", `{}`),
		toolCall("u1", search.Name, `{"q":"pipelines"}`),
		toolCall("u2", create.Name, `{"q":"ticket"}`),
	))
	model.When(fakellm.CallNumber(2)).ThenRespondWith(respondWith(
		toolCall("u3", search.Name, `{"q":"again"}`),
		toolCall("s2", "zeta", `{}`),
	))
	model.When(fakellm.Any()).ThenRespondText("done")

	ag, err := New("compat", "prompt", model, WithTools(registry), WithInterceptors(outer, user))
	require.NoError(t, err)

	sess := &session.State{ID: "compat", Messages: userMessage("go")}
	events, err := compatRun(context.WithValue(t.Context(), compatCallerKey{}, "alice"), ag, agent.NewInvocationMetadata(sess, ag.Info()))
	require.NoError(t, err)
	require.Equal(t, agent.FinishReasonStop, finishReason(events))

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 3)
	assert.Equal(t, int32(3), discoveries.Load(), "one discovery per model call")

	for i, call := range calls {
		assert.Equal(t, fakellm.CallGenerateEvents, call.Kind, "call %d", i)
		assert.Equal(t, []string{"alpha", "zeta", search.Name, create.Name}, requestToolNames(call.Request), "sorted static prefix, then the injected block once, call %d", i)
		assert.Equal(t, calls[0].Request.Tools, call.Request.Tools, "request tools must be stable across steps, call %d", i)
	}

	assert.Equal(t, search, defByName(t, calls[0].Request.Tools, search.Name))
	assert.Equal(t, int32(1), alpha.calls.Load())
	assert.Equal(t, int32(1), zeta.calls.Load())

	inner := user.recorded()
	require.Len(t, inner, 5)
	require.Len(t, outerLog, 5)

	for _, seen := range slices.Concat(inner, outerLog) {
		require.NotNil(t, seen.definition, "definition for %s", seen.name)
		require.Less(t, seen.turn, len(calls))
		assert.Equal(t, defByName(t, calls[seen.turn].Request.Tools, seen.name), *seen.definition, "interceptor definition must be the one the model saw for %s", seen.name)
	}

	for _, seen := range inner {
		assert.Equal(t, "alice", seen.caller)
		assert.Equal(t, strings.HasPrefix(seen.name, compatPrefix), seen.routed, seen.name)
	}

	assert.Equal(t, []string{`{"static":"alpha"}`}, toolResultsFor(sess.Messages, "alpha"))
	assert.Equal(t, []string{`{"static":"zeta"}`}, toolResultsFor(sess.Messages, "zeta"))

	searches := toolResultsFor(sess.Messages, search.Name)
	require.Len(t, searches, 2)
	assert.JSONEq(t, `{"caller":"alice","tool":"usermcp__search","args":{"q":"pipelines"}}`, searches[0])
	assert.JSONEq(t, `{"caller":"alice","tool":"usermcp__search","args":{"q":"again"}}`, searches[1])

	creates := toolResultsFor(sess.Messages, create.Name)
	require.Len(t, creates, 1)
	assert.JSONEq(t, `{"caller":"alice","tool":"usermcp__create","args":{"q":"ticket"}}`, creates[0])

	for _, name := range []string{"alpha", "zeta", search.Name, create.Name} {
		for _, result := range toolsResults(sess, name) {
			assert.False(t, result.IsError, name)
		}
	}

	var streamed, responded []string

	for _, event := range events {
		switch event := event.(type) {
		case agent.AssistantDeltaEvent:
			if req, ok := event.Delta.Part.(*llm.ToolRequestPart); ok {
				streamed = append(streamed, req.ID+":"+req.Name)
			}
		case agent.ToolResponseEvent:
			responded = append(responded, event.Response.ID+":"+event.Response.Name)
		}
	}

	want := []string{"s1:alpha", "u1:" + search.Name, "u2:" + create.Name, "u3:" + search.Name, "s2:zeta"}
	assert.Equal(t, want, streamed)
	assert.ElementsMatch(t, want, responded)
}

func TestInterceptorCompatConcurrentCallersKeepInjectedDefinitions(t *testing.T) {
	t.Parallel()

	const whoami = compatPrefix + "whoami"

	callers := []string{"alice", "bob"}
	shared := &stubTool{def: llm.ToolDefinition{Name: "shared", Parameters: json.RawMessage(`{"type":"object"}`)}, result: json.RawMessage(`{"static":"shared"}`)}
	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(shared))

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	modelArrived, toolArrived := make(chan string, len(callers)), make(chan string, len(callers))
	modelRelease, toolRelease := make(chan struct{}), make(chan struct{})

	var mu sync.Mutex

	visible := make(map[string]llm.ToolDefinition, len(callers))

	user := &compatUserTools{
		discover: func(_ context.Context, caller string, _ int) []llm.ToolDefinition {
			return []llm.ToolDefinition{compatDef(whoami, caller)}
		},
		execute: func(ctx context.Context, caller string, _ *agent.ToolCallInfo) (json.RawMessage, error) {
			toolArrived <- caller

			select {
			case <-toolRelease:
			case <-ctx.Done():
				return nil, ctx.Err()
			}

			return fmt.Appendf(nil, `{"caller":%q}`, caller), nil
		},
	}

	model := fakellm.NewFakeModel()
	model.When(fakellm.LastMessageHasToolResponse(whoami)).ThenRespondText("done")
	model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		caller := req.Messages[len(req.Messages)-1].TextContent()
		def, _ := compatFind(req.Tools, whoami)

		mu.Lock()
		visible[caller] = *compatCopyDefinition(&def)
		mu.Unlock()

		modelArrived <- caller

		select {
		case <-modelRelease:
		case <-ctx.Done():
			return nil, ctx.Err()
		}

		return respondWith(toolCall(caller+"-whoami", whoami, fmt.Sprintf(`{"q":%q}`, caller)), toolCall(caller+"-shared", "shared", `{}`))(req, nil)
	})

	ag, err := New("shared-agent", "prompt", model, WithTools(registry), WithInterceptors(user))
	require.NoError(t, err)

	var wg sync.WaitGroup

	sessions := make([]*session.State, len(callers))
	errs := make([]error, len(callers))

	for i, caller := range callers {
		sessions[i] = &session.State{ID: caller, Messages: userMessage(caller)}

		wg.Go(func() {
			_, errs[i] = compatRun(context.WithValue(ctx, compatCallerKey{}, caller), ag, agent.NewInvocationMetadata(sessions[i], ag.Info()))
		})
	}

	compatAwait(ctx, t, modelArrived, callers, "both callers must be inside the model call before either dispatches")
	close(modelRelease)
	compatAwait(ctx, t, toolArrived, callers, "both callers' injected tools must execute concurrently")
	close(toolRelease)
	wg.Wait()

	for i, caller := range callers {
		require.NoError(t, errs[i], caller)
		require.Contains(t, visible, caller)
		assert.Equal(t, compatDef(whoami, caller), visible[caller], "model must see only its caller's definition")

		whoamis := toolsResults(sessions[i], whoami)
		require.Len(t, whoamis, 1, caller)
		assert.False(t, whoamis[0].IsError, caller)
		assert.JSONEq(t, fmt.Sprintf(`{"caller":%q}`, caller), string(whoamis[0].Result))
		assert.Equal(t, []string{`{"static":"shared"}`}, toolResultsFor(sessions[i].Messages, "shared"), caller)
	}

	assert.Equal(t, int32(2), shared.calls.Load())

	for i, call := range model.CallsMatching(fakellm.Any()) {
		caller := call.Request.Messages[1].TextContent()
		def, count := compatFind(call.Request.Tools, whoami)
		assert.Equal(t, 1, count, "call %d", i)
		assert.Equal(t, compatDef(whoami, caller), def, "call %d", i)
		assert.Equal(t, []string{"shared", whoami}, requestToolNames(call.Request), "call %d", i)
	}

	seen := user.recorded()
	require.Len(t, seen, 4)

	for _, call := range seen {
		assert.Equal(t, call.session, call.caller, "tool %s must run under its own caller's context", call.name)

		if call.name == whoami {
			require.NotNil(t, call.definition)
			assert.Equal(t, compatDef(whoami, call.caller), *call.definition, "interceptor must see its own caller's definition")
		}
	}
}

func TestInterceptorCompatUnroutedCallsReachSnapshotLookup(t *testing.T) {
	t.Parallel()

	const (
		injected = compatPrefix + "live"
		retired  = "retired_tool"
	)

	cases := []struct {
		name       string
		static     []fixtureTool
		native     bool
		wantStatic []string
	}{
		{name: "static registry", static: []fixtureTool{{name: "alpha"}}, wantStatic: []string{"alpha"}},
		{name: "empty registry"},
		{name: "local lazy loading", static: []fixtureTool{{name: "alpha"}, {name: "dormant", deferred: true}}, wantStatic: []string{"alpha", toolloading.SearchToolName}},
		{name: "native lazy loading", static: []fixtureTool{{name: "alpha"}, {name: "dormant", deferred: true}}, native: true, wantStatic: []string{"alpha", "dormant"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			fake := fakellm.NewFakeModel()

			var model llm.Model = fake
			if tc.native {
				fake = fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{Tools: true, Streaming: true, ToolSearchProtocol: llm.ToolSearchProtocolAnthropic}))
				model = nativeSearchModel{FakeModel: fake, provider: "anthropic"}
			}

			parts := []llm.Part{toolCall("live", injected, `{"q":"x"}`), toolCall("gone", retired, `{}`)}
			if slices.Contains(tc.wantStatic, "alpha") {
				parts = append(parts, toolCall("static", "alpha", `{}`))
			}

			fake.When(fakellm.FirstCall()).ThenRespondWith(respondWith(parts...))
			fake.When(fakellm.Any()).ThenRespondText("done")

			user := &compatUserTools{
				discover: func(_ context.Context, caller string, _ int) []llm.ToolDefinition {
					return []llm.ToolDefinition{compatDef(injected, caller)}
				},
				execute: func(context.Context, string, *agent.ToolCallInfo) (json.RawMessage, error) {
					return json.RawMessage(`{"routed":true}`), nil
				},
			}
			ag, err := New("unrouted", "prompt", model, WithTools(newFixtureRegistry(t, tc.static...)), WithInterceptors(user))
			require.NoError(t, err)

			sess := &session.State{ID: "unrouted", Messages: userMessage("go")}
			events, err := compatRun(context.WithValue(t.Context(), compatCallerKey{}, "alice"), ag, agent.NewInvocationMetadata(sess, ag.Info()))
			require.NoError(t, err)
			require.Equal(t, agent.FinishReasonStop, finishReason(events))

			calls := fake.CallsMatching(fakellm.Any())
			require.Len(t, calls, 2)
			assert.Equal(t, append(slices.Clone(tc.wantStatic), injected), requestToolNames(calls[0].Request))
			assert.Equal(t, tc.native, calls[0].Request.ToolSearch)

			live := toolsResults(sess, injected)
			require.Len(t, live, 1)
			assert.False(t, live[0].IsError)
			assert.JSONEq(t, `{"routed":true}`, string(live[0].Result))

			gone := toolsResults(sess, retired)
			require.Len(t, gone, 1)
			assert.True(t, gone[0].IsError, "an unrouted unknown call must not succeed silently")
			assert.Contains(t, string(gone[0].Result), tool.ErrToolNotFound.Error())
			assert.Contains(t, string(gone[0].Result), retired)
			assert.NotContains(t, string(gone[0].Result), "tool_not_loaded")

			if slices.Contains(tc.wantStatic, "alpha") {
				static := toolsResults(sess, "alpha")
				require.Len(t, static, 1)
				assert.False(t, static[0].IsError)
			}

			seen := user.recorded()
			require.Len(t, seen, len(parts), "every call must reach the interceptor chain")

			for _, call := range seen {
				switch call.name {
				case injected:
					assert.True(t, call.routed)
					require.NotNil(t, call.definition)
					assert.Equal(t, defByName(t, calls[0].Request.Tools, injected), *call.definition)
				case retired:
					assert.False(t, call.routed)
					assert.Nil(t, call.definition, "no definition may be invented for a name nobody advertised")
				default:
					assert.False(t, call.routed)
					require.NotNil(t, call.definition)
					assert.Equal(t, defByName(t, calls[0].Request.Tools, call.name), *call.definition)
				}
			}
		})
	}
}

func TestInterceptorCompatReactiveRetryReinjectsOnce(t *testing.T) {
	t.Parallel()

	const injected = compatPrefix + "lookup"

	static := &stubTool{def: llm.ToolDefinition{Name: "alpha", Parameters: json.RawMessage(`{"type":"object"}`)}}
	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(static))

	user := &compatUserTools{
		discover: func(_ context.Context, caller string, attempt int) []llm.ToolDefinition {
			return []llm.ToolDefinition{compatDef(injected, fmt.Sprintf("%s attempt %d", caller, attempt))}
		},
		execute: func(context.Context, string, *agent.ToolCallInfo) (json.RawMessage, error) {
			return json.RawMessage(`{"routed":true}`), nil
		},
	}

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(200_000))
	model.When(fakellm.FirstCall()).ThenRespondWith(func(*llm.Request, *fakellm.CallContext) (*llm.Response, error) {
		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, &llm.ProviderError{Base: llm.ErrContextOverflow, Code: "invalid_request_error", Message: "prompt is too long"})
	})
	model.When(fakellm.CallNumber(2)).ThenRespondWithToolCall(injected, map[string]any{"q": "retry"})
	model.When(fakellm.Any()).ThenRespondText("done")

	var turns []int

	turnContext := toolsTurnInterceptor(func(ctx context.Context, info *agent.TurnInfo, next agent.TurnNext) (agent.FinishReason, error) {
		return next(context.WithValue(ctx, toolsTurnKey{}, info.Inv.Turn()), info)
	})
	ag, err := New("retry", "prompt", model, WithCompaction(CompactionConfig{}), WithInterceptors(turnContext, user),
		WithTools(toolsContextRegistry{snapshot: func(ctx context.Context) (tool.Registry, error) {
			turns = append(turns, toolsContextTurn(ctx, t))

			return tool.Snapshot(ctx, registry)
		}}),
	)
	require.NoError(t, err)

	sess := &session.State{ID: "retry", Messages: userMessage("start")}

	for i := range 4 {
		id := fmt.Sprintf("old_%d", i)
		sess.Messages = append(sess.Messages,
			llm.NewMessage(llm.RoleAssistant, toolCall(id, "alpha", `{}`)),
			llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(id, "alpha", fmt.Appendf(nil, `{"data":%q}`, strings.Repeat("x", 16_000)), false)),
		)
	}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")))

	events, err := compatRun(context.WithValue(t.Context(), compatCallerKey{}, "alice"), ag, agent.NewInvocationMetadata(sess, ag.Info()))
	require.NoError(t, err)
	require.Equal(t, agent.FinishReasonStop, finishReason(events))
	assert.Equal(t, []int{0, 1}, turns, "the retry must reuse the step's snapshot")
	assert.True(t, slices.ContainsFunc(events, func(event agent.Event) bool {
		compacted, ok := event.(agent.CompactionEvent)

		return ok && compacted.Report.Phase == agent.CompactionPhaseReactive
	}))

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 3)

	for i, call := range calls {
		def, count := compatFind(call.Request.Tools, injected)
		assert.Equal(t, 1, count, "call %d must carry the injected tool exactly once", i)
		assert.Equal(t, compatDef(injected, fmt.Sprintf("alice attempt %d", i+1)), def, "call %d", i)
		assert.Equal(t, []string{"alpha", injected}, requestToolNames(call.Request), "call %d", i)
	}

	assert.Less(t, calls[1].InputTokens, calls[0].InputTokens)

	seen := user.recorded()
	require.Len(t, seen, 1)
	require.NotNil(t, seen[0].definition)
	assert.Equal(t, defByName(t, calls[1].Request.Tools, injected), *seen[0].definition, "the definition must come from the attempt the model answered")

	results := toolsResults(sess, injected)
	require.Len(t, results, 1)
	assert.False(t, results[0].IsError)
	assert.Zero(t, static.calls.Load())
}
