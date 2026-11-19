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
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const wrapperCompatAdmin = "admin"

// wrapperCompatTool records the caller of every execution from its call context.
type wrapperCompatTool struct {
	def    llm.ToolDefinition
	result json.RawMessage
	gate   func(context.Context) error

	mu      sync.Mutex
	callers []string
}

func (w *wrapperCompatTool) Definition() llm.ToolDefinition { return w.def }

func (w *wrapperCompatTool) Execute(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
	w.mu.Lock()
	w.callers = append(w.callers, compatCaller(ctx))
	w.mu.Unlock()

	if w.gate != nil {
		if err := w.gate(ctx); err != nil {
			return nil, err
		}
	}

	return w.result, nil
}

func (w *wrapperCompatTool) ranFor(caller string) int {
	w.mu.Lock()
	defer w.mu.Unlock()

	runs := 0

	for _, seen := range w.callers {
		if seen == caller {
			runs++
		}
	}

	return runs
}

func wrapperCompatNewTool(name, description, result string) *wrapperCompatTool {
	return &wrapperCompatTool{
		def:    llm.ToolDefinition{Name: name, Description: description, Parameters: json.RawMessage(`{"type":"object"}`)},
		result: json.RawMessage(result),
	}
}

// wrapperCompatLegacyDeny was written before registries had snapshots: it
// embeds a registry and overrides only Execute, whose signature never changed.
type wrapperCompatLegacyDeny struct {
	tool.Registry

	denied atomic.Int32
}

func (r *wrapperCompatLegacyDeny) Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	if req.Name == "secret" {
		r.denied.Add(1)

		return llm.NewToolResponsePart(req.ID, req.Name, json.RawMessage(`{"error":"denied by wrapper"}`), true), nil
	}

	return r.Registry.Execute(ctx, req)
}

func TestToolsWrapperCompatLegacyExecuteOverrideGatesDispatch(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		beside bool
	}{
		{name: "alone"},
		{name: "beside native source", beside: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			secret := wrapperCompatNewTool("secret", "secret", `{"secret":"leaked"}`)
			public := wrapperCompatNewTool("public", "public", `{"public":true}`)
			native := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, native.Register(secret))
			require.NoError(t, native.Register(public))

			wrapper := &wrapperCompatLegacyDeny{Registry: native}
			sources := []tool.Registry{wrapper}
			parts := []llm.Part{toolCall("s", "secret", `{}`), toolCall("p", "public", `{}`)}
			names := []string{"public", "secret"}

			other := wrapperCompatNewTool("other", "other", `{"other":true}`)

			if tc.beside {
				next := tool.NewRegistry(tool.RegistryConfig{})
				require.NoError(t, next.Register(other))

				sources = append(sources, next)
				parts = append(parts, toolCall("o", "other", `{}`))
				names = []string{"other", "public", "secret"}
			}

			model := fakellm.NewFakeModel()
			model.When(fakellm.FirstCall()).ThenRespondWith(respondWith(parts...))
			model.When(fakellm.Any()).ThenRespondText("done")

			ag, err := New("legacy-auth", "prompt", model, WithTools(sources...))
			require.NoError(t, err)

			sess := &session.State{ID: "legacy-auth", Messages: userMessage("read the secret")}
			require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))

			assert.Zero(t, secret.ranFor(""), "the wrapper's Execute denial must prevent execution")
			assert.Equal(t, int32(1), wrapper.denied.Load(), "dispatch must go through the wrapper's Execute")

			results := toolsResults(sess, "secret")
			require.Len(t, results, 1)
			assert.True(t, results[0].IsError)
			assert.JSONEq(t, `{"error":"denied by wrapper"}`, string(results[0].Result))

			assert.Equal(t, 1, public.ranFor(""))
			assert.Equal(t, []string{`{"public":true}`}, toolResultsFor(sess.Messages, "public"))

			if tc.beside {
				assert.Equal(t, 1, other.ranFor(""))
				assert.Equal(t, []string{`{"other":true}`}, toolResultsFor(sess.Messages, "other"))
			}

			calls := model.CallsMatching(fakellm.Any())
			require.Len(t, calls, 2)

			for _, call := range calls {
				assert.Equal(t, names, requestToolNames(call.Request))
			}
		})
	}
}

type wrapperCompatRecord struct {
	tool           string
	caller         string
	conversation   string
	captured       bool
	snapshotCaller string
}

// wrapperCompatPolicy is one decorator's behavior. decide returns a denial
// result, or nil to pass the call to the inner registry.
type wrapperCompatPolicy struct {
	name   string
	decide func(context.Context, *llm.ToolRequestPart) json.RawMessage

	snapshots atomic.Int32

	mu      sync.Mutex
	records []wrapperCompatRecord
}

func (p *wrapperCompatPolicy) execute(ctx context.Context, inner tool.Registry, req *llm.ToolRequestPart, captured bool, snapshotCaller string) (*llm.ToolResponsePart, error) {
	p.mu.Lock()

	p.records = append(p.records, wrapperCompatRecord{
		tool: req.Name, caller: compatCaller(ctx), conversation: agent.ConversationIDFromContext(ctx),
		captured: captured, snapshotCaller: snapshotCaller,
	})
	p.mu.Unlock()

	if denial := p.decide(ctx, req); denial != nil {
		return llm.NewToolResponsePart(req.ID, req.Name, denial, true), nil
	}

	return inner.Execute(ctx, req)
}

func (p *wrapperCompatPolicy) recorded() []wrapperCompatRecord {
	p.mu.Lock()
	defer p.mu.Unlock()

	return slices.Clone(p.records)
}

// wrapperCompatLegacy applies a policy by overriding only Execute.
type wrapperCompatLegacy struct {
	tool.Registry

	policy *wrapperCompatPolicy
}

func (w wrapperCompatLegacy) Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	return w.policy.execute(ctx, w.Registry, req, false, "")
}

// wrapperCompatExplicit applies the same policy and snapshots by capturing its
// inner registry explicitly and wrapping the capture.
type wrapperCompatExplicit struct {
	tool.Registry

	policy         *wrapperCompatPolicy
	captured       bool
	snapshotCaller string
}

func (w wrapperCompatExplicit) Snapshot(ctx context.Context) (tool.Registry, error) {
	inner, err := tool.Snapshot(ctx, w.Registry)
	if err != nil {
		return nil, err
	}

	w.policy.snapshots.Add(1)

	return wrapperCompatExplicit{Registry: inner, policy: w.policy, captured: true, snapshotCaller: compatCaller(ctx)}, nil
}

func (w wrapperCompatExplicit) Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	return w.policy.execute(ctx, w.Registry, req, w.captured, w.snapshotCaller)
}

type wrapperCompatLayer struct {
	name     string
	explicit bool
}

func wrapperCompatPolicies() map[string]*wrapperCompatPolicy {
	var (
		mu      sync.Mutex
		metered = make(map[string]int)
	)

	return map[string]*wrapperCompatPolicy{
		"auth": {name: "auth", decide: func(ctx context.Context, req *llm.ToolRequestPart) json.RawMessage {
			if req.Name == "secret" && compatCaller(ctx) != wrapperCompatAdmin {
				return json.RawMessage(`{"error":"denied by auth"}`)
			}

			return nil
		}},
		"log": {name: "log", decide: func(context.Context, *llm.ToolRequestPart) json.RawMessage { return nil }},
		"rate": {name: "rate", decide: func(ctx context.Context, req *llm.ToolRequestPart) json.RawMessage {
			if req.Name != "metered" {
				return nil
			}

			mu.Lock()
			defer mu.Unlock()

			if metered[compatCaller(ctx)] > 0 {
				return json.RawMessage(`{"error":"rate limited"}`)
			}

			metered[compatCaller(ctx)]++

			return nil
		}},
	}
}

// wrapperCompatWant is how often the layer at pos sees each tool for caller:
// auth stops the secret for everyone but admin, rate stops the second metered call.
func wrapperCompatWant(layers []wrapperCompatLayer, pos int, caller string) map[string]int {
	want := map[string]int{"secret": 1, "public": 2, "metered": 2}

	if pos > slices.IndexFunc(layers, func(l wrapperCompatLayer) bool { return l.name == "auth" }) && caller != wrapperCompatAdmin {
		delete(want, "secret")
	}

	if pos > slices.IndexFunc(layers, func(l wrapperCompatLayer) bool { return l.name == "rate" }) {
		want["metered"] = 1
	}

	return want
}

func TestToolsWrapperCompatStackedPolicyLayers(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		layers []wrapperCompatLayer
	}{
		{name: "explicit chain", layers: []wrapperCompatLayer{{"auth", true}, {"log", true}, {"rate", true}}},
		{name: "legacy between explicit", layers: []wrapperCompatLayer{{"auth", true}, {"log", false}, {"rate", true}}},
		{name: "legacy outermost", layers: []wrapperCompatLayer{{"log", false}, {"auth", true}, {"rate", true}}},
		{name: "legacy innermost", layers: []wrapperCompatLayer{{"auth", true}, {"rate", true}, {"log", false}}},
		{name: "legacy chain", layers: []wrapperCompatLayer{{"auth", false}, {"log", false}, {"rate", false}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			wrapperCompatRunStack(t, tc.layers)
		})
	}
}

func wrapperCompatRunStack(t *testing.T, layers []wrapperCompatLayer) {
	t.Helper()

	callers := []string{"alice", wrapperCompatAdmin}

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	modelArrived, meteredArrived := make(chan string, 2*len(callers)), make(chan string, 4*len(callers))
	modelRelease, meteredRelease := make(chan struct{}), make(chan struct{})

	secret := wrapperCompatNewTool("secret", "secret", `{"secret":"value"}`)
	publicV1 := wrapperCompatNewTool("public", "public v1", `{"public":1}`)
	publicV2 := wrapperCompatNewTool("public", "public v2", `{"public":2}`)
	metered := wrapperCompatNewTool("metered", "metered", `{"metered":true}`)
	metered.gate = func(ctx context.Context) error {
		meteredArrived <- compatCaller(ctx)

		select {
		case <-meteredRelease:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	native := tool.NewRegistry(tool.RegistryConfig{})
	for _, registered := range []tool.Tool{secret, publicV1, metered} {
		require.NoError(t, native.Register(registered))
	}

	policies := wrapperCompatPolicies()

	var stack tool.Registry = native

	for _, layer := range slices.Backward(layers) {
		if layer.explicit {
			stack = wrapperCompatExplicit{Registry: stack, policy: policies[layer.name]}
		} else {
			stack = wrapperCompatLegacy{Registry: stack, policy: policies[layer.name]}
		}
	}

	model := fakellm.NewFakeModel()
	model.When(fakellm.LastMessageHasToolResponse("metered")).ThenRespondWith(respondWith(toolCall("public-2", "public", `{}`)))
	model.When(fakellm.LastMessageHasToolResponse("public")).ThenRespondText("done")
	model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		modelArrived <- req.Messages[len(req.Messages)-1].TextContent()

		select {
		case <-modelRelease:
		case <-ctx.Done():
			return nil, ctx.Err()
		}

		return respondWith(
			toolCall("secret", "secret", `{}`), toolCall("public-1", "public", `{}`),
			toolCall("metered-a", "metered", `{}`), toolCall("metered-b", "metered", `{}`),
		)(req, nil)
	})

	ag, err := New("stacked", "prompt", model, WithTools(stack))
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

	compatAwait(ctx, t, modelArrived, callers, "both callers must capture their tools before the source changes")
	require.NoError(t, native.Unregister("public"))
	require.NoError(t, native.Register(publicV2))
	close(modelRelease)
	compatAwait(ctx, t, meteredArrived, callers, "each caller's one admitted metered call must run concurrently")
	close(meteredRelease)
	wg.Wait()

	firstLegacy := slices.IndexFunc(layers, func(l wrapperCompatLayer) bool { return !l.explicit })
	if firstLegacy < 0 {
		firstLegacy = len(layers)
	}

	for i, caller := range callers {
		require.NoError(t, errs[i], caller)
		wrapperCompatAssertResults(t, sessions[i], caller)

		if firstLegacy == len(layers) {
			assert.Equal(t, 1, publicV1.ranFor(caller), "an explicit chain must dispatch the captured tool, %s", caller)
			assert.Equal(t, 1, publicV2.ranFor(caller), caller)
			assert.Equal(t, []string{`{"public":1}`, `{"public":2}`}, toolResultsFor(sessions[i].Messages, "public"), caller)
		} else {
			assert.Equal(t, 2, publicV1.ranFor(caller)+publicV2.ranFor(caller), caller)
			assert.Equal(t, `{"public":2}`, toolResultsFor(sessions[i].Messages, "public")[1], caller)
		}

		wantSecret := 0
		if caller == wrapperCompatAdmin {
			wantSecret = 1
		}

		assert.Equal(t, wantSecret, secret.ranFor(caller), "secret for %s", caller)
		assert.Equal(t, 1, metered.ranFor(caller), caller)

		var steps []fakellm.Call

		for _, call := range model.CallsMatching(fakellm.Any()) {
			if call.Request.Messages[1].TextContent() == caller {
				steps = append(steps, call)
			}
		}

		require.Len(t, steps, 3, caller)
		assert.Equal(t, "public v1", defByName(t, steps[0].Request.Tools, "public").Description, caller)
		assert.Equal(t, "public v2", defByName(t, steps[1].Request.Tools, "public").Description, caller)

		for _, step := range steps {
			assert.Equal(t, []string{"metered", "public", "secret"}, requestToolNames(step.Request), caller)
		}
	}

	for pos, layer := range layers {
		policy := policies[layer.name]
		records := policy.recorded()

		for _, caller := range callers {
			seen := make(map[string]int)

			for _, record := range records {
				if record.caller == caller {
					seen[record.tool]++
				}
			}

			assert.Equal(t, wrapperCompatWant(layers, pos, caller), seen, "%s layer must apply for %s", layer.name, caller)
		}

		for _, record := range records {
			assert.Contains(t, callers, record.caller, layer.name)
			assert.Equal(t, record.caller, record.conversation, "%s must receive the call's own context", layer.name)

			if pos < firstLegacy {
				assert.True(t, record.captured, "%s must dispatch through its capture", layer.name)
				assert.Equal(t, record.caller, record.snapshotCaller, "%s must be captured from its caller's step", layer.name)
			}
		}

		if pos < firstLegacy {
			assert.Equal(t, 3*len(callers), int(policy.snapshots.Load()), "%s must be captured once per step", layer.name)
		}
	}
}

func wrapperCompatAssertResults(t *testing.T, sess *session.State, caller string) {
	t.Helper()

	secret := toolsResults(sess, "secret")
	require.Len(t, secret, 1, caller)

	if caller == wrapperCompatAdmin {
		assert.False(t, secret[0].IsError)
		assert.JSONEq(t, `{"secret":"value"}`, string(secret[0].Result))
	} else {
		assert.True(t, secret[0].IsError)
		assert.JSONEq(t, `{"error":"denied by auth"}`, string(secret[0].Result))
	}

	assert.ElementsMatch(t, []string{`{"metered":true}`, `{"error":"rate limited"}`}, toolResultsFor(sess.Messages, "metered"), caller)

	public := toolsResults(sess, "public")
	require.Len(t, public, 2, caller)

	for _, result := range public {
		assert.False(t, result.IsError, caller)
	}
}
