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

package tool_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

func compatTool(name, result string, calls *atomic.Int32) *snapshotProbe {
	return &snapshotProbe{
		def: llm.ToolDefinition{Name: name, Description: result, Parameters: json.RawMessage(`{"type":"object"}`)},
		run: func(context.Context, json.RawMessage) (json.RawMessage, error) {
			calls.Add(1)

			return json.RawMessage(strconv.Quote(result)), nil
		},
	}
}

func compatNative(t *testing.T, names ...string) (tool.Registry, map[string]*atomic.Int32) {
	t.Helper()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	calls := make(map[string]*atomic.Int32, len(names))

	for _, name := range names {
		calls[name] = new(atomic.Int32)
		require.NoError(t, registry.Register(compatTool(name, name, calls[name])))
	}

	return registry, calls
}

func compatDispatch(t *testing.T, registry tool.Registry, batch bool, names ...string) map[string]*llm.ToolResponsePart {
	t.Helper()

	reqs := make([]*llm.ToolRequestPart, len(names))
	for i, name := range names {
		reqs[i] = &llm.ToolRequestPart{ID: "call-" + name, Name: name, Arguments: json.RawMessage(`{}`)}
	}

	var resps []*llm.ToolResponsePart

	if batch {
		resps = registry.ExecuteAll(t.Context(), reqs)
	} else {
		for _, req := range reqs {
			resp, err := registry.Execute(t.Context(), req)
			require.NoError(t, err)

			resps = append(resps, resp)
		}
	}

	require.Len(t, resps, len(names))

	byName := make(map[string]*llm.ToolResponsePart, len(resps))
	for _, resp := range resps {
		require.NotNil(t, resp)
		assert.Equal(t, "call-"+resp.Name, resp.ID)
		byName[resp.Name] = resp
	}

	require.Len(t, byName, len(names))

	return byName
}

// compatPolicy is an Execute gate shared by a legacy wrapper and its migrated form.
type compatPolicy struct {
	deny      string
	denied    atomic.Int32
	live      atomic.Int32
	snapshots atomic.Int32
}

func (p *compatPolicy) execute(ctx context.Context, next tool.Registry, captured bool, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	if !captured {
		p.live.Add(1)
	}

	if req.Name == p.deny {
		p.denied.Add(1)

		return llm.NewToolResponsePart(req.ID, req.Name, json.RawMessage(`{"error":"denied by gate"}`), true), nil
	}

	return next.Execute(ctx, req)
}

// compatGate is Execute middleware written against the base Registry: it
// embeds the interface and overrides only Execute.
type compatGate struct {
	tool.Registry

	policy *compatPolicy
}

func (g *compatGate) Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	return g.policy.execute(ctx, g.Registry, false, req)
}

// compatCapturingGate is compatGate migrated to Snapshot: it captures what it
// wraps and applies its policy to the capture.
type compatCapturingGate struct {
	tool.Registry

	policy   *compatPolicy
	captured bool
}

func (g compatCapturingGate) Snapshot(ctx context.Context) (tool.Registry, error) {
	g.policy.snapshots.Add(1)

	inner, err := tool.Snapshot(ctx, g.Registry)
	if err != nil {
		return nil, err
	}

	g.Registry, g.captured = inner, true

	return g, nil
}

func (g compatCapturingGate) Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	return g.policy.execute(ctx, g.Registry, g.captured, req)
}

func TestSnapshotLegacyWrapperKeepsExecuteGate(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		build func(t *testing.T, native tool.Registry, gates ...*compatPolicy) []tool.Registry
		gates []string
	}{
		{name: "direct", gates: []string{"secret"}, build: func(_ *testing.T, native tool.Registry, gates ...*compatPolicy) []tool.Registry {
			return []tool.Registry{&compatGate{Registry: native, policy: gates[0]}}
		}},
		{name: "nested", gates: []string{"secret", "admin"}, build: func(_ *testing.T, native tool.Registry, gates ...*compatPolicy) []tool.Registry {
			return []tool.Registry{&compatGate{Registry: &compatGate{Registry: native, policy: gates[0]}, policy: gates[1]}}
		}},
		{name: "beside native and nil", gates: []string{"secret"}, build: func(t *testing.T, native tool.Registry, gates ...*compatPolicy) []tool.Registry {
			t.Helper()

			other, _ := compatNative(t, "other")

			return []tool.Registry{nil, other, &compatGate{Registry: native, policy: gates[0]}, nil}
		}},
		{name: "captured twice", gates: []string{"secret"}, build: func(t *testing.T, native tool.Registry, gates ...*compatPolicy) []tool.Registry {
			t.Helper()

			first, err := tool.Snapshot(t.Context(), &compatGate{Registry: native, policy: gates[0]})
			require.NoError(t, err)

			return []tool.Registry{first}
		}},
	}

	for _, mode := range []struct {
		name  string
		batch bool
	}{{name: "Execute"}, {name: "ExecuteAll", batch: true}} {
		for _, tc := range cases {
			t.Run(mode.name+"/"+tc.name, func(t *testing.T) {
				t.Parallel()

				native, calls := compatNative(t, "secret", "admin", "public")
				policies := make([]*compatPolicy, len(tc.gates))

				for i, deny := range tc.gates {
					policies[i] = &compatPolicy{deny: deny}
				}

				frozen, err := tool.Snapshot(t.Context(), tc.build(t, native, policies...)...)
				require.NoError(t, err)

				resps := compatDispatch(t, frozen, mode.batch, "secret", "admin", "public")

				for _, policy := range policies {
					assert.EqualValues(t, 1, policy.denied.Load(), "gate %q must run", policy.deny)
					assert.True(t, resps[policy.deny].IsError)
					assert.JSONEq(t, `{"error":"denied by gate"}`, string(resps[policy.deny].Result))
					assert.Zero(t, calls[policy.deny].Load(), "%q must not execute past its gate", policy.deny)
				}

				assert.False(t, resps["public"].IsError)
				assert.JSONEq(t, `"public"`, string(resps["public"].Result))
				assert.EqualValues(t, 1, calls["public"].Load())
			})
		}
	}
}

// compatVisibility hides a tool from discovery, as a per-caller tool filter does.
type compatVisibility struct {
	tool.Registry

	hidden string
}

func (v compatVisibility) List(ctx context.Context) ([]llm.ToolDefinition, error) {
	defs, err := v.Registry.List(ctx)

	return slices.DeleteFunc(defs, func(def llm.ToolDefinition) bool { return def.Name == v.hidden }), err
}

func (v compatVisibility) Get(ctx context.Context, name string) (tool.Tool, error) {
	if name == v.hidden {
		return nil, tool.ErrToolNotFound
	}

	return v.Registry.Get(ctx, name)
}

func TestSnapshotLegacyWrapperKeepsDiscoveryFilter(t *testing.T) {
	t.Parallel()

	native, calls := compatNative(t, "secret", "public")
	frozen, err := tool.Snapshot(t.Context(), compatVisibility{Registry: native, hidden: "secret"})
	require.NoError(t, err)

	defs, err := frozen.List(t.Context())
	require.NoError(t, err)
	assert.Equal(t, []string{"public"}, definitionNames(defs))
	_, err = frozen.Get(t.Context(), "secret")
	require.ErrorIs(t, err, tool.ErrToolNotFound)

	resps := compatDispatch(t, frozen, false, "secret", "public")
	assert.True(t, resps["secret"].IsError)
	assert.Contains(t, string(resps["secret"].Result), tool.ErrToolNotFound.Error())
	assert.Zero(t, calls["secret"].Load())
	assert.False(t, resps["public"].IsError)
}

// TestSnapshotLegacyDispatchIsLive pins what a source without Snapshot gets:
// the returned registry fixes its definitions and membership, while each call
// goes through the source's current Execute, which decides what runs.
func TestSnapshotLegacyDispatchIsLive(t *testing.T) {
	t.Parallel()

	native, calls := compatNative(t, "read")
	policy := &compatPolicy{deny: "secret"}
	frozen, err := tool.Snapshot(t.Context(), &compatGate{Registry: native, policy: policy})
	require.NoError(t, err)

	replacement := new(atomic.Int32)

	require.NoError(t, native.Unregister("read"))
	require.NoError(t, native.Register(compatTool("read", "replacement", replacement), tool.WithDeferred()))
	require.NoError(t, native.Register(compatTool("added", "added", new(atomic.Int32))))

	defs, err := frozen.List(t.Context())
	require.NoError(t, err)
	require.Equal(t, []string{"read"}, definitionNames(defs))
	assert.Equal(t, "read", defs[0].Description)
	assert.False(t, defs[0].Deferred)

	captured, err := frozen.Get(t.Context(), "read")
	require.NoError(t, err)
	assert.Equal(t, "read", captured.Definition().Description)
	_, err = frozen.Get(t.Context(), "added")
	require.ErrorIs(t, err, tool.ErrToolNotFound)

	result, err := captured.Execute(t.Context(), json.RawMessage(`{}`))
	require.NoError(t, err)
	assert.JSONEq(t, `"replacement"`, string(result), "a tool from Get executes through the source")

	resps := compatDispatch(t, frozen, false, "added", "read")
	assert.True(t, resps["added"].IsError, "membership is fixed at capture")
	assert.Contains(t, string(resps["added"].Result), tool.ErrToolNotFound.Error())
	assert.JSONEq(t, `"replacement"`, string(resps["read"].Result), "the source resolves the tool at call time")
	assert.Zero(t, calls["read"].Load())
	assert.EqualValues(t, 2, replacement.Load())

	require.NoError(t, native.Unregister("read"))

	resps = compatDispatch(t, frozen, false, "read")
	assert.True(t, resps["read"].IsError)
	assert.Contains(t, string(resps["read"].Result), tool.ErrToolNotFound.Error())

	_, err = captured.Execute(t.Context(), json.RawMessage(`{}`))

	var resultErr *tool.ResultError
	require.ErrorAs(t, err, &resultErr)
	assert.Contains(t, string(resultErr.Result), tool.ErrToolNotFound.Error())
	assert.EqualValues(t, 4, policy.live.Load(), "each call reaches the source's Execute once")
}

// TestSnapshotLegacyCaptureListsOnly pins that a source without Snapshot is
// captured from List alone: a tool removed before Get, or a Get stricter than
// Execute, does not fail the capture, and lookups execute through the source.
func TestSnapshotLegacyCaptureListsOnly(t *testing.T) {
	t.Parallel()

	t.Run("removed after List", func(t *testing.T) {
		t.Parallel()

		native, calls := compatNative(t, "read")
		policy := &compatPolicy{deny: "secret"}
		reloading := compatDiscovery{Registry: native, list: func(ctx context.Context) ([]llm.ToolDefinition, error) {
			defs, err := native.List(ctx)
			if err == nil {
				err = native.Unregister("read")
			}

			return defs, err
		}}

		frozen, err := tool.Snapshot(t.Context(), &compatGate{Registry: reloading, policy: policy})
		require.NoError(t, err)

		defs, err := frozen.List(t.Context())
		require.NoError(t, err)
		assert.Equal(t, []string{"read"}, definitionNames(defs))

		resps := compatDispatch(t, frozen, false, "read")
		assert.True(t, resps["read"].IsError)
		assert.Contains(t, string(resps["read"].Result), tool.ErrToolNotFound.Error())
		assert.EqualValues(t, 1, policy.live.Load())
		assert.Zero(t, calls["read"].Load())
	})

	t.Run("Get stricter than Execute", func(t *testing.T) {
		t.Parallel()

		native, calls := compatNative(t, "secret", "public")
		policy := &compatPolicy{deny: "secret"}
		strict := compatDiscovery{Registry: native, get: func(context.Context, string) (tool.Tool, error) {
			return nil, errors.New("lookup refused")
		}}

		frozen, err := tool.Snapshot(t.Context(), &compatGate{Registry: strict, policy: policy})
		require.NoError(t, err)

		secret, err := frozen.Get(t.Context(), "secret")
		require.NoError(t, err)
		assert.Equal(t, "secret", secret.Definition().Description)

		_, err = secret.Execute(t.Context(), json.RawMessage(`{}`))

		var resultErr *tool.ResultError
		require.ErrorAs(t, err, &resultErr)
		assert.JSONEq(t, `{"error":"denied by gate"}`, string(resultErr.Result))

		rewrapped := tool.NewRegistry(tool.RegistryConfig{})
		require.NoError(t, rewrapped.Register(secret))
		resps := compatDispatch(t, rewrapped, false, "secret")
		assert.True(t, resps["secret"].IsError)
		assert.JSONEq(t, `{"error":"denied by gate"}`, string(resps["secret"].Result))

		public, err := frozen.Get(t.Context(), "public")
		require.NoError(t, err)
		result, err := public.Execute(t.Context(), json.RawMessage(`{}`))
		require.NoError(t, err)
		assert.JSONEq(t, `"public"`, string(result))

		assert.Zero(t, calls["secret"].Load())
		assert.EqualValues(t, 1, calls["public"].Load())
		assert.EqualValues(t, 2, policy.denied.Load())
		assert.EqualValues(t, 3, policy.live.Load())
	})
}

func TestSnapshotWrapperChains(t *testing.T) {
	t.Parallel()

	// Layers are listed innermost first; layer 0 denies secret, layer 1 admin.
	cases := []struct {
		name          string
		explicit      []bool
		wantFrozen    bool
		wantSnapshots []int32
		wantLive      []bool
	}{
		{name: "explicit", explicit: []bool{true}, wantFrozen: true, wantSnapshots: []int32{1}, wantLive: []bool{false}},
		{name: "legacy", explicit: []bool{false}, wantSnapshots: []int32{0}, wantLive: []bool{true}},
		{name: "explicit over explicit", explicit: []bool{true, true}, wantFrozen: true, wantSnapshots: []int32{1, 1}, wantLive: []bool{false, false}},
		{name: "explicit over legacy", explicit: []bool{false, true}, wantSnapshots: []int32{0, 1}, wantLive: []bool{true, false}},
		{name: "legacy over explicit", explicit: []bool{true, false}, wantSnapshots: []int32{0, 0}, wantLive: []bool{true, true}},
		{name: "legacy over legacy", explicit: []bool{false, false}, wantSnapshots: []int32{0, 0}, wantLive: []bool{true, true}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			native, calls := compatNative(t, "secret", "admin", "read")
			source := native
			policies := make([]*compatPolicy, len(tc.explicit))

			for i, explicit := range tc.explicit {
				policies[i] = &compatPolicy{deny: []string{"secret", "admin"}[i]}
				if explicit {
					source = compatCapturingGate{Registry: source, policy: policies[i]}
				} else {
					source = &compatGate{Registry: source, policy: policies[i]}
				}
			}

			frozen, err := tool.Snapshot(t.Context(), source)
			require.NoError(t, err)

			replacement := new(atomic.Int32)

			require.NoError(t, native.Unregister("read"))
			require.NoError(t, native.Register(compatTool("read", "replacement", replacement)))

			resps := compatDispatch(t, frozen, false, "secret", "admin", "read")

			for i, policy := range policies {
				assert.EqualValues(t, 1, policy.denied.Load(), "layer %d gate must run", i)
				assert.Zero(t, calls[policy.deny].Load())
				assert.Equal(t, tc.wantSnapshots[i], policy.snapshots.Load(), "layer %d Snapshot calls", i)
				assert.Equal(t, tc.wantLive[i], policy.live.Load() > 0, "layer %d dispatched uncaptured", i)
			}

			if len(policies) == 1 {
				assert.EqualValues(t, 1, calls["admin"].Load())
			}

			if tc.wantFrozen {
				assert.JSONEq(t, `"read"`, string(resps["read"].Result))
				assert.EqualValues(t, 1, calls["read"].Load())
				assert.Zero(t, replacement.Load())
			} else {
				assert.JSONEq(t, `"replacement"`, string(resps["read"].Result))
				assert.Zero(t, calls["read"].Load())
				assert.EqualValues(t, 1, replacement.Load())
			}
		})
	}
}

type compatDiscovery struct {
	tool.Registry

	list func(context.Context) ([]llm.ToolDefinition, error)
	get  func(context.Context, string) (tool.Tool, error)
}

func (d compatDiscovery) List(ctx context.Context) ([]llm.ToolDefinition, error) {
	if d.list == nil {
		return d.Registry.List(ctx)
	}

	return d.list(ctx)
}

func (d compatDiscovery) Get(ctx context.Context, name string) (tool.Tool, error) {
	if d.get == nil {
		return d.Registry.Get(ctx, name)
	}

	return d.get(ctx, name)
}

func TestSnapshotLegacyDiscoveryErrors(t *testing.T) {
	t.Parallel()

	failure := errors.New("discovery failed")
	probe := &mockTool{name: "probe"}
	defs := func(defs ...llm.ToolDefinition) func(context.Context) ([]llm.ToolDefinition, error) {
		return func(context.Context) ([]llm.ToolDefinition, error) { return defs, nil }
	}

	cases := []struct {
		name     string
		canceled bool
		inputs   func(t *testing.T, native tool.Registry, cancel context.CancelFunc) []tool.Registry
		wantErr  error
	}{
		{name: "List", wantErr: failure, inputs: func(_ *testing.T, native tool.Registry, _ context.CancelFunc) []tool.Registry {
			return []tool.Registry{&failingDiscoveryRegistry{Registry: native, err: failure}}
		}},
		{name: "explicit capture Get", wantErr: failure, inputs: func(_ *testing.T, native tool.Registry, _ context.CancelFunc) []tool.Registry {
			return []tool.Registry{&customSnapshotRegistry{Registry: native, captured: &failingDiscoveryRegistry{Registry: native, err: failure, getErr: true}}}
		}},
		{name: "canceled before capture", canceled: true, wantErr: context.Canceled, inputs: func(_ *testing.T, native tool.Registry, _ context.CancelFunc) []tool.Registry {
			return []tool.Registry{compatDiscovery{Registry: native}}
		}},
		{name: "canceled during List", wantErr: context.Canceled, inputs: func(_ *testing.T, native tool.Registry, cancel context.CancelFunc) []tool.Registry {
			return []tool.Registry{compatDiscovery{Registry: native, list: func(ctx context.Context) ([]llm.ToolDefinition, error) {
				cancel()

				return native.List(context.WithoutCancel(ctx))
			}}}
		}},
		{name: "duplicate in one input", wantErr: tool.ErrToolAlreadyRegistered, inputs: func(_ *testing.T, native tool.Registry, _ context.CancelFunc) []tool.Registry {
			return []tool.Registry{compatDiscovery{Registry: native, list: defs(probe.Definition(), probe.Definition())}}
		}},
		{name: "duplicate across inputs", wantErr: tool.ErrToolAlreadyRegistered, inputs: func(_ *testing.T, native tool.Registry, _ context.CancelFunc) []tool.Registry {
			return []tool.Registry{native, compatDiscovery{Registry: native}}
		}},
		{name: "empty name", wantErr: tool.ErrToolNameEmpty, inputs: func(_ *testing.T, native tool.Registry, _ context.CancelFunc) []tool.Registry {
			return []tool.Registry{compatDiscovery{Registry: native, list: defs(llm.ToolDefinition{})}}
		}},
		{name: "group conflict", wantErr: tool.ErrInvalidToolConfig, inputs: func(t *testing.T, native tool.Registry, _ context.CancelFunc) []tool.Registry {
			t.Helper()

			grouped := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, grouped.Register(&mockTool{name: "grouped"}, tool.WithGroup(llm.ToolGroup{Name: "g", Description: "one"})))

			return []tool.Registry{grouped, compatDiscovery{
				Registry: native, list: defs(llm.ToolDefinition{Name: "probe", Group: llm.ToolGroup{Name: "g", Description: "two"}}),
			}}
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			native := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, native.Register(probe))

			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()

			if tc.canceled {
				cancel()
			}

			captured, err := tool.Snapshot(ctx, tc.inputs(t, native, cancel)...)
			require.ErrorIs(t, err, tc.wantErr)
			assert.Nil(t, captured)
		})
	}
}

// compatPlainRegistry implements Registry by hand and has no Snapshot method.
type compatPlainRegistry struct {
	tools   map[string]tool.Tool
	calls   atomic.Int32
	callCtx atomic.Value
}

type compatCallKey struct{}

func (*compatPlainRegistry) Register(tool.Tool, ...tool.Option) error {
	return tool.ErrRegistryReadOnly
}

func (*compatPlainRegistry) Unregister(string) error { return tool.ErrRegistryReadOnly }

func (r *compatPlainRegistry) List(ctx context.Context) ([]llm.ToolDefinition, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	defs := make([]llm.ToolDefinition, 0, len(r.tools))
	for _, t := range r.tools {
		def := t.Definition()
		def.Deferred = true
		def.Group = llm.ToolGroup{Name: "plain", Description: "Plain tools"}
		defs = append(defs, def)
	}

	return defs, nil
}

func (r *compatPlainRegistry) Get(_ context.Context, name string) (tool.Tool, error) {
	t, ok := r.tools[name]
	if !ok {
		return nil, tool.ErrToolNotFound
	}

	return t, nil
}

func (r *compatPlainRegistry) Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	r.calls.Add(1)

	if value, ok := ctx.Value(compatCallKey{}).(string); ok {
		r.callCtx.Store(value)
	}

	found, ok := r.tools[req.Name]
	if !ok {
		return nil, tool.ErrToolNotFound
	}

	result, err := found.Execute(ctx, req.Arguments)
	if err != nil {
		return nil, err
	}

	return llm.NewToolResponsePart(req.ID, req.Name, result, false), nil
}

func (r *compatPlainRegistry) ExecuteAll(ctx context.Context, reqs []*llm.ToolRequestPart, _ ...tool.BatchOption) []*llm.ToolResponsePart {
	resps := make([]*llm.ToolResponsePart, len(reqs))
	for i, req := range reqs {
		resps[i], _ = r.Execute(ctx, req)
	}

	return resps
}

func TestSnapshotOptionalCapability(t *testing.T) {
	t.Parallel()

	calls := new(atomic.Int32)
	plain := &compatPlainRegistry{tools: map[string]tool.Tool{"plain": compatTool("plain", "plain", calls)}}
	native := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, native.Register(compatTool("native", "native", new(atomic.Int32))))

	for _, tc := range []struct {
		name     string
		registry tool.Registry
		want     bool
	}{
		{name: "native", registry: native, want: true},
		{name: "migrated wrapper", registry: compatCapturingGate{Registry: native, policy: &compatPolicy{}}, want: true},
		{name: "plain", registry: plain},
		{name: "embedded base interface", registry: &compatGate{Registry: native, policy: &compatPolicy{}}},
	} {
		_, ok := tc.registry.(tool.SnapshotRegistry)
		assert.Equal(t, tc.want, ok, tc.name)
	}

	direct, err := native.Snapshot(t.Context())
	require.NoError(t, err)

	_, ok := direct.(tool.SnapshotRegistry)
	assert.True(t, ok, "a native capture can be captured again")

	frozen, err := tool.Snapshot(t.Context(), plain, direct)
	require.NoError(t, err)

	defs, err := frozen.List(t.Context())
	require.NoError(t, err)
	require.Equal(t, []string{"native", "plain"}, definitionNames(defs))
	assert.True(t, defs[1].Deferred)
	assert.Equal(t, llm.ToolGroup{Name: "plain", Description: "Plain tools"}, defs[1].Group)

	captured, err := frozen.Get(t.Context(), "plain")
	require.NoError(t, err)
	assert.Equal(t, "plain", captured.Definition().Name)

	ctx := context.WithValue(t.Context(), compatCallKey{}, "call")
	resp, err := frozen.Execute(ctx, &llm.ToolRequestPart{ID: "one", Name: "plain"})
	require.NoError(t, err)
	assert.JSONEq(t, `"plain"`, string(resp.Result))
	assert.Equal(t, "call", plain.callCtx.Load(), "the source executes with the call's context")

	resps := frozen.ExecuteAll(t.Context(), []*llm.ToolRequestPart{{ID: "two", Name: "plain"}, {ID: "three", Name: "native"}})
	require.Len(t, resps, 2)
	assert.False(t, resps[0].IsError)
	assert.False(t, resps[1].IsError)
	assert.EqualValues(t, 2, plain.calls.Load())
	assert.EqualValues(t, 2, calls.Load())
}
