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
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

type snapshotProbe struct {
	def llm.ToolDefinition
	run func(context.Context, json.RawMessage) (json.RawMessage, error)
}

func (p *snapshotProbe) Definition() llm.ToolDefinition { return p.def }

func (p *snapshotProbe) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	return p.run(ctx, args)
}

func TestSnapshotPreservesPolicies(t *testing.T) {
	t.Parallel()

	group := llm.ToolGroup{Name: "issues", Description: "Issue tracker", Instructions: "Find the ID first."}
	cases := []struct {
		name      string
		opts      []tool.Option
		run       func(context.Context, json.RawMessage) (json.RawMessage, error)
		wantError bool
		contains  string
	}{
		{
			name: "timeout", opts: []tool.Option{tool.WithTimeout(time.Millisecond)},
			run: func(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
				<-ctx.Done()
				return nil, ctx.Err()
			},
			wantError: true, contains: tool.ErrToolExecutionTimeout.Error(),
		},
		{
			name: "result cap", opts: []tool.Option{tool.WithMaxResponseTokens(1), tool.WithResponseTooLargeMessage("Use pagination")},
			run: func(context.Context, json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"data":"a large response"}`), nil
			},
			contains: "Use pagination",
		},
		{
			name: "result error cap", opts: []tool.Option{tool.WithMaxResponseTokens(1)},
			run: func(context.Context, json.RawMessage) (json.RawMessage, error) {
				return nil, &tool.ResultError{Result: json.RawMessage(`{"error":"a large failure"}`)}
			},
			wantError: true, contains: "response_too_large",
		},
		{
			name: "unlimited", opts: []tool.Option{tool.WithTimeout(0), tool.WithMaxResponseTokens(0)},
			run: func(context.Context, json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`"` + strings.Repeat("x", 100004) + `"`), nil
			},
			contains: strings.Repeat("x", 100004),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			source := tool.NewRegistry(tool.RegistryConfig{})

			opts := append([]tool.Option{tool.WithDeferred(), tool.WithGroup(group)}, tc.opts...)
			require.NoError(t, source.Register(&snapshotProbe{def: llm.ToolDefinition{Name: "read"}, run: tc.run}, opts...))
			frozen, err := tool.Snapshot(t.Context(), source)
			require.NoError(t, err)
			composed, err := tool.Snapshot(t.Context(), nil, frozen)
			require.NoError(t, err)

			defs, err := composed.List(t.Context())
			require.NoError(t, err)
			require.Len(t, defs, 1)
			assert.True(t, defs[0].Deferred)
			assert.Equal(t, group, defs[0].Group)

			resp, err := composed.Execute(t.Context(), &llm.ToolRequestPart{ID: "call", Name: "read"})
			require.NoError(t, err)
			assert.Equal(t, "call", resp.ID)
			assert.Equal(t, "read", resp.Name)
			assert.Equal(t, tc.wantError, resp.IsError)
			assert.Contains(t, string(resp.Result), tc.contains)
		})
	}
}

func TestSnapshotIsolation(t *testing.T) {
	t.Parallel()

	original := &snapshotProbe{
		def: llm.ToolDefinition{Name: "read", Description: "Original", Parameters: json.RawMessage(`{"type":"object"}`), Metadata: map[string]any{"version": "old"}},
		run: func(context.Context, json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`"old"`), nil },
	}
	source := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, source.Register(original))
	frozen, err := tool.Snapshot(t.Context(), source)
	require.NoError(t, err)

	original.def.Description = "Changed"
	original.def.Parameters[0] = 'x'
	original.def.Metadata["version"] = mutatedSnapshotValue

	require.NoError(t, source.Unregister("read"))
	require.NoError(t, source.Register(&snapshotProbe{
		def: llm.ToolDefinition{Name: "read"},
		run: func(context.Context, json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`"new"`), nil },
	}))
	require.NoError(t, source.Register(&mockTool{name: "added"}))

	defs, err := frozen.List(t.Context())
	require.NoError(t, err)
	require.Len(t, defs, 1)
	assert.Equal(t, "Original", defs[0].Description)
	assert.JSONEq(t, `{"type":"object"}`, string(defs[0].Parameters))
	assert.Equal(t, "old", defs[0].Metadata["version"])
	defs[0].Parameters[0] = 'y'
	defs[0].Metadata["version"] = "caller mutation"

	defs, err = frozen.List(t.Context())
	require.NoError(t, err)
	assert.JSONEq(t, `{"type":"object"}`, string(defs[0].Parameters))
	assert.Equal(t, "old", defs[0].Metadata["version"])

	capturedTool, err := frozen.Get(t.Context(), "read")
	require.NoError(t, err)
	assert.Equal(t, "Original", capturedTool.Definition().Description)
	assert.JSONEq(t, `{"type":"object"}`, string(capturedTool.Definition().Parameters))
	capturedTool.Definition().Parameters[0] = 'z'
	assert.JSONEq(t, `{"type":"object"}`, string(capturedTool.Definition().Parameters))

	resp, err := frozen.Execute(t.Context(), &llm.ToolRequestPart{Name: "read"})
	require.NoError(t, err)
	assert.JSONEq(t, `"old"`, string(resp.Result))
	require.ErrorIs(t, frozen.Register(&mockTool{name: "blocked"}), tool.ErrRegistryReadOnly)
	require.ErrorIs(t, frozen.Unregister("read"), tool.ErrRegistryReadOnly)
	_, err = frozen.Get(t.Context(), "added")
	require.ErrorIs(t, err, tool.ErrToolNotFound)
}

const mutatedSnapshotValue = "new"

func snapshotNestedMetadata() map[string]any {
	return map[string]any{
		"policy": map[string]any{"scope": "old"},
		"items":  []any{map[string]any{"scope": "old"}, []any{"old"}},
		"names":  []string{"old"},
		"labels": map[string]string{"scope": "old"},
		"groups": []map[string]any{{"items": []any{"old"}}},
		"raw":    json.RawMessage(`"old"`),
		"bytes":  []byte("old"),
	}
}

func mutateSnapshotMetadata(t *testing.T, metadata map[string]any) {
	t.Helper()

	policy, ok := metadata["policy"].(map[string]any)
	require.True(t, ok)

	policy["scope"] = mutatedSnapshotValue

	items, ok := metadata["items"].([]any)
	require.True(t, ok)
	item, ok := items[0].(map[string]any)
	require.True(t, ok)

	item["scope"] = mutatedSnapshotValue

	nested, ok := items[1].([]any)
	require.True(t, ok)

	nested[0] = mutatedSnapshotValue
	items[0] = "replaced"

	names, ok := metadata["names"].([]string)
	require.True(t, ok)

	names[0] = mutatedSnapshotValue

	labels, ok := metadata["labels"].(map[string]string)
	require.True(t, ok)

	labels["scope"] = mutatedSnapshotValue

	groups, ok := metadata["groups"].([]map[string]any)
	require.True(t, ok)
	groupItems, ok := groups[0]["items"].([]any)
	require.True(t, ok)

	groupItems[0] = mutatedSnapshotValue
	groups[0]["added"] = true

	raw, ok := metadata["raw"].(json.RawMessage)
	require.True(t, ok)
	copy(raw, `"new"`)

	bytes, ok := metadata["bytes"].([]byte)
	require.True(t, ok)
	copy(bytes, mutatedSnapshotValue)
}

func TestSnapshotNestedMetadataIsolation(t *testing.T) {
	t.Parallel()

	for _, mutation := range []string{"source", "List", "Get"} {
		t.Run(mutation, func(t *testing.T) {
			t.Parallel()

			metadata := snapshotNestedMetadata()
			source := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, source.Register(&snapshotProbe{def: llm.ToolDefinition{Name: "read", Metadata: metadata}}))
			frozen, err := tool.Snapshot(t.Context(), source)
			require.NoError(t, err)
			captured, err := frozen.Get(t.Context(), "read")
			require.NoError(t, err)

			switch mutation {
			case "source":
				mutateSnapshotMetadata(t, metadata)
			case "List":
				defs, err := frozen.List(t.Context())
				require.NoError(t, err)
				mutateSnapshotMetadata(t, defs[0].Metadata)
			case "Get":
				mutateSnapshotMetadata(t, captured.Definition().Metadata)
			}

			defs, err := frozen.List(t.Context())
			require.NoError(t, err)
			assert.Equal(t, snapshotNestedMetadata(), defs[0].Metadata)
			assert.Equal(t, snapshotNestedMetadata(), captured.Definition().Metadata)
		})
	}
}

func TestSnapshotNestedMetadataConcurrentIsolation(t *testing.T) {
	t.Parallel()

	for _, mutation := range []string{"source", "List", "Get"} {
		t.Run(mutation, func(t *testing.T) {
			t.Parallel()

			metadata := snapshotNestedMetadata()
			source := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, source.Register(&snapshotProbe{def: llm.ToolDefinition{Name: "read", Metadata: metadata}}))
			frozen, err := tool.Snapshot(t.Context(), source)
			require.NoError(t, err)
			captured, err := frozen.Get(t.Context(), "read")
			require.NoError(t, err)

			switch mutation {
			case "List":
				defs, err := frozen.List(t.Context())
				require.NoError(t, err)

				metadata = defs[0].Metadata
			case "Get":
				metadata = captured.Definition().Metadata
			}

			policy, ok := metadata["policy"].(map[string]any)
			require.True(t, ok)
			items, ok := metadata["items"].([]any)
			require.True(t, ok)

			done := make(chan struct{})

			go func() {
				defer close(done)

				for range 1000 {
					policy["scope"] = mutatedSnapshotValue
					items[0] = "changed"

					runtime.Gosched()
				}
			}()

			expected := snapshotNestedMetadata()

			for range 100 {
				defs, err := frozen.List(t.Context())
				require.NoError(t, err)
				assert.Equal(t, expected, defs[0].Metadata)
				assert.Equal(t, expected, captured.Definition().Metadata)
			}

			<-done
		})
	}
}

func TestSnapshotComposition(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		secondName  string
		firstGroup  llm.ToolGroup
		secondGroup llm.ToolGroup
		wantErr     error
	}{
		{name: "duplicates", secondName: "a", wantErr: tool.ErrToolAlreadyRegistered},
		{name: "description conflict", secondName: "b", firstGroup: llm.ToolGroup{Name: "g", Description: "one"}, secondGroup: llm.ToolGroup{Name: "g", Description: "two"}, wantErr: tool.ErrInvalidToolConfig},
		{name: "instructions conflict", secondName: "b", firstGroup: llm.ToolGroup{Name: "g", Instructions: "one"}, secondGroup: llm.ToolGroup{Name: "g", Instructions: "two"}, wantErr: tool.ErrInvalidToolConfig},
		{name: "resolved group", secondName: "b", firstGroup: llm.ToolGroup{Name: "g", Description: "Group"}, secondGroup: llm.ToolGroup{Name: "g", Instructions: "Instructions"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			first, second := tool.NewRegistry(tool.RegistryConfig{}), tool.NewRegistry(tool.RegistryConfig{})

			var firstOpts, secondOpts []tool.Option
			if tc.firstGroup.Name != "" {
				firstOpts = append(firstOpts, tool.WithGroup(tc.firstGroup))
				secondOpts = append(secondOpts, tool.WithGroup(tc.secondGroup))
			}

			require.NoError(t, first.Register(&mockTool{name: "a"}, firstOpts...))
			require.NoError(t, second.Register(&mockTool{name: tc.secondName}, secondOpts...))

			combined, err := tool.Snapshot(t.Context(), second, first)
			if tc.wantErr != nil {
				require.ErrorIs(t, err, tc.wantErr)
				assert.Nil(t, combined)

				return
			}

			require.NoError(t, err)

			defs, err := combined.List(t.Context())
			require.NoError(t, err)
			assert.Equal(t, []string{"a", "b"}, definitionNames(defs))

			for _, def := range defs {
				assert.Equal(t, llm.ToolGroup{Name: "g", Description: "Group", Instructions: "Instructions"}, def.Group)
			}
		})
	}
}

type customSnapshotRegistry struct {
	tool.Registry

	captured tool.Registry
	err      error
	calls    atomic.Int32
}

func (r *customSnapshotRegistry) Snapshot(ctx context.Context) (tool.Registry, error) {
	r.calls.Add(1)

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return r.captured, r.err
}

type opaqueRegistry struct{ tool.Registry }

func TestSnapshotCustomRegistry(t *testing.T) {
	t.Parallel()

	source := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, source.Register(&mockTool{name: "custom", execFunc: func(context.Context, json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`"` + strings.Repeat("x", 100004) + `"`), nil
	}}, tool.WithMaxResponseTokens(0), tool.WithGroup(llm.ToolGroup{Name: "g"}), tool.WithDeferred()))
	captured, err := tool.Snapshot(t.Context(), source)
	require.NoError(t, err)

	custom := &customSnapshotRegistry{Registry: source, captured: &opaqueRegistry{Registry: captured}}
	frozen, err := tool.Snapshot(t.Context(), custom)
	require.NoError(t, err)
	assert.EqualValues(t, 1, custom.calls.Load())
	require.NoError(t, source.Unregister("custom"))
	defs, err := frozen.List(t.Context())
	require.NoError(t, err)
	assert.True(t, defs[0].Deferred)
	assert.Equal(t, "g", defs[0].Group.Name)
	resp, err := frozen.Execute(t.Context(), &llm.ToolRequestPart{ID: "call", Name: "custom"})
	require.NoError(t, err)
	assert.False(t, resp.IsError)
	assert.Len(t, resp.Result, 100006)
	assert.Equal(t, "call", resp.ID)
}

type failingDiscoveryRegistry struct {
	tool.Registry

	err    error
	getErr bool
}

func (r *failingDiscoveryRegistry) List(ctx context.Context) ([]llm.ToolDefinition, error) {
	if !r.getErr {
		return nil, r.err
	}

	return r.Registry.List(ctx)
}

func (r *failingDiscoveryRegistry) Get(context.Context, string) (tool.Tool, error) {
	return nil, r.err
}

func TestSnapshotCustomRegistryErrors(t *testing.T) {
	t.Parallel()

	source := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, source.Register(&mockTool{name: "probe"}))

	failure := errors.New("discovery failed")

	for _, tc := range []struct {
		name     string
		registry tool.Registry
	}{
		{name: "Snapshot", registry: &customSnapshotRegistry{Registry: source, err: failure}},
		{name: "List", registry: &customSnapshotRegistry{Registry: source, captured: &failingDiscoveryRegistry{Registry: source, err: failure}}},
		{name: "Get", registry: &customSnapshotRegistry{Registry: source, captured: &failingDiscoveryRegistry{Registry: source, err: failure, getErr: true}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			captured, err := tool.Snapshot(t.Context(), source, tc.registry)
			require.ErrorIs(t, err, failure)
			assert.Nil(t, captured)
		})
	}

	t.Run("nil snapshot", func(t *testing.T) {
		t.Parallel()

		captured, err := tool.Snapshot(t.Context(), &customSnapshotRegistry{Registry: source})
		require.Error(t, err)
		assert.Nil(t, captured)
	})
}

func TestRegistryDiscoveryCancellation(t *testing.T) {
	t.Parallel()

	source := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, source.Register(&mockTool{name: "probe"}))
	frozen, err := source.Snapshot(t.Context())
	require.NoError(t, err)

	for _, tc := range []struct {
		name     string
		registry tool.Registry
	}{
		{name: "native", registry: source},
		{name: "frozen", registry: frozen},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithCancel(t.Context())
			cancel()

			defs, err := tc.registry.List(ctx)
			require.ErrorIs(t, err, context.Canceled)
			assert.Nil(t, defs)

			found, err := tc.registry.Get(ctx, "probe")
			require.ErrorIs(t, err, context.Canceled)
			assert.Nil(t, found)

			captured, err := tc.registry.Snapshot(ctx)
			require.ErrorIs(t, err, context.Canceled)
			assert.Nil(t, captured)
			captured, err = tool.Snapshot(ctx, tc.registry)
			require.ErrorIs(t, err, context.Canceled)
			assert.Nil(t, captured)
		})
	}
}

func TestSnapshotBatchConcurrency(t *testing.T) {
	t.Parallel()

	source := tool.NewRegistry(tool.RegistryConfig{})
	started, release := make(chan struct{}, 3), make(chan struct{})
	var active, maximum atomic.Int32

	require.NoError(t, source.Register(&mockTool{name: "work", execFunc: func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		current := active.Add(1)
		defer active.Add(-1)

		for previous := maximum.Load(); current > previous; previous = maximum.Load() {
			if maximum.CompareAndSwap(previous, current) {
				break
			}
		}

		started <- struct{}{}

		select {
		case <-release:
			return args, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}}))
	frozen, err := tool.Snapshot(t.Context(), source)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	requests := []*llm.ToolRequestPart{{ID: "1", Name: "work", Arguments: json.RawMessage(`1`)}, {ID: "2", Name: "work", Arguments: json.RawMessage(`2`)}, {ID: "3", Name: "work", Arguments: json.RawMessage(`3`)}}
	completed := make(chan []*llm.ToolResponsePart, 1)

	go func() { completed <- frozen.ExecuteAll(ctx, requests, tool.WithMaxConcurrency(2)) }()

	for range 2 {
		select {
		case <-started:
		case <-ctx.Done():
			t.Fatal("two workers did not start")
		}
	}

	assert.EqualValues(t, 2, active.Load())
	close(release)

	select {
	case results := <-completed:
		require.Len(t, results, 3)

		for i, result := range results {
			assert.Equal(t, requests[i].ID, result.ID)
			assert.Equal(t, requests[i].Arguments, result.Result)
		}
	case <-ctx.Done():
		t.Fatal("batch did not complete")
	}

	assert.EqualValues(t, 2, maximum.Load())
}

func TestSnapshotExecutionErrors(t *testing.T) {
	t.Parallel()

	frozen, err := tool.Snapshot(t.Context(), nil)
	require.NoError(t, err)
	defs, err := frozen.List(t.Context())
	require.NoError(t, err)
	assert.Empty(t, defs)
	_, err = frozen.Execute(t.Context(), nil)
	require.ErrorIs(t, err, tool.ErrToolRequestNil)
	results := frozen.ExecuteAll(t.Context(), []*llm.ToolRequestPart{nil, {ID: "missing", Name: "missing"}})
	require.Len(t, results, 2)

	for _, result := range results {
		assert.True(t, result.IsError)
	}

	assert.Equal(t, "missing", results[1].ID)
	assert.Contains(t, string(results[1].Result), tool.ErrToolNotFound.Error())
	assert.Empty(t, frozen.ExecuteAll(t.Context(), nil))

	source := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, source.Register(&mockTool{name: "failure", execFunc: func(context.Context, json.RawMessage) (json.RawMessage, error) {
		return nil, errors.New("execution failed")
	}}))
	frozen, err = tool.Snapshot(t.Context(), source)
	require.NoError(t, err)
	resp, err := frozen.Execute(t.Context(), &llm.ToolRequestPart{Name: "failure"})
	require.NoError(t, err)
	assert.True(t, resp.IsError)
	assert.Contains(t, string(resp.Result), "execution failed")
}
