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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
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

type registryCredentialsKey struct{}

type registryWireRequest struct {
	method  string
	auth    string
	session string
}

type registryPeer struct {
	url     string
	servers map[string]*sdkmcp.Server
	mu      sync.Mutex
	wire    []registryWireRequest
	handle  func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error)
}

func newRegistryPeer(t *testing.T, legacy bool) *registryPeer {
	t.Helper()

	peer := &registryPeer{servers: make(map[string]*sdkmcp.Server)}

	for _, owner := range []string{"alice", "bob"} {
		opts := &sdkmcp.ServerOptions{HasTools: true}
		if legacy {
			opts.SupportedProtocolVersions = []string{"2025-06-18"}
		}

		server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: owner, Version: "1"}, opts)
		peer.servers[owner] = server
		peer.addTool(owner, owner+"_only", "original", "old")
		peer.addTool(owner, "shared", "shared", "value")
	}

	handler := sdkmcp.NewStreamableHTTPHandler(func(req *http.Request) *sdkmcp.Server {
		owner := strings.Split(strings.TrimPrefix(req.Header.Get("Authorization"), "Bearer "), ":")[0]
		return peer.servers[owner]
	}, &sdkmcp.StreamableHTTPOptions{Stateless: !legacy, JSONResponse: true})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var message struct {
			Method string `json:"method"`
		}

		if req.Body != nil {
			body, err := io.ReadAll(req.Body)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}

			req.Body = io.NopCloser(bytes.NewReader(body))
			_ = json.Unmarshal(body, &message)
		}

		peer.mu.Lock()
		peer.wire = append(peer.wire, registryWireRequest{method: message.Method, auth: req.Header.Get("Authorization"), session: req.Header.Get("Mcp-Session-Id")})
		peer.mu.Unlock()
		handler.ServeHTTP(w, req)
	}))
	peer.url = server.URL
	t.Cleanup(server.Close)

	return peer
}

func (p *registryPeer) addTool(owner, name, description, property string) {
	p.servers[owner].AddTool(&sdkmcp.Tool{
		Name: name, Description: description,
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{property: map[string]any{"type": "string"}}},
	}, func(ctx context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		if p.handle != nil {
			return p.handle(ctx, req)
		}

		return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: req.Extra.Header.Get("Authorization") + " " + req.Params.Name}}}, nil
	})
}

func (p *registryPeer) requests(method string) []registryWireRequest {
	p.mu.Lock()
	defer p.mu.Unlock()

	var requests []registryWireRequest

	for _, req := range p.wire {
		if method == "" || req.method == method {
			requests = append(requests, req)
		}
	}

	return requests
}

func registryCaller(ctx context.Context, partition, bearer string) context.Context {
	return context.WithValue(ctx, registryCredentialsKey{}, Credentials{PartitionKey: partition, Headers: http.Header{"Authorization": {"Bearer " + bearer}}})
}

func registryContextCredentials(ctx context.Context) (Credentials, error) {
	credentials, ok := ctx.Value(registryCredentialsKey{}).(Credentials)
	if !ok {
		return Credentials{}, errors.New("trusted caller identity missing")
	}

	return credentials, nil
}

func newContextRegistry(t *testing.T, peer *registryPeer, opts ...RegistryOption) *Registry {
	t.Helper()

	registry, err := NewRegistry("remote", peer.url, append([]RegistryOption{WithCredentials(registryContextCredentials)}, opts...)...)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, registry.Close()) })

	return registry
}

func registryRequest(name string) *llm.ToolRequestPart {
	return &llm.ToolRequestPart{ID: name, Name: name, Arguments: json.RawMessage(`{}`)}
}

func registryNames(definitions []llm.ToolDefinition) []string {
	names := make([]string, len(definitions))
	for i, definition := range definitions {
		names[i] = definition.Name
	}

	return names
}

func TestContextRegistryPartitionsAndProtocol(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRegistryPeer(t, legacy)
			registry := newContextRegistry(t, peer)
			alice := registryCaller(t.Context(), "alice/tenant-a", "alice:one")
			bob := registryCaller(t.Context(), "bob/tenant-b", "bob:one")

			for _, owner := range []string{"alice", "bob"} {
				ctx := alice
				if owner == "bob" {
					ctx = bob
				}

				snapshot, err := registry.Snapshot(ctx)
				require.NoError(t, err)
				definitions, err := snapshot.List(ctx)
				require.NoError(t, err)
				assert.Equal(t, []string{"remote__" + owner + "_only", "remote__shared"}, registryNames(definitions))

				response, err := snapshot.Execute(ctx, registryRequest("remote__shared"))
				require.NoError(t, err)
				require.NotNil(t, response)
				assert.False(t, response.IsError, string(response.Result))
				assert.Contains(t, string(response.Result), "Bearer "+owner+":one shared")
			}

			calls := peer.requests("tools/call")
			require.Len(t, calls, 2)
			assert.Equal(t, "Bearer alice:one", calls[0].auth)
			assert.Equal(t, "Bearer bob:one", calls[1].auth)

			if legacy {
				assert.Len(t, peer.requests("initialize"), 2)
				assert.NotEmpty(t, calls[0].session)
				assert.NotEmpty(t, calls[1].session)
				assert.NotEqual(t, calls[0].session, calls[1].session)
			} else {
				assert.Empty(t, peer.requests("initialize"))
				assert.Len(t, peer.requests("server/discover"), 2)

				for _, req := range peer.requests("") {
					assert.Empty(t, req.session)
				}
			}
		})
	}
}

func TestContextRegistryPartitionKeyIsAuthoritative(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	registryAdvertiseToolTTL(peer, time.Minute, true)

	clock := newRegistryTestClock()
	registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(time.Minute), func(cfg *registryConfig) { cfg.now = clock.Now })
	first := registryCaller(t.Context(), "scope-a", "alice:one")
	second := registryCaller(t.Context(), "scope-b", "alice:one")
	snapshot, err := registry.Snapshot(first)
	require.NoError(t, err)
	original, err := snapshot.List(first)
	require.NoError(t, err)
	_, err = registry.List(second)
	require.NoError(t, err)
	assert.Len(t, peer.requests("server/discover"), 2, "identical headers must not collapse distinct authorization partitions")
	assert.Len(t, peer.requests("tools/list"), 2)

	peer.addTool("alice", "alice_only", "rotated catalog", "new")
	peer.addTool("alice", "added", "new tool", "value")

	rotated := registryCaller(t.Context(), "scope-a", "alice:two")
	cached, err := registry.List(rotated)
	require.NoError(t, err)
	assert.Equal(t, original, cached, "header rotation must preserve the same partition's unexpired catalog")
	assert.Len(t, peer.requests("tools/list"), 2, "header changes must not invalidate a populated catalog cache")

	response, err := snapshot.Execute(rotated, registryRequest("remote__shared"))
	require.NoError(t, err)
	require.NotNil(t, response)
	assert.False(t, response.IsError, string(response.Result))
	assert.Contains(t, string(response.Result), "Bearer alice:two")
	assert.Len(t, peer.requests("server/discover"), 2, "header rotation alone must not choose a new SDK client")
	calls := peer.requests("tools/call")
	require.Len(t, calls, 1)
	assert.Equal(t, "Bearer alice:two", calls[0].auth)

	clock.Advance(time.Minute)

	fresh, err := registry.List(rotated)
	require.NoError(t, err)
	assert.Contains(t, registryNames(fresh), "remote__added")
	assert.Len(t, peer.requests("tools/list"), 3, "expiry, not credential rotation, selects the updated catalog")
	assert.Equal(t, "Bearer alice:two", peer.requests("tools/list")[2].auth)
}

func TestContextRegistryRefreshAndFrozenMembership(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	registry := newContextRegistry(t, peer)
	ctx := registryCaller(t.Context(), "alice", "alice:one")
	old, err := registry.Snapshot(ctx)
	require.NoError(t, err)

	before := len(peer.requests("tools/list"))
	peer.addTool("alice", "alice_only", "updated", "new")
	peer.servers["alice"].RemoveTools("shared")
	peer.addTool("alice", "added", "new tool", "value")

	current, err := registry.Snapshot(ctx)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(peer.requests("tools/list")), before+1, "each step must perform remote discovery without a server-advertised TTL")

	oldDefinitions, err := old.List(ctx)
	require.NoError(t, err)
	newDefinitions, err := current.List(ctx)
	require.NoError(t, err)
	assert.Equal(t, []string{"remote__alice_only", "remote__shared"}, registryNames(oldDefinitions))
	assert.Equal(t, []string{"remote__added", "remote__alice_only"}, registryNames(newDefinitions))
	assert.Equal(t, "original", oldDefinitions[0].Description)
	assert.Contains(t, string(oldDefinitions[0].Parameters), `"old"`)
	assert.Equal(t, "updated", newDefinitions[1].Description)
	assert.Contains(t, string(newDefinitions[1].Parameters), `"new"`)

	oldDefinitions[0].Parameters[0] = 'x'
	again, err := old.List(ctx)
	require.NoError(t, err)
	assert.True(t, json.Valid(again[0].Parameters), "returned schemas must not mutate the frozen snapshot")

	for _, name := range []string{"remote__shared", "remote__unknown", "shared"} {
		response, err := current.Execute(ctx, registryRequest(name))
		require.NoError(t, err)
		require.NotNil(t, response)
		assert.True(t, response.IsError)

		_, err = current.Get(ctx, name)
		require.ErrorIs(t, err, tool.ErrToolNotFound)
	}

	assert.Empty(t, peer.requests("tools/call"), "removed and invented names must never reach the remote peer")
	require.ErrorIs(t, registry.Register(nil), tool.ErrRegistryReadOnly)
	require.ErrorIs(t, registry.Unregister("remote__alice_only"), tool.ErrRegistryReadOnly)
	require.ErrorIs(t, old.Unregister("remote__alice_only"), tool.ErrRegistryReadOnly)
}

func TestContextRegistryFailsClosed(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name     string
		provider CredentialsProvider
	}{
		{"missing identity", registryContextCredentials},
		{"empty partition", func(context.Context) (Credentials, error) {
			return Credentials{Headers: http.Header{"Authorization": {"Bearer alice:one"}}}, nil
		}},
		{"provider failure", func(context.Context) (Credentials, error) {
			return Credentials{}, errors.New("identity backend unavailable")
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			peer := newRegistryPeer(t, false)
			registry, err := NewRegistry("remote", peer.url, WithCredentials(test.provider))
			require.NoError(t, err)
			t.Cleanup(func() { assert.NoError(t, registry.Close()) })
			_, err = registry.Snapshot(t.Context())
			require.Error(t, err)
			_, err = registry.List(t.Context())
			require.Error(t, err)
			_, err = registry.Get(t.Context(), "remote__shared")
			require.Error(t, err)
			response, err := registry.Execute(t.Context(), registryRequest("remote__shared"))
			assert.True(t, err != nil || response != nil && response.IsError)
			responses := registry.ExecuteAll(t.Context(), []*llm.ToolRequestPart{registryRequest("remote__shared"), nil}, tool.WithMaxConcurrency(1))
			require.Len(t, responses, 2)

			for _, response := range responses {
				require.NotNil(t, response)
				assert.True(t, response.IsError)
			}

			assert.Equal(t, "remote__shared", responses[0].ID)
			assert.Equal(t, "remote__shared", responses[0].Name)
			assert.Empty(t, peer.requests(""))
		})
	}

	t.Run("snapshot rejects another caller", func(t *testing.T) {
		t.Parallel()

		peer := newRegistryPeer(t, false)
		registry := newContextRegistry(t, peer)
		alice := registryCaller(t.Context(), "alice", "alice:one")
		bob := registryCaller(t.Context(), "bob", "bob:one")
		snapshot, err := registry.Snapshot(alice)
		require.NoError(t, err)
		response, err := snapshot.Execute(bob, registryRequest("remote__shared"))
		assert.True(t, err != nil || response != nil && response.IsError)
		frozenTool, err := snapshot.Get(alice, "remote__shared")
		require.NoError(t, err)
		_, err = frozenTool.Execute(bob, json.RawMessage(`{}`))
		require.Error(t, err)
		_, err = frozenTool.Execute(t.Context(), json.RawMessage(`{}`))
		require.Error(t, err)
		assert.Empty(t, peer.requests("tools/call"))
	})

	t.Run("canceled before discovery", func(t *testing.T) {
		t.Parallel()

		peer := newRegistryPeer(t, false)
		registry := newContextRegistry(t, peer)
		ctx, cancel := context.WithCancel(registryCaller(t.Context(), "alice", "alice:one"))
		cancel()

		_, err := registry.Snapshot(ctx)
		require.ErrorIs(t, err, context.Canceled)
		assert.Empty(t, peer.requests(""))
	})
}

func TestContextRegistryToolPolicies(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	release := make(chan struct{})

	t.Cleanup(func() { close(release) })

	peer.handle = func(ctx context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		if req.Params.Name == "shared" {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-release:
				return &sdkmcp.CallToolResult{}, nil
			}
		}

		return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: strings.Repeat("large", 100)}}}, nil
	}
	group := llm.ToolGroup{Name: "capability", Description: "caller data", Instructions: "Use only for the caller"}
	registry := newContextRegistry(t, peer, WithRegistryToolOptions(tool.WithTimeout(2*time.Second), tool.WithMaxResponseTokens(20), tool.WithResponseTooLargeMessage("paginate"), tool.WithDeferred(), tool.WithGroup(group)))
	ctx := registryCaller(t.Context(), "alice", "alice:one")
	snapshot, err := registry.Snapshot(ctx)
	require.NoError(t, err)
	definitions, err := snapshot.List(ctx)
	require.NoError(t, err)

	for _, definition := range definitions {
		assert.True(t, definition.Deferred)
		assert.Equal(t, group, definition.Group)
	}

	response, err := snapshot.Execute(ctx, registryRequest("remote__alice_only"))
	require.NoError(t, err)
	require.NotNil(t, response)
	assert.Contains(t, string(response.Result), "response_too_large")
	assert.Contains(t, string(response.Result), "paginate")
	response, err = snapshot.Execute(ctx, registryRequest("remote__shared"))
	require.NoError(t, err)
	require.NotNil(t, response)
	assert.True(t, response.IsError)
	assert.Contains(t, string(response.Result), "timeout")
}

func TestContextRegistryFullResultPayload(t *testing.T) {
	t.Parallel()

	for _, isError := range []bool{false, true} {
		t.Run(fmt.Sprintf("isError=%t", isError), func(t *testing.T) {
			t.Parallel()

			peer := newRegistryPeer(t, false)
			peer.handle = func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
				return &sdkmcp.CallToolResult{
					Content:           []sdkmcp.Content{&sdkmcp.TextContent{Text: "explanation"}},
					StructuredContent: map[string]any{"account": "alice", "count": 7},
					Meta:              sdkmcp.Meta{"trace": "server-trace"}, IsError: isError,
				}, nil
			}
			registry := newContextRegistry(t, peer)
			ctx := registryCaller(t.Context(), "alice", "alice:one")
			response, err := registry.Execute(ctx, registryRequest("remote__shared"))
			require.NoError(t, err)
			require.NotNil(t, response)
			assert.Equal(t, isError, response.IsError)
			var payload map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(response.Result, &payload))
			assert.JSONEq(t, `[{"type":"text","text":"explanation"}]`, string(payload["content"]))
			assert.JSONEq(t, `{"account":"alice","count":7}`, string(payload["structuredContent"]))
			var metadata map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(payload["_meta"], &metadata))
			assert.JSONEq(t, `"server-trace"`, string(metadata["trace"]))

			captured, err := registry.Get(ctx, "remote__shared")
			require.NoError(t, err)
			_, err = captured.Execute(ctx, json.RawMessage(`{}`))

			if isError {
				var resultError *tool.ResultError
				require.ErrorAs(t, err, &resultError)
				assert.JSONEq(t, string(response.Result), string(resultError.Result))
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestContextRegistryExecuteAll(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	var active, maximum atomic.Int32
	var parallelPhase atomic.Bool
	arrived := make(chan struct{}, 3)
	release := make(chan struct{})

	var unblock sync.Once
	defer unblock.Do(func() { close(release) })

	peer.handle = func(ctx context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		n := active.Add(1)
		defer active.Add(-1)

		for previous := maximum.Load(); n > previous; previous = maximum.Load() {
			if maximum.CompareAndSwap(previous, n) {
				break
			}
		}

		if parallelPhase.Load() {
			arrived <- struct{}{}

			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-release:
			}
		}

		return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: req.Params.Name}}}, nil
	}
	registry := newContextRegistry(t, peer)
	ctx := registryCaller(t.Context(), "alice", "alice:one")

	reqs := []*llm.ToolRequestPart{registryRequest("remote__shared"), nil, registryRequest("remote__alice_only"), registryRequest("remote__missing"), registryRequest("remote__shared")}
	for i, req := range reqs {
		if req != nil {
			req.ID = fmt.Sprintf("call-%d", i)
		}
	}

	responses := registry.ExecuteAll(ctx, reqs, tool.WithMaxConcurrency(1))
	require.Len(t, responses, len(reqs))

	for i, response := range responses {
		require.NotNil(t, response)

		if reqs[i] != nil {
			assert.Equal(t, reqs[i].ID, response.ID)
			assert.Equal(t, reqs[i].Name, response.Name)
		}

		assert.Equal(t, i == 1 || i == 3, response.IsError, string(response.Result))
	}

	assert.EqualValues(t, 1, maximum.Load())
	assert.Len(t, peer.requests("tools/call"), 3)
	maximum.Store(0)

	parallelPhase.Store(true)

	completed := make(chan []*llm.ToolResponsePart, 1)

	go func() {
		completed <- registry.ExecuteAll(ctx, []*llm.ToolRequestPart{reqs[0], reqs[2], reqs[4]}, tool.WithMaxConcurrency(2))
	}()

	for range 2 {
		select {
		case <-arrived:
		case <-time.After(5 * time.Second):
			t.Fatal("WithMaxConcurrency(2) did not allow two real MCP handlers to enter before release")
		}
	}

	assert.EqualValues(t, 2, active.Load())
	unblock.Do(func() { close(release) })

	var parallel []*llm.ToolResponsePart
	select {
	case parallel = <-completed:
	case <-time.After(5 * time.Second):
		t.Fatal("parallel batch did not finish after release")
	}

	require.Len(t, parallel, 3)

	for _, response := range parallel {
		require.NotNil(t, response)
		assert.False(t, response.IsError, string(response.Result))
	}

	assert.EqualValues(t, 2, maximum.Load())
	assert.Len(t, peer.requests("tools/call"), 6)
	assert.Empty(t, registry.ExecuteAll(ctx, nil))
	response, err := registry.Execute(ctx, nil)
	assert.Nil(t, response)
	require.ErrorIs(t, err, tool.ErrToolRequestNil)
}

func TestContextRegistryCapacityCancellationAndShutdown(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	started := make(chan struct{})
	release := make(chan struct{})

	t.Cleanup(func() { close(release) })

	peer.handle = func(ctx context.Context, _ *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		close(started)

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-release:
			return &sdkmcp.CallToolResult{}, nil
		}
	}
	registry := newContextRegistry(t, peer, WithRegistryMaxConnections(1))
	alice := registryCaller(t.Context(), "alice", "alice:one")
	snapshot, err := registry.Snapshot(alice)
	require.NoError(t, err)

	executed := make(chan *llm.ToolResponsePart, 1)

	go func() {
		response, _ := snapshot.Execute(alice, registryRequest("remote__shared"))
		executed <- response
	}()

	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("remote execution never started")
	}

	bob, cancel := context.WithTimeout(registryCaller(t.Context(), "bob", "bob:one"), 50*time.Millisecond)
	defer cancel()

	_, err = registry.Snapshot(bob)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	assert.Len(t, peer.requests("server/discover"), 1, "capacity wait must not create an extra client")

	shutdown, stop := context.WithTimeout(t.Context(), 5*time.Second)
	defer stop()

	require.NoError(t, registry.Shutdown(shutdown))

	select {
	case response := <-executed:
		require.NotNil(t, response)
		assert.True(t, response.IsError)
	case <-shutdown.Done():
		t.Fatal("shutdown did not drain execution")
	}

	_, err = registry.List(alice)
	require.ErrorIs(t, err, ErrClosed)
	assert.NoError(t, registry.Close())
}

func TestContextRegistryIdleEviction(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	registry := newContextRegistry(t, peer, WithRegistryMaxConnections(1), WithRegistryIdleTimeout(time.Millisecond))
	clock := newRegistryTestClock()
	registry.config.now = clock.Now
	alice := registryCaller(t.Context(), "alice", "alice:one")
	_, err := registry.List(alice)
	require.NoError(t, err)
	clock.Advance(10 * time.Millisecond)

	_, err = registry.List(alice)
	require.NoError(t, err)
	assert.Len(t, peer.requests("server/discover"), 2, "expired idle client must be replaced")
}

func TestContextRegistryDiscoveryWaitRespectsDeadline(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	captured := make(chan struct{})
	release := make(chan struct{})

	var first sync.Once

	peer.servers["alice"].AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
		return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
			result, err := next(ctx, method, req)
			if method == "tools/list" {
				first.Do(func() {
					close(captured)
					<-release
				})
			}

			return result, err
		}
	})
	registry := newContextRegistry(t, peer)
	t.Cleanup(func() { close(release) })
	ctx := registryCaller(t.Context(), "alice", "alice:one")
	finished := make(chan error, 1)

	go func() {
		_, err := registry.Snapshot(ctx)
		finished <- err
	}()

	select {
	case <-captured:
	case <-time.After(5 * time.Second):
		t.Fatal("first discovery never reached the peer")
	}

	waiting, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()

	returned := make(chan error, 1)

	go func() {
		_, err := registry.Snapshot(waiting)
		returned <- err
	}()

	select {
	case err := <-returned:
		require.ErrorIs(t, err, context.DeadlineExceeded)
	case <-time.After(2 * time.Second):
		t.Fatal("same-partition discovery ignored its deadline while waiting for another list")
	}

	assert.Len(t, peer.requests("tools/list"), 1, "the blocked first list must not be bypassed")

	select {
	case err := <-finished:
		t.Fatalf("first discovery unexpectedly completed before release: %v", err)
	default:
	}
}

func TestContextRegistryCatalogExpiryIsOnDemand(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	registryAdvertiseToolTTL(peer, time.Minute, true)

	clock := newRegistryTestClock()
	registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(time.Minute), func(cfg *registryConfig) { cfg.now = clock.Now })
	ctx := registryCaller(t.Context(), "alice", "alice:one")
	_, err := registry.List(ctx)
	require.NoError(t, err)
	_, err = registry.List(ctx)
	require.NoError(t, err)
	assert.Len(t, peer.requests("tools/list"), 1, "the catalog must be cached before testing expiry")

	peer.addTool("alice", "added", "new tool", "value")
	clock.Advance(time.Minute)
	assert.Len(t, peer.requests("tools/list"), 1, "expiration alone must not discover tools")

	fresh, err := registry.List(ctx)
	require.NoError(t, err)
	assert.Contains(t, registryNames(fresh), "remote__added")
	assert.Len(t, peer.requests("tools/list"), 2, "an explicit operation must discover the expired catalog")
}

func TestContextRegistrySharedAgentIsolation(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	registry := newContextRegistry(t, peer)
	model := fakellm.NewFakeModel()

	for _, owner := range []string{"alice", "bob"} {
		name := "remote__" + owner + "_only"
		model.When(fakellm.LastMessageHasToolResponse(name)).ThenRespondText("done")
		model.When(fakellm.HasTool(name)).ThenRespondWithToolCall(name, map[string]any{})
	}

	ag, err := llmagent.New("shared", "Use your caller's tools", model, llmagent.WithTools(registry))
	require.NoError(t, err)

	var wg sync.WaitGroup

	runErrors := make(chan error, 2)

	for _, owner := range []string{"alice", "bob"} {
		wg.Go(func() {
			ctx := registryCaller(t.Context(), owner, owner+":one")
			state := &session.State{ID: owner, Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart(owner))}}

			inv := agent.NewInvocationMetadata(state, agent.Info{})
			for _, runErr := range ag.Run(ctx, inv) {
				if runErr != nil {
					runErrors <- runErr
					return
				}
			}

			var results []*llm.ToolResponsePart
			for _, message := range state.Messages {
				results = append(results, message.ToolResponses()...)
			}

			if assert.Len(t, results, 1) {
				assert.False(t, results[0].IsError, string(results[0].Result))
				assert.Equal(t, "remote__"+owner+"_only", results[0].Name)
				assert.Contains(t, string(results[0].Result), "Bearer "+owner+":one")
			}
		})
	}

	wg.Wait()
	close(runErrors)

	for runErr := range runErrors {
		require.NoError(t, runErr)
	}

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 4)

	for _, call := range calls {
		names := registryNames(call.Request.Tools)
		assert.NotEqual(t, slices.Contains(names, "remote__alice_only"), slices.Contains(names, "remote__bob_only"), "one model request must never contain both caller catalogs")
	}

	assert.Len(t, peer.requests("tools/call"), 2)
}
