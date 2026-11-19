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
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegistryLegacySubscriptionReconnectUsesLatestValidatedCaller(t *testing.T) {
	t.Parallel()

	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "peer", Version: "1"}, &sdkmcp.ServerOptions{
		HasTools: true, SupportedProtocolVersions: []string{"2025-06-18"},
	})
	addRegistryConnectionTool(server, "echo")
	var lists atomic.Int32

	server.AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
		return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
			result, err := next(ctx, method, req)
			if list, ok := result.(*sdkmcp.ListToolsResult); ok {
				lists.Add(1)

				list.TTLMs = int(time.Hour.Milliseconds())
			}

			return result, err
		}
	})

	handler := sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server }, &sdkmcp.StreamableHTTPOptions{JSONResponse: true})
	var expired atomic.Bool
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if expired.Load() && req.Header.Get("Authorization") != "Bearer alice:two" {
			http.Error(w, "expired bearer", http.StatusUnauthorized)

			return
		}

		handler.ServeHTTP(w, req)
	}))
	t.Cleanup(peer.Close)

	streams := make(chan string, 8)
	transport := newRegistryConnectionHTTPTransport(t)
	t.Cleanup(transport.CloseIdleConnections)

	client := &http.Client{Transport: registryConnectionRoundTripper(func(req *http.Request) (*http.Response, error) {
		response, err := transport.RoundTrip(req)
		if req.Method == http.MethodGet && err == nil {
			streams <- req.Header.Get("Authorization")
		}

		return response, err
	})}
	registry, err := NewRegistry("peer", peer.URL, WithCredentials(registryContextCredentials), WithRegistryHTTPClient(client), WithRegistryMaxToolCacheTTL(time.Hour))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, registry.Close()) })
	first := registryCaller(t.Context(), "alice", "alice:one")
	_, err = registry.Snapshot(first)
	require.NoError(t, err)

	select {
	case auth := <-streams:
		require.Equal(t, "Bearer alice:one", auth)
	case <-time.After(5 * time.Second):
		t.Fatal("initial legacy SSE stream never opened")
	}

	second := registryCaller(t.Context(), "alice", "alice:two")
	_, err = registry.Snapshot(second)
	require.NoError(t, err)
	require.EqualValues(t, 1, lists.Load(), "updating trusted caller context must not invalidate the cached catalog")
	registry.mu.Lock()
	entry := registry.entries["alice"]
	registry.mu.Unlock()

	generation := entry.generation.Load()

	expired.Store(true)
	peer.CloseClientConnections()

	select {
	case auth := <-streams:
		require.Equal(t, "Bearer alice:two", auth, "the actual SDK SSE reconnect must resolve the current partition caller")
	case <-time.After(10 * time.Second):
		t.Fatal("legacy SSE did not reconnect")
	}

	addRegistryConnectionTool(server, "added")
	require.Eventually(t, func() bool { return entry.generation.Load() > generation }, 5*time.Second, time.Millisecond,
		"the reconnected authenticated stream must still deliver a real server push")

	definitions, err := registry.List(second)
	require.NoError(t, err)
	assert.Contains(t, registryNames(definitions), "peer__added")
	assert.EqualValues(t, 2, lists.Load(), "push must invalidate the otherwise hour-long cached catalog")
}

func addRegistryConnectionTool(server *sdkmcp.Server, name string) {
	server.AddTool(&sdkmcp.Tool{Name: name, InputSchema: map[string]any{"type": "object"}},
		func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
			return &sdkmcp.CallToolResult{}, nil
		})
}

func newRegistryConnectionHTTPTransport(t *testing.T) *http.Transport {
	t.Helper()

	transport, ok := http.DefaultTransport.(*http.Transport)
	require.True(t, ok)

	return transport.Clone()
}

type registryConnectionRoundTripper func(*http.Request) (*http.Response, error)

func (f registryConnectionRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
