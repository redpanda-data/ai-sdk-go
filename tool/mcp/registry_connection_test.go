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
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const registrySubscriptionMethod = "subscriptions/listen"

func TestRegistrySubscriptionUnavailableRetainsTrackedSession(t *testing.T) {
	t.Parallel()

	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "peer", Version: "1"}, &sdkmcp.ServerOptions{HasTools: true})
	server.AddTool(&sdkmcp.Tool{Name: "echo", InputSchema: map[string]any{"type": "object"}},
		func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
			return &sdkmcp.CallToolResult{}, nil
		})

	handler := sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server }, &sdkmcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
	var discovers, rejected, active atomic.Int32
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		active.Add(1)
		defer active.Add(-1)

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
			if len(body) > 0 {
				if err := json.Unmarshal(body, &message); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)

					return
				}
			}
		}

		switch message.Method {
		case registryBenchmarkDiscoverMethod:
			discovers.Add(1)
		case registrySubscriptionMethod:
			rejected.Add(1)
			http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)

			return
		}

		handler.ServeHTTP(w, req)
	}))
	t.Cleanup(peer.Close)

	for range 3 {
		registry, err := NewRegistry("peer", peer.URL, WithCredentials(registryContextCredentials))
		require.NoError(t, err)
		t.Cleanup(func() { assert.NoError(t, registry.Close()) })

		ctx := registryCaller(t.Context(), "alice", "alice:one")

		for range 3 {
			definitions, err := registry.List(ctx)
			require.NoError(t, err, "SDK v1.8 does not propagate the subscription HTTP 503 from Connect")
			require.Len(t, definitions, 1)
		}

		registry.mu.Lock()
		entryCount := len(registry.entries)
		entry := registry.entries["alice"]
		registry.mu.Unlock()
		require.Equal(t, 1, entryCount)
		require.NotNil(t, entry)
		session := entry.session
		require.NotNil(t, session, "the session returned by Connect must remain tracked")
		require.NoError(t, registry.Close())

		closed := make(chan struct{})

		go func() { _ = session.Wait(); close(closed) }()

		select {
		case <-closed:
		case <-time.After(5 * time.Second):
			t.Fatal("Registry.Close did not drain the retained SDK connection")
		}
	}

	assert.EqualValues(t, 3, discovers.Load(), "each registry reuses its single returned session")
	assert.EqualValues(t, 3, rejected.Load(), "the peer actually returned HTTP 503 to every subscription")
	require.Eventually(t, func() bool { return active.Load() == 0 }, 5*time.Second, time.Millisecond)
}
