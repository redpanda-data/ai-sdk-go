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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// discoverRejectSessionID is the session ID that the stand-in server puts on
// its server/discover rejection. It only has to differ from the session ID the
// real handler mints for initialize, which is random.
const discoverRejectSessionID = "session-from-the-discover-rejection"

// newDiscoverRejectingServer starts an MCP server over streamable HTTP that
// rejects server/discover the way a server behind the AI Gateway does: HTTP
// 200, a JSON-RPC error, and a Mcp-Session-Id header. Every other request goes
// to a real MCP handler, which mints its own session ID on initialize.
//
// The two session IDs are what the client trips over, so the test needs a
// server that answers discover and initialize with different ones. A plain
// HTTP 400 rejection, which is what the MCP Python SDK sends on its own, does
// not reproduce it: the client drops the response before it reads the header.
func newDiscoverRejectingServer(t *testing.T) string {
	t.Helper()

	server := sdkmcp.NewServer(&sdkmcp.Implementation{
		Name:    "discover-rejecting-server",
		Version: "v1.0.0",
	}, nil)
	sdkmcp.AddTool(server, &sdkmcp.Tool{
		Name:        "ping",
		Description: "Returns pong.",
	}, func(_ context.Context, _ *sdkmcp.CallToolRequest, _ struct{}) (*sdkmcp.CallToolResult, any, error) {
		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "pong"}},
		}, nil, nil
	})

	handler := sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server {
		return server
	}, nil)

	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)

			return
		}

		var envelope struct {
			Method string          `json:"method"`
			ID     json.RawMessage `json:"id"`
		}
		_ = json.Unmarshal(body, &envelope)

		if envelope.Method == "server/discover" {
			if len(envelope.ID) == 0 {
				envelope.ID = json.RawMessage("null")
			}

			reject := fmt.Sprintf(
				`{"jsonrpc":"2.0","id":%s,"error":{"code":-32600,"message":"Bad Request: Missing session ID"}}`,
				envelope.ID,
			)

			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Mcp-Session-Id", discoverRejectSessionID)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(reject))

			return
		}

		if envelope.Method == "initialize" {
			// A gateway answers initialize with a session of its own, whatever
			// session ID the client carries over from the discover rejection.
			// The real handler would answer a stale session ID with HTTP 404
			// instead, which is a different failure.
			r.Header.Del("Mcp-Session-Id")
		}

		r.Body = io.NopCloser(bytes.NewReader(body))
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(httpServer.Close)

	return httpServer.URL
}

// TestIntegrationConnectToDiscoverRejectingServer checks that the client
// reaches a usable session against a server that rejects server/discover with
// a session ID of its own.
//
// Without the retry in connect, Start fails with `mismatching session IDs` and
// the agent never makes its first model call: every tool on that server is
// unreachable.
func TestIntegrationConnectToDiscoverRejectingServer(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	endpoint := newDiscoverRejectingServer(t)

	client, err := NewClient("discover-rejecting-server", NewStreamableTransport(endpoint))
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)

	require.NoError(t, client.Start(ctx))
	t.Cleanup(func() {
		assert.NoError(t, client.Close())
	})

	tools, err := client.ListTools(ctx)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	assert.Equal(t, "discover-rejecting-server__ping", tools[0].Name)
}
