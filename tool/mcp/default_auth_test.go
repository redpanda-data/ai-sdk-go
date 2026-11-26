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
	"sync"
	"testing"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDefaultHTTPClientsSendNoAuthorization pins the default wire behavior
// that the opt-in auth/workload DPoP transport must not change: with default
// options, MCP HTTP requests carry no Authorization or DPoP header and go
// only to the MCP endpoint, with no token request anywhere.
func TestDefaultHTTPClientsSendNoAuthorization(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		use  func(t *testing.T, endpoint string)
	}{
		{
			name: "contextual registry",
			use: func(t *testing.T, endpoint string) {
				t.Helper()

				registry, err := NewRegistry("peer", endpoint, WithCredentials(func(context.Context) (Credentials, error) {
					return NewCredentials(WithPartitionKey("public"))
				}))
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, registry.Close()) })

				definitions, err := registry.List(t.Context())
				require.NoError(t, err)
				require.Len(t, definitions, 1)

				_, err = registry.Execute(t.Context(), cleanupRequest())
				require.NoError(t, err)
			},
		},
		{
			name: "legacy client",
			use: func(t *testing.T, endpoint string) {
				t.Helper()

				client, err := NewClient("peer", NewStreamableTransport(endpoint))
				require.NoError(t, err)
				require.NoError(t, client.Start(t.Context()))
				t.Cleanup(func() { assert.NoError(t, client.Close()) })

				definitions, err := client.ListTools(t.Context())
				require.NoError(t, err)
				require.Len(t, definitions, 1)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "peer", Version: "1"}, &sdkmcp.ServerOptions{HasTools: true})
			addRegistryConnectionTool(server, cleanupToolName)

			handler := sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server }, &sdkmcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})

			type seen struct {
				path   string
				header http.Header
			}

			var mu sync.Mutex
			var requests []seen
			peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				mu.Lock()

				requests = append(requests, seen{path: req.URL.Path, header: req.Header.Clone()})
				mu.Unlock()

				handler.ServeHTTP(w, req)
			}))
			t.Cleanup(peer.Close)

			tt.use(t, peer.URL+"/mcp")

			mu.Lock()
			defer mu.Unlock()

			require.NotEmpty(t, requests)

			for _, req := range requests {
				assert.Equal(t, "/mcp", req.path)
				assert.Empty(t, req.header.Values("Authorization"))
				assert.Empty(t, req.header.Values("DPoP"))
			}
		})
	}
}
