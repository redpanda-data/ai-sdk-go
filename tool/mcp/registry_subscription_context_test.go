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
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegistryModernSubscriptionContextUsesLatestValidatedCaller(t *testing.T) {
	t.Parallel()

	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "peer", Version: "1"}, &sdkmcp.ServerOptions{HasTools: true})
	addRegistryConnectionTool(server, "echo")

	handler := sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server }, &sdkmcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
	var expired atomic.Bool
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if expired.Load() && req.Header.Get("Authorization") != "Bearer alice:two" {
			http.Error(w, "expired bearer", http.StatusUnauthorized)

			return
		}

		handler.ServeHTTP(w, req)
	}))
	t.Cleanup(peer.Close)

	subscriptions := make(chan *http.Request, 8)
	transport := newRegistryConnectionHTTPTransport(t)
	t.Cleanup(transport.CloseIdleConnections)

	client := &http.Client{Transport: registryConnectionRoundTripper(func(req *http.Request) (*http.Response, error) {
		if req.Method == http.MethodPost && req.GetBody != nil {
			body, err := req.GetBody()
			if err != nil {
				return nil, err
			}

			data, err := io.ReadAll(body)
			_ = body.Close()

			if err != nil {
				return nil, err
			}

			if strings.Contains(string(data), `"method":"subscriptions/listen"`) {
				subscriptions <- req.Clone(req.Context())
			}
		}

		return transport.RoundTrip(req)
	})}
	registry, err := NewRegistry("peer", peer.URL, WithCredentials(registryContextCredentials), WithRegistryHTTPClient(client))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, registry.Close()) })
	first := registryCaller(t.Context(), "alice", "alice:one")
	captured, err := registry.Snapshot(first)
	require.NoError(t, err)

	var subscription *http.Request

	select {
	case subscription = <-subscriptions:
	case <-time.After(5 * time.Second):
		t.Fatal("the SDK never opened its modern subscription")
	}

	second := registryCaller(t.Context(), "alice", "alice:two")
	_, err = registry.Snapshot(second)
	require.NoError(t, err)
	frozen, err := captured.Get(second, "peer__echo")
	require.NoError(t, err)
	_, err = frozen.Execute(registryCaller(t.Context(), "bob", "bob:token"), nil)
	require.ErrorIs(t, err, ErrPartitionMismatch, "an untrusted partition cannot update subscription credentials")
	expired.Store(true)

	// v1.8 does not automatically resume modern POST subscriptions without
	// event IDs. Reissue the actual SDK request with its retained context to
	// exercise the same per-request credential resolution on the wire.
	reissued := subscription.Clone(registryRetainedSubscriptionContext(t, subscription))
	reissued.Body, err = subscription.GetBody()
	require.NoError(t, err)
	response, err := registry.httpClient("alice").Do(reissued)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, response.Body.Close()) })
	require.Equal(t, http.StatusOK, response.StatusCode)
	scanner := bufio.NewScanner(response.Body)
	var acknowledgment string

	for scanner.Scan() {
		if strings.Contains(scanner.Text(), "notifications/subscriptions/acknowledged") {
			acknowledgment = scanner.Text()

			break
		}
	}

	require.NoError(t, scanner.Err())
	assert.NotEmpty(t, acknowledgment, "the real modern MCP peer must accept the reissued authenticated subscription")

	ordinary := subscription.Clone(first)
	ordinary.Body, err = subscription.GetBody()
	require.NoError(t, err)
	rejected, err := registry.httpClient("alice").Do(ordinary)
	require.NoError(t, err)
	require.NoError(t, rejected.Body.Close())
	assert.Equal(t, http.StatusUnauthorized, rejected.StatusCode, "ordinary requests retain their own caller context, not the partition background context")
}

func registryRetainedSubscriptionContext(t *testing.T, req *http.Request) context.Context {
	t.Helper()

	frozen, ok := req.Context().(registryRequestIdentityContext)
	require.True(t, ok, "the custom transport must receive a per-request frozen caller context")

	return frozen.Context
}
