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
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type registryIdentityPartKey int

func TestRegistryBackgroundCredentialResolutionUsesOneCallerContext(t *testing.T) {
	t.Parallel()

	firstRead := make(chan struct{})
	release := make(chan struct{})
	var unblock sync.Once
	var armed atomic.Bool
	provider := func(ctx context.Context) (Credentials, error) {
		prefix, ok := ctx.Value(registryIdentityPartKey(0)).(string)
		if !ok {
			return Credentials{}, errors.New("missing caller prefix")
		}

		if prefix == "first" && armed.CompareAndSwap(true, false) {
			close(firstRead)

			select {
			case <-release:
			case <-ctx.Done():
				return Credentials{}, ctx.Err()
			}
		}

		suffix, ok := ctx.Value(registryIdentityPartKey(1)).(string)
		if !ok {
			return Credentials{}, errors.New("missing caller suffix")
		}

		return Credentials{PartitionKey: "shared", Headers: http.Header{"Authorization": {"Bearer " + prefix + ":" + suffix}}}, nil
	}
	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "peer", Version: "1"}, &sdkmcp.ServerOptions{HasTools: true})
	addRegistryConnectionTool(server, "echo")

	handler := sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server }, &sdkmcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		auth := req.Header.Get("Authorization")
		if auth != "Bearer first:first" && auth != "Bearer second:second" {
			http.Error(w, "mixed caller credentials", http.StatusUnauthorized)

			return
		}

		handler.ServeHTTP(w, req)
	}))
	t.Cleanup(peer.Close)

	subscriptions := make(chan *http.Request, 1)
	downstream := make(chan [3]string, 1)
	transport := newRegistryConnectionHTTPTransport(t)
	t.Cleanup(transport.CloseIdleConnections)

	client := &http.Client{Transport: registryConnectionRoundTripper(func(req *http.Request) (*http.Response, error) {
		if req.Header.Get("Mcp-Method") == registrySubscriptionMethod {
			select {
			case subscriptions <- req.Clone(req.Context()):
			default:
			}
		}

		if req.Header.Get("X-Registry-Identity-Probe") == "true" {
			prefix, _ := req.Context().Value(registryIdentityPartKey(0)).(string)
			suffix, _ := req.Context().Value(registryIdentityPartKey(1)).(string)

			downstream <- [3]string{prefix, suffix, req.Header.Get("Authorization")}
		}

		return transport.RoundTrip(req)
	})}
	registry, err := NewRegistry("peer", peer.URL, WithCredentials(provider), WithRegistryHTTPClient(client))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, registry.Close()) })
	t.Cleanup(func() { unblock.Do(func() { close(release) }) })
	first := context.WithValue(context.WithValue(t.Context(), registryIdentityPartKey(0), "first"), registryIdentityPartKey(1), "first")
	_, err = registry.Snapshot(first)
	require.NoError(t, err)

	var subscription *http.Request
	select {
	case subscription = <-subscriptions:
	case <-time.After(5 * time.Second):
		t.Fatal("SDK did not open its real subscription request")
	}

	reissued := subscription.Clone(registryRetainedSubscriptionContext(t, subscription))
	reissued.Header.Set("X-Registry-Identity-Probe", "true")
	reissued.Body, err = subscription.GetBody()
	require.NoError(t, err)
	armed.Store(true)
	type outcome struct {
		status int
		err    error
	}
	finished := make(chan outcome, 1)

	go func() {
		response, err := registry.httpClient("shared").Do(reissued)
		status := 0

		if response != nil {
			status = response.StatusCode
			err = errors.Join(err, response.Body.Close())
		}

		finished <- outcome{status: status, err: err}
	}()

	select {
	case <-firstRead:
	case <-time.After(5 * time.Second):
		t.Fatal("background provider did not reach its first caller-value read")
	}

	second := context.WithValue(context.WithValue(t.Context(), registryIdentityPartKey(0), "second"), registryIdentityPartKey(1), "second")
	_, err = registry.Snapshot(second)
	require.NoError(t, err)
	unblock.Do(func() { close(release) })

	select {
	case result := <-finished:
		require.NoError(t, result.err)
		assert.Equal(t, http.StatusOK, result.status, "one request must not combine the first caller's prefix with the second caller's suffix")
	case <-time.After(5 * time.Second):
		t.Fatal("background request did not finish after its provider was released")
	}

	select {
	case observed := <-downstream:
		assert.Equal(t, [3]string{"first", "first", "Bearer first:first"}, observed,
			"the custom HTTP transport must receive the same frozen caller context as the credentials provider")
	default:
		t.Fatal("the real HTTP request never reached the custom transport")
	}
}
