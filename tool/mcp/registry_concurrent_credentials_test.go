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
	"encoding/json"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const registryConnectionCallMethod = "tools/call"

func TestRegistryConcurrentSamePartitionRequestsRetainCallerCredentials(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	prepared := make(chan struct{})
	release := make(chan struct{})
	var unblock sync.Once

	transport := newRegistryConnectionHTTPTransport(t)
	t.Cleanup(transport.CloseIdleConnections)

	client := &http.Client{Transport: registryConnectionRoundTripper(func(req *http.Request) (*http.Response, error) {
		if req.Header.Get("Mcp-Method") == registryConnectionCallMethod && req.Header.Get("Authorization") == "Bearer alice:one" {
			close(prepared)

			select {
			case <-release:
			case <-req.Context().Done():
				return nil, req.Context().Err()
			}
		}

		return transport.RoundTrip(req)
	})}
	registry := newContextRegistry(t, peer, WithRegistryHTTPClient(client))
	t.Cleanup(func() { unblock.Do(func() { close(release) }) })
	first := registryCaller(t.Context(), "alice", "alice:one")
	second := registryCaller(t.Context(), "alice", "alice:two")
	captured, err := registry.Get(first, "remote__shared")
	require.NoError(t, err)
	type outcome struct {
		result json.RawMessage
		err    error
	}
	one := make(chan outcome, 1)

	go func() {
		result, err := captured.Execute(first, nil)
		one <- outcome{result: result, err: err}
	}()

	select {
	case <-prepared:
	case <-time.After(5 * time.Second):
		t.Fatal("first caller never prepared its HTTP request")
	}

	two := make(chan outcome, 1)

	go func() {
		result, err := captured.Execute(second, nil)
		two <- outcome{result: result, err: err}
	}()

	select {
	case response := <-two:
		require.NoError(t, response.err)
		assert.Contains(t, string(response.result), "Bearer alice:two")
		assert.NotContains(t, string(response.result), "Bearer alice:one")
	case <-time.After(5 * time.Second):
		t.Fatal("second caller could not execute while the first request was paused")
	}

	registry.mu.Lock()
	entry := registry.entries["alice"]
	registry.mu.Unlock()

	latest, err := registryContextCredentials(entry.identity.Load().context)
	require.NoError(t, err)
	require.Equal(t, "Bearer alice:two", latest.Headers.Get("Authorization"))
	unblock.Do(func() { close(release) })

	select {
	case response := <-one:
		require.NoError(t, response.err)
		assert.Contains(t, string(response.result), "Bearer alice:one")
		assert.NotContains(t, string(response.result), "Bearer alice:two")
	case <-time.After(5 * time.Second):
		t.Fatal("first caller did not finish after release")
	}

	calls := peer.requests(registryConnectionCallMethod)
	require.Len(t, calls, 2)
	assert.Equal(t, "Bearer alice:two", calls[0].auth)
	assert.Equal(t, "Bearer alice:one", calls[1].auth, "the first request must keep its own credentials even after the shared background identity changes")
}
