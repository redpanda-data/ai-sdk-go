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
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegistryCanceledDiscoveryDoesNotPoisonPartition(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRegistryPeer(t, legacy)
			registryAdvertiseToolTTL(peer, time.Hour, true)

			started := make(chan struct{})
			release := make(chan struct{})
			var first atomic.Bool

			peer.servers["alice"].AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
				return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
					result, err := next(ctx, method, req)
					if method == registryParameterListMethod && first.CompareAndSwap(false, true) {
						close(started)

						select {
						case <-release:
						case <-ctx.Done():
							return nil, ctx.Err()
						}
					}

					return result, err
				}
			})
			registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(time.Hour))
			t.Cleanup(func() { close(release) })

			ctx, cancel := context.WithCancel(registryCaller(t.Context(), "alice", "alice:one"))
			defer cancel()

			finished := make(chan error, 1)

			go func() {
				_, err := registry.Snapshot(ctx)
				finished <- err
			}()

			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("discovery never reached the MCP server")
			}

			cancel()

			select {
			case err := <-finished:
				require.ErrorIs(t, err, context.Canceled)
			case <-time.After(5 * time.Second):
				t.Fatal("canceled discovery did not return")
			}

			fresh, stop := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:two"), 5*time.Second)
			defer stop()

			definitions, err := registry.List(fresh)
			require.NoError(t, err)
			assert.Equal(t, []string{"remote__alice_only", "remote__shared"}, registryNames(definitions))

			lists := peer.requests(registryParameterListMethod)
			require.Len(t, lists, 2, "a canceled list must not publish an hour-long cached response")
			assert.Equal(t, "Bearer alice:two", lists[1].auth)

			_, err = registry.List(fresh)
			require.NoError(t, err)
			assert.Len(t, peer.requests(registryParameterListMethod), 2, "successful recovery must populate the cache")
		})
	}
}

func TestRegistryRecoversAfterLegacySessionTermination(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, true)
	registry := newContextRegistry(t, peer)

	ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 5*time.Second)
	defer cancel()

	captured, err := registry.Snapshot(ctx)
	require.NoError(t, err)

	var sessions []*sdkmcp.ServerSession
	for session := range peer.servers["alice"].Sessions() {
		sessions = append(sessions, session)
	}

	require.Len(t, sessions, 1)
	require.NoError(t, sessions[0].Close())

	failed, err := captured.Execute(ctx, registryRequest("remote__alice_only"))
	require.NoError(t, err)
	require.NotNil(t, failed)
	assert.True(t, failed.IsError, "a terminated session must fail rather than execute under different credentials")

	fresh, err := registry.Snapshot(ctx)
	require.NoError(t, err, "the failed session must not remain stuck in the partition pool")
	recovered, err := fresh.Execute(ctx, registryRequest("remote__alice_only"))
	require.NoError(t, err)
	require.NotNil(t, recovered)
	assert.False(t, recovered.IsError, string(recovered.Result))
	assert.Contains(t, string(recovered.Result), "Bearer alice:one alice_only")

	initializations := peer.requests("initialize")
	require.Len(t, initializations, 2)
	assert.NotEqual(t, peer.requests("tools/list")[0].session, peer.requests("tools/list")[1].session)
}

func TestRegistryCapturedToolSurvivesPoolEviction(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRegistryPeer(t, legacy)
			registryAdvertiseToolTTL(peer, time.Hour, true)
			registry := newContextRegistry(t, peer, WithRegistryMaxConnections(1), WithRegistryMaxToolCacheTTL(time.Hour))

			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()

			alice := registryCaller(ctx, "alice", "alice:one")
			captured, err := registry.Snapshot(alice)
			require.NoError(t, err)
			original, err := captured.Get(alice, "remote__alice_only")
			require.NoError(t, err)
			_, err = registry.List(registryCaller(ctx, "bob", "bob:one"))
			require.NoError(t, err)
			peer.addTool("alice", "alice_only", "changed after eviction", "new")

			rotated := registryCaller(ctx, "alice", "alice:two")
			response, err := captured.Execute(rotated, registryRequest("remote__alice_only"))
			require.NoError(t, err)
			require.NotNil(t, response)
			assert.False(t, response.IsError, string(response.Result))
			assert.Contains(t, string(response.Result), "Bearer alice:two alice_only")
			assert.Equal(t, "original", original.Definition().Description)
			assert.Len(t, peer.requests(registryParameterListMethod), 2, "reacquiring a connection for captured execution must not rediscover tools")
			calls := peer.requests("tools/call")
			require.Len(t, calls, 1)
			assert.Equal(t, "Bearer alice:two", calls[0].auth)

			current, err := registry.Get(rotated, "remote__alice_only")
			require.NoError(t, err)
			assert.Equal(t, "changed after eviction", current.Definition().Description)
			assert.Equal(t, "original", original.Definition().Description)
			assert.Len(t, peer.requests(registryParameterListMethod), 3)

			if legacy {
				assert.Len(t, peer.requests("initialize"), 3)
			} else {
				assert.Len(t, peer.requests("server/discover"), 3)
			}
		})
	}
}
