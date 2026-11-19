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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

type registryTestClock struct{ nanos atomic.Int64 }

func newRegistryTestClock() *registryTestClock { return new(registryTestClock) }

func (c *registryTestClock) Now() time.Time { return time.Unix(0, c.nanos.Load()) }

func (c *registryTestClock) Advance(d time.Duration) { c.nanos.Add(int64(d)) }

func registryAdvertiseToolTTL(peer *registryPeer, ttl time.Duration, suppressChanges bool) <-chan struct{} {
	sent := make(chan struct{}, 16)

	for _, server := range peer.servers {
		server.AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
			return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
				result, err := next(ctx, method, req)
				if list, ok := result.(*sdkmcp.ListToolsResult); ok && err == nil {
					list.TTLMs = int(ttl.Milliseconds())
				}

				return result, err
			}
		})
		server.AddSendingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
			return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
				if method == "notifications/tools/list_changed" && suppressChanges {
					return nil, nil
				}

				result, err := next(ctx, method, req)
				if method == "notifications/tools/list_changed" && err == nil {
					select {
					case sent <- struct{}{}:
					default:
					}
				}

				return result, err
			}
		})
	}

	return sent
}

func registryToolChangeBarrier(t *testing.T, registry *Registry, partition string, sent <-chan struct{}) func() {
	t.Helper()
	registry.mu.Lock()
	entry := registry.entries[partition]
	registry.mu.Unlock()
	require.NotNil(t, entry)
	generation := entry.generation.Load()

	return func() {
		t.Helper()

		select {
		case <-sent:
		case <-time.After(5 * time.Second):
			t.Fatal("server never pushed a tool-list notification")
		}

		require.Eventually(t, func() bool { return entry.generation.Load() > generation }, 5*time.Second, time.Millisecond,
			"client must process the wire notification before discovery resumes")
	}
}

func TestContextRegistryPushInvalidatesCatalog(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()
			peer := newRegistryPeer(t, legacy)
			sent := registryAdvertiseToolTTL(peer, time.Hour, false)
			registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(time.Hour))
			alice := registryCaller(t.Context(), "alice", "alice:one")
			bob := registryCaller(t.Context(), "bob", "bob:one")
			old, err := registry.Snapshot(alice)
			require.NoError(t, err)
			bobBefore, err := registry.List(bob)
			require.NoError(t, err)
			_, err = registry.Snapshot(alice)
			require.NoError(t, err)

			before := len(peer.requests("tools/list"))
			require.Equal(t, 2, before, "a live server TTL must actually cache both catalogs")

			waitForChange := registryToolChangeBarrier(t, registry, "alice", sent)

			peer.addTool("alice", "alice_only", "updated", "new")
			peer.servers["alice"].RemoveTools("shared")
			peer.addTool("alice", "added", "new tool", "value")
			waitForChange()

			current, err := registry.Snapshot(alice)
			require.NoError(t, err)
			assert.Len(t, peer.requests("tools/list"), before+1, "the next snapshot must refetch despite the hour-long TTL")

			definitions, err := current.List(alice)
			require.NoError(t, err)
			assert.Equal(t, []string{"remote__added", "remote__alice_only"}, registryNames(definitions))
			require.Len(t, definitions, 2)
			assert.Equal(t, "updated", definitions[1].Description)
			assert.Contains(t, string(definitions[1].Parameters), `"new"`)

			frozen, err := old.List(alice)
			require.NoError(t, err)
			assert.Equal(t, []string{"remote__alice_only", "remote__shared"}, registryNames(frozen))
			require.Len(t, frozen, 2)
			assert.Equal(t, "original", frozen[0].Description)
			assert.Contains(t, string(frozen[0].Parameters), `"old"`)

			_, err = current.Get(alice, "remote__shared")
			require.ErrorIs(t, err, tool.ErrToolNotFound)
			bobAfter, err := registry.List(bob)
			require.NoError(t, err)
			assert.Equal(t, bobBefore, bobAfter)
			assert.Len(t, peer.requests("tools/list"), before+1, "Alice's push must not invalidate Bob's cached catalog")

			if legacy {
				assert.Len(t, peer.requests("initialize"), 2)

				for _, req := range peer.requests("tools/list") {
					assert.NotEmpty(t, req.session)
					assert.Contains(t, []string{"Bearer alice:one", "Bearer bob:one"}, req.auth)
				}
			} else {
				assert.Empty(t, peer.requests("initialize"))
				assert.Len(t, peer.requests(registryBenchmarkDiscoverMethod), 2)
				listens := peer.requests("subscriptions/listen")
				require.Len(t, listens, 2)

				for _, req := range listens {
					assert.Empty(t, req.session)
					assert.Contains(t, []string{"Bearer alice:one", "Bearer bob:one"}, req.auth)
				}
			}
		})
	}
}

func TestContextRegistryToolCacheTTLBounds(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name      string
		serverTTL time.Duration
		maxTTL    time.Duration
		advance   time.Duration
		cached    bool
	}{
		{"local cap", time.Hour, time.Second, time.Second, true},
		{"server cap", time.Second, time.Hour, time.Second, true},
		{"disabled", time.Hour, 0, 0, false},
		{"server disables", 0, time.Hour, 0, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			peer := newRegistryPeer(t, false)
			registryAdvertiseToolTTL(peer, test.serverTTL, true)

			clock := newRegistryTestClock()
			registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(test.maxTTL), func(cfg *registryConfig) { cfg.now = clock.Now })
			ctx := registryCaller(t.Context(), "alice", "alice:one")
			_, err := registry.List(ctx)
			require.NoError(t, err)
			_, err = registry.List(ctx)
			require.NoError(t, err)

			before := len(peer.requests("tools/list"))
			if test.cached {
				assert.Equal(t, 1, before)
			} else {
				assert.Equal(t, 2, before, "zero TTL must disable even SDK-side caching")
			}

			peer.addTool("alice", "added", "new", "value")
			clock.Advance(test.advance)

			definitions, err := registry.List(ctx)
			require.NoError(t, err)
			assert.Contains(t, registryNames(definitions), "remote__added")
			assert.Len(t, peer.requests("tools/list"), before+1)
		})
	}
}

func TestContextRegistryDefaultToolCacheTTL(t *testing.T) {
	t.Parallel()
	peer := newRegistryPeer(t, false)
	registryAdvertiseToolTTL(peer, time.Hour, true)

	clock := newRegistryTestClock()
	registry := newContextRegistry(t, peer, func(cfg *registryConfig) { cfg.now = clock.Now })
	ctx := registryCaller(t.Context(), "alice", "alice:one")
	_, err := registry.List(ctx)
	require.NoError(t, err)

	peer.addTool("alice", "added", "new", "value")
	clock.Advance(30*time.Second - time.Nanosecond)

	beforeExpiry, err := registry.List(ctx)
	require.NoError(t, err)
	assert.NotContains(t, registryNames(beforeExpiry), "remote__added")
	assert.Len(t, peer.requests("tools/list"), 1)

	clock.Advance(time.Nanosecond)

	afterExpiry, err := registry.List(ctx)
	require.NoError(t, err)
	assert.Contains(t, registryNames(afterExpiry), "remote__added")
	assert.Len(t, peer.requests("tools/list"), 2)
}

func TestContextRegistryRejectsNegativeToolCacheTTL(t *testing.T) {
	t.Parallel()
	peer := newRegistryPeer(t, false)

	registry, err := NewRegistry("remote", peer.url, WithCredentials(registryContextCredentials), WithRegistryMaxToolCacheTTL(-time.Nanosecond))
	if registry != nil {
		t.Cleanup(func() { assert.NoError(t, registry.Close()) })
	}

	require.Error(t, err)
	assert.Empty(t, peer.requests(""))
}

func TestContextRegistryPushDuringDiscovery(t *testing.T) {
	t.Parallel()
	peer := newRegistryPeer(t, false)
	sent := registryAdvertiseToolTTL(peer, time.Hour, false)
	captured := make(chan struct{})
	release := make(chan struct{})
	var unblock sync.Once

	t.Cleanup(func() { unblock.Do(func() { close(release) }) })
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
	registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(time.Hour))
	ctx := registryCaller(t.Context(), "alice", "alice:one")
	finished := make(chan error, 1)

	go func() {
		_, err := registry.Snapshot(ctx)
		finished <- err
	}()

	select {
	case <-captured:
	case <-time.After(5 * time.Second):
		t.Fatal("discovery never captured its response")
	}

	waitForChange := registryToolChangeBarrier(t, registry, "alice", sent)

	peer.addTool("alice", "added", "new", "value")
	waitForChange()
	unblock.Do(func() { close(release) })

	select {
	case err := <-finished:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("discovery did not complete")
	}

	definitions, err := registry.List(ctx)
	require.NoError(t, err)
	assert.Contains(t, registryNames(definitions), "remote__added", "an in-flight stale response must not overwrite a push invalidation")
	assert.GreaterOrEqual(t, len(peer.requests("tools/list")), 2)
}
