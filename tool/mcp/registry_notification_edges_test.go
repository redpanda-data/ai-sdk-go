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

func newPaginatedRegistryPeer(t *testing.T, legacy bool) *registryPeer {
	t.Helper()

	peer := newRegistryPeer(t, legacy)

	opts := &sdkmcp.ServerOptions{HasTools: true, PageSize: 1}
	if legacy {
		opts.SupportedProtocolVersions = []string{"2025-06-18"}
	}

	peer.servers["alice"] = sdkmcp.NewServer(&sdkmcp.Implementation{Name: "paginated-alice", Version: "1"}, opts)
	peer.addTool("alice", "alice_only", "original", "old")
	peer.addTool("alice", "shared", "shared", "value")
	peer.addTool("alice", "z_last", "last page", "value")

	return peer
}

func registryCallerListCount(peer *registryPeer, bearer string) int {
	count := 0

	for _, req := range peer.requests(registryParameterListMethod) {
		if req.auth == "Bearer "+bearer {
			count++
		}
	}

	return count
}

func TestContextRegistryPushAcrossCatalogPages(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newPaginatedRegistryPeer(t, legacy)
			sent := registryAdvertiseToolTTL(peer, time.Hour, false)
			captured := make(chan struct{})
			release := make(chan struct{})

			var unblock sync.Once
			defer unblock.Do(func() { close(release) })
			var armed atomic.Bool
			var firstPages atomic.Int32

			peer.servers["alice"].AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
				return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
					result, err := next(ctx, method, req)
					if request, ok := req.(*sdkmcp.ListToolsRequest); method == registryParameterListMethod && ok {
						if request.Params == nil || request.Params.Cursor == "" {
							firstPages.Add(1)
						} else if armed.CompareAndSwap(true, false) {
							close(captured)
							<-release
						}
					}

					return result, err
				}
			})
			registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(time.Hour))
			alice := registryCaller(t.Context(), "alice", "alice:one")
			bob := registryCaller(t.Context(), "bob", "bob:one")
			old, err := registry.Snapshot(alice)
			require.NoError(t, err)
			original, err := old.List(alice)
			require.NoError(t, err)
			bobOriginal, err := registry.List(bob)
			require.NoError(t, err)
			assert.Equal(t, []string{"remote__bob_only", "remote__shared"}, registryNames(bobOriginal))

			_, err = registry.List(alice)
			require.NoError(t, err)
			_, err = registry.List(bob)
			require.NoError(t, err)
			require.Equal(t, 3, registryCallerListCount(peer, "alice:one"))
			require.Equal(t, 1, registryCallerListCount(peer, "bob:one"))

			waitForChange := registryToolChangeBarrier(t, registry, "alice", sent)

			peer.addTool("alice", "early", "starts refresh", "value")
			waitForChange()
			armed.Store(true)

			finished := make(chan struct {
				registry tool.Registry
				err      error
			}, 1)

			go func() {
				snapshot, err := registry.Snapshot(alice)
				finished <- struct {
					registry tool.Registry
					err      error
				}{snapshot, err}
			}()

			select {
			case <-captured:
			case <-time.After(5 * time.Second):
				t.Fatal("a later catalog page never reached the peer")
			}

			waitForChange = registryToolChangeBarrier(t, registry, "alice", sent)

			peer.addTool("alice", "alice_only", "updated during pagination", "new")
			peer.servers["alice"].RemoveTools("shared")
			peer.addTool("alice", "added", "new first page", "value")
			waitForChange()
			unblock.Do(func() { close(release) })

			var current tool.Registry

			select {
			case result := <-finished:
				require.NoError(t, result.err)
				current = result.registry
			case <-time.After(5 * time.Second):
				t.Fatal("paginated discovery failed to complete after push")
			}

			definitions, err := current.List(alice)
			require.NoError(t, err)
			assert.Equal(t, []string{"remote__added", "remote__alice_only", "remote__early", "remote__z_last"}, registryNames(definitions))
			require.Len(t, definitions, 4)
			assert.Equal(t, "updated during pagination", definitions[1].Description)
			assert.Contains(t, string(definitions[1].Parameters), `"new"`)
			assert.GreaterOrEqual(t, firstPages.Load(), int32(3), "the interrupted catalog must restart from its first page")

			frozen, err := old.List(alice)
			require.NoError(t, err)
			assert.Equal(t, original, frozen)

			again, err := registry.List(alice)
			require.NoError(t, err)
			assert.Equal(t, definitions, again, "a later cache read must not resurrect a mixed-page catalog")

			bobAfter, err := registry.List(bob)
			require.NoError(t, err)
			assert.Equal(t, bobOriginal, bobAfter)
			assert.Equal(t, 1, registryCallerListCount(peer, "bob:one"))
			beforeCalls := len(peer.requests("tools/call"))
			response, err := current.Execute(alice, registryRequest("remote__shared"))
			require.NoError(t, err)
			require.NotNil(t, response)
			assert.True(t, response.IsError)
			assert.Len(t, peer.requests("tools/call"), beforeCalls, "a tool removed between pages must not dispatch")

			response, err = current.Execute(alice, registryRequest("remote__added"))
			require.NoError(t, err)
			require.NotNil(t, response)
			assert.False(t, response.IsError, string(response.Result))
			assert.Contains(t, string(response.Result), "Bearer alice:one added")
		})
	}
}

func TestContextRegistryEmptyCatalogThenNotificationBurst(t *testing.T) {
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
			original, err := old.List(alice)
			require.NoError(t, err)
			bobOriginal, err := registry.List(bob)
			require.NoError(t, err)
			assert.Equal(t, []string{"remote__bob_only", "remote__shared"}, registryNames(bobOriginal))

			_, err = registry.List(alice)
			require.NoError(t, err)
			_, err = registry.List(bob)
			require.NoError(t, err)
			require.Len(t, peer.requests(registryParameterListMethod), 2)

			waitForChange := registryToolChangeBarrier(t, registry, "alice", sent)

			peer.servers["alice"].RemoveTools("alice_only", "shared")
			waitForChange()

			empty, err := registry.Snapshot(alice)
			require.NoError(t, err)
			definitions, err := empty.List(alice)
			require.NoError(t, err)
			assert.Empty(t, definitions)
			definitions, err = registry.List(alice)
			require.NoError(t, err)
			assert.Empty(t, definitions)
			require.Equal(t, 2, registryCallerListCount(peer, "alice:one"), "an empty catalog must be a genuine cache hit before re-add")

			waitForBobChange := registryToolChangeBarrier(t, registry, "bob", sent)

			peer.addTool("bob", "bob_added", "Bob's catalog change", "value")
			waitForBobChange()

			bobOriginal, err = registry.List(bob)
			require.NoError(t, err)
			assert.Contains(t, registryNames(bobOriginal), "remote__bob_added")

			definitions, err = registry.List(alice)
			require.NoError(t, err)
			assert.Empty(t, definitions)
			assert.Equal(t, 2, registryCallerListCount(peer, "alice:one"), "Bob's push must not invalidate Alice's cached empty catalog")

			waitForChange = registryToolChangeBarrier(t, registry, "alice", sent)

			for i := range 8 {
				name := fmt.Sprintf("transient_%d", i)
				peer.addTool("alice", name, "transient", "value")
				peer.servers["alice"].RemoveTools(name)
				peer.addTool("alice", "restored", fmt.Sprintf("revision %d", i), fmt.Sprintf("version_%d", i))
			}

			waitForChange()

			current, err := registry.Snapshot(alice)
			require.NoError(t, err)
			definitions, err = current.List(alice)
			require.NoError(t, err)
			require.Len(t, definitions, 1)
			assert.Equal(t, "remote__restored", definitions[0].Name)
			assert.Equal(t, "revision 7", definitions[0].Description)
			assert.Contains(t, string(definitions[0].Parameters), `"version_7"`)

			frozen, err := old.List(alice)
			require.NoError(t, err)
			assert.Equal(t, original, frozen)

			stillEmpty, err := empty.List(alice)
			require.NoError(t, err)
			assert.Empty(t, stillEmpty)

			_, err = current.Get(alice, "remote__transient_7")
			require.ErrorIs(t, err, tool.ErrToolNotFound)

			for _, snapshot := range []tool.Registry{empty, current} {
				response, err := snapshot.Execute(alice, registryRequest("remote__shared"))
				require.NoError(t, err)
				require.NotNil(t, response)
				assert.True(t, response.IsError)
			}

			assert.Empty(t, peer.requests("tools/call"))

			response, err := current.Execute(alice, registryRequest("remote__restored"))
			require.NoError(t, err)
			require.NotNil(t, response)
			assert.False(t, response.IsError, string(response.Result))
			assert.Contains(t, string(response.Result), "Bearer alice:one restored")

			bobAfter, err := registry.List(bob)
			require.NoError(t, err)
			assert.Equal(t, bobOriginal, bobAfter)
			assert.Equal(t, 2, registryCallerListCount(peer, "bob:one"), "Alice's burst must leave Bob's refreshed cache intact")
		})
	}
}

func TestContextRegistryCatalogUsesShortestPageTTL(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		for _, lastTTL := range []time.Duration{0, time.Second} {
			t.Run(fmt.Sprintf("legacy=%t/lastTTL=%s", legacy, lastTTL), func(t *testing.T) {
				t.Parallel()

				peer := newPaginatedRegistryPeer(t, legacy)
				registryAdvertiseToolTTL(peer, time.Hour, true)
				peer.servers["alice"].AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
					return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
						result, err := next(ctx, method, req)
						if list, ok := result.(*sdkmcp.ListToolsResult); ok && list.NextCursor == "" {
							list.TTLMs = int(lastTTL.Milliseconds())
						}

						return result, err
					}
				})

				clock := newRegistryTestClock()
				registry := newContextRegistry(t, peer, WithRegistryMaxToolCacheTTL(time.Hour), func(cfg *registryConfig) { cfg.now = clock.Now })
				alice := registryCaller(t.Context(), "alice", "alice:one")
				bob := registryCaller(t.Context(), "bob", "bob:one")
				_, err := registry.List(alice)
				require.NoError(t, err)
				_, err = registry.List(bob)
				require.NoError(t, err)
				require.Equal(t, 3, registryCallerListCount(peer, "alice:one"))

				peer.addTool("alice", "alice_only", "changed after pagination", "new")

				if lastTTL > 0 {
					clock.Advance(lastTTL - time.Nanosecond)

					cached, err := registry.List(alice)
					require.NoError(t, err)
					require.Len(t, cached, 3)
					assert.Equal(t, "original", cached[0].Description)
					assert.Equal(t, 3, registryCallerListCount(peer, "alice:one"))
					clock.Advance(time.Nanosecond)
				}

				fresh, err := registry.List(alice)
				require.NoError(t, err)
				require.Len(t, fresh, 3)
				assert.Equal(t, "changed after pagination", fresh[0].Description)
				assert.Contains(t, string(fresh[0].Parameters), `"new"`)
				assert.Equal(t, 6, registryCallerListCount(peer, "alice:one"), "the last page's TTL must bound the entire catalog")

				_, err = registry.List(bob)
				require.NoError(t, err)
				assert.Equal(t, 1, registryCallerListCount(peer, "bob:one"))
			})
		}
	}
}
