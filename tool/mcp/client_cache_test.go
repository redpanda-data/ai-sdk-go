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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

func TestClientManualSyncBypassesServerCatalogTTL(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	registryAdvertiseToolTTL(peer, time.Hour, true)
	client, err := NewClient("old", NewStreamableTransport(peer.url, WithHTTPHeaders(map[string]string{"Authorization": "Bearer alice:one"})))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	require.NoError(t, client.Start(t.Context()))
	assert.Len(t, peer.requests("tools/list"), 1)

	peer.addTool("alice", "added", "new", "value")
	require.NoError(t, client.SyncTools(t.Context()))
	assert.Len(t, peer.requests("tools/list"), 2, "manual SyncTools must bypass positive server TTLs")
	definitions, err := client.ListTools(t.Context())
	require.NoError(t, err)
	assert.Contains(t, registryNames(definitions), "old__added")

	require.NoError(t, client.SyncTools(t.Context()))
	assert.Len(t, peer.requests("tools/list"), 3)
}

func TestClientAutoSyncStillReceivesToolChanges(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	registryAdvertiseToolTTL(peer, time.Hour, false)

	registry := tool.NewRegistry(tool.RegistryConfig{})
	client, err := NewClient("old", NewStreamableTransport(peer.url, WithHTTPHeaders(map[string]string{"Authorization": "Bearer alice:one"})),
		WithRegistry(registry), WithAutoSync(time.Hour))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	require.NoError(t, client.Start(t.Context()))

	peer.addTool("alice", "added", "new", "value")
	require.EventuallyWithT(t, func(collect *assert.CollectT) {
		definitions, err := registry.List(t.Context())
		if assert.NoError(collect, err) {
			assert.Contains(collect, registryNames(definitions), "old__added")
		}
	}, 5*time.Second, time.Millisecond)
	assert.GreaterOrEqual(t, len(peer.requests("tools/list")), 2)
}
