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
	"testing"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

// TestExecuteToolForwardsLLMToolCallID verifies that when the execution context
// carries an LLM tool-call id (as the Registry sets via tool.WithCallID),
// ExecuteTool forwards it to the server in the tools/call request _meta under
// MetaKeyLLMToolCallID. Without an id on the context, no such _meta key is sent.
func TestExecuteToolForwardsLLMToolCallID(t *testing.T) {
	t.Parallel()

	t.Run("forwards id from context", func(t *testing.T) {
		t.Parallel()

		ctx := t.Context()

		server := newMockMCPServer()
		transport, err := server.start(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { _ = server.stop() })

		client, err := NewClient("test-server", func() (sdkmcp.Transport, error) { return transport, nil })
		require.NoError(t, err)
		require.NoError(t, client.Start(ctx))
		t.Cleanup(func() { _ = client.Close() })

		callCtx := tool.WithCallID(ctx, "call_abc123")

		_, err = client.ExecuteTool(callCtx, "test-server__echo", json.RawMessage(`{"message":"hi"}`))
		require.NoError(t, err)

		meta := server.lastMeta()
		require.NotNil(t, meta, "server should have received _meta")
		assert.Equal(t, "call_abc123", meta[MetaKeyLLMToolCallID],
			"the originating LLM tool-call id must be forwarded under MetaKeyLLMToolCallID")
	})

	t.Run("no _meta key when context has no id", func(t *testing.T) {
		t.Parallel()

		ctx := t.Context()

		server := newMockMCPServer()
		transport, err := server.start(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { _ = server.stop() })

		client, err := NewClient("test-server", func() (sdkmcp.Transport, error) { return transport, nil })
		require.NoError(t, err)
		require.NoError(t, client.Start(ctx))
		t.Cleanup(func() { _ = client.Close() })

		// No tool.WithCallID on the context.
		_, err = client.ExecuteTool(ctx, "test-server__echo", json.RawMessage(`{"message":"hi"}`))
		require.NoError(t, err)

		_, present := server.lastMeta()[MetaKeyLLMToolCallID]
		assert.False(t, present, "no tool-call id should be sent when the context carries none")
	})
}
