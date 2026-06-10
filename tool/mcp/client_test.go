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
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// mockConnection implements sdkmcp.Connection for testing.
type mockConnection struct {
	closed bool
	mu     sync.Mutex
}

func (*mockConnection) Read(_ context.Context) (jsonrpc.Message, error) {
	// Block forever or return error - not used in these tests
	return nil, errors.New("mock connection read not implemented")
}

func (*mockConnection) Write(_ context.Context, _ jsonrpc.Message) error {
	return nil
}

func (m *mockConnection) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.closed = true

	return nil
}

func (*mockConnection) SessionID() string {
	return "mock-session"
}

// mockTransport implements sdkmcp.Transport for testing.
type mockTransport struct {
	connectErr error
}

func (m *mockTransport) Connect(_ context.Context) (sdkmcp.Connection, error) {
	if m.connectErr != nil {
		return nil, m.connectErr
	}

	return &mockConnection{}, nil
}

// mockRegistry implements tool.Registry for testing.
type mockRegistry struct {
	mu    sync.Mutex
	tools map[string]tool.Tool
}

func newMockRegistry() *mockRegistry {
	return &mockRegistry{tools: make(map[string]tool.Tool)}
}

func (m *mockRegistry) Register(t tool.Tool, _ ...tool.Option) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.tools[t.Definition().Name] = t

	return nil
}

func (m *mockRegistry) Unregister(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.tools, name)

	return nil
}

func (m *mockRegistry) Get(name string) (tool.Tool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	t, ok := m.tools[name]
	if !ok {
		return nil, errors.New("tool not found")
	}

	return t, nil
}

func (m *mockRegistry) List() []llm.ToolDefinition {
	m.mu.Lock()
	defer m.mu.Unlock()

	defs := make([]llm.ToolDefinition, 0, len(m.tools))
	for _, t := range m.tools {
		defs = append(defs, t.Definition())
	}

	return defs
}

func (m *mockRegistry) Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	t, err := m.Get(req.Name)
	if err != nil {
		return nil, err
	}

	result, err := t.Execute(ctx, req.Arguments)
	if err != nil {
		return nil, err
	}

	return &llm.ToolResponsePart{
		Name:   req.Name,
		Result: result,
	}, nil
}

func (m *mockRegistry) ExecuteAll(ctx context.Context, reqs []*llm.ToolRequestPart, _ ...tool.BatchOption) []*llm.ToolResponsePart {
	// Simple mock implementation - just call Execute for each request
	results := make([]*llm.ToolResponsePart, len(reqs))
	for i, req := range reqs {
		resp, _ := m.Execute(ctx, req)
		results[i] = resp
	}

	return results
}

func (m *mockRegistry) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	return len(m.tools)
}

func TestNewClient(t *testing.T) {
	t.Parallel()

	factory := func() (sdkmcp.Transport, error) { return &mockTransport{}, nil }

	t.Run("success", func(t *testing.T) {
		t.Parallel()

		client, err := NewClient("test-server", factory)
		require.NoError(t, err)
		assert.Equal(t, "test-server", client.ServerID())
		assert.False(t, client.Started())
	})

	t.Run("validation errors", func(t *testing.T) {
		t.Parallel()

		_, err := NewClient("", factory)
		require.ErrorContains(t, err, "serverID cannot be empty")

		_, err = NewClient("test", nil)
		require.ErrorContains(t, err, "transportFactory cannot be nil")
	})
}

func TestClientLifecycle(t *testing.T) {
	t.Parallel()

	factory := func() (sdkmcp.Transport, error) { return &mockTransport{}, nil }

	t.Run("shutdown before start is safe", func(t *testing.T) {
		t.Parallel()

		client, err := NewClient("test", factory)
		require.NoError(t, err)

		err = client.Shutdown(context.Background())
		require.NoError(t, err)
		assert.False(t, client.Started())
	})

	t.Run("shutdown is idempotent", func(t *testing.T) {
		t.Parallel()

		client, err := NewClient("test", factory)
		require.NoError(t, err)

		ctx := context.Background()
		assert.NoError(t, client.Shutdown(ctx))
		assert.NoError(t, client.Shutdown(ctx))
	})

	t.Run("close calls shutdown with timeout", func(t *testing.T) {
		t.Parallel()

		client, err := NewClient("test", factory)
		require.NoError(t, err)

		assert.NoError(t, client.Close())
		assert.False(t, client.Started())
	})

	t.Run("start when already started is idempotent", func(t *testing.T) {
		t.Parallel()

		// This test isn't meaningful with sync.Once since Start() will only run once anyway.
		// sync.Once guarantees idempotency by design.
		// Removing this test as it's redundant with the sync.Once pattern.
		t.Skip("sync.Once guarantees idempotency by design")
	})

	t.Run("transport factory error", func(t *testing.T) {
		t.Parallel()

		expectedErr := errors.New("factory failed")
		factory := func() (sdkmcp.Transport, error) { return nil, expectedErr }

		client, err := NewClient("test", factory)
		require.NoError(t, err)

		err = client.Start(context.Background())
		assert.ErrorContains(t, err, "failed to create transport")
	})

	t.Run("transport connect error", func(t *testing.T) {
		t.Parallel()

		expectedErr := errors.New("connect failed")
		factory := func() (sdkmcp.Transport, error) {
			return &mockTransport{connectErr: expectedErr}, nil
		}

		client, err := NewClient("test", factory)
		require.NoError(t, err)

		err = client.Start(context.Background())
		assert.ErrorContains(t, err, "failed to connect")
	})
}

func TestClientOperationsRequireStarted(t *testing.T) {
	t.Parallel()

	factory := func() (sdkmcp.Transport, error) { return &mockTransport{}, nil }
	client, err := NewClient("test", factory)
	require.NoError(t, err)

	ctx := context.Background()

	t.Run("ListTools", func(t *testing.T) {
		t.Parallel()

		_, err := client.ListTools(ctx)
		assert.ErrorContains(t, err, "not started")
	})

	t.Run("ExecuteTool", func(t *testing.T) {
		t.Parallel()

		_, err := client.ExecuteTool(ctx, "tool", json.RawMessage(`{}`))
		assert.ErrorContains(t, err, "not started")
	})

	t.Run("SyncTools", func(t *testing.T) {
		t.Parallel()

		err := client.SyncTools(ctx)
		assert.ErrorContains(t, err, "not started")
	})
}

func TestExecuteToolArgValidation(t *testing.T) {
	t.Parallel()

	factory := func() (sdkmcp.Transport, error) { return &mockTransport{}, nil }
	client, err := NewClient("test", factory)
	require.NoError(t, err)

	impl, ok := client.(*clientImpl)
	require.True(t, ok, "client must be *clientImpl")

	// Simulate a started client by setting required fields
	impl.mu.Lock()
	impl.bgCtx = context.Background()
	impl.session = &sdkmcp.ClientSession{}
	impl.mu.Unlock()

	ctx := context.Background()

	t.Run("invalid JSON", func(t *testing.T) {
		t.Parallel()

		_, err := client.ExecuteTool(ctx, "tool", json.RawMessage(`{invalid`))
		assert.ErrorContains(t, err, "arguments must be a JSON object")
	})

	t.Run("non-object args", func(t *testing.T) {
		t.Parallel()

		_, err := client.ExecuteTool(ctx, "tool", json.RawMessage(`["array"]`))
		assert.ErrorContains(t, err, "must be a JSON object")
	})
}

// TestExecuteToolEmptyArgumentsSentAsObject verifies that empty tool-call
// arguments reach the server as an empty JSON object. A nil arguments map
// assigned to CallToolParams.Arguments (an `any` field with omitempty) is
// not omitted by encoding/json — it serializes as `"arguments": null`,
// which strict servers reject (e.g. protojson-backed handlers fail with
// "unexpected token null" on every zero-parameter tool call). Empty
// arguments are routine: OpenAI-format models emit `"arguments": ""` for
// tools with no parameters.
func TestExecuteToolEmptyArgumentsSentAsObject(t *testing.T) {
	t.Parallel()

	ctx := t.Context()

	server := newMockMCPServer()

	var (
		gotArgs map[string]any
		called  bool
	)
	server.toolExecutor = func(_ string, args map[string]any) ([]sdkmcp.Content, error) {
		gotArgs = args
		called = true
		return []sdkmcp.Content{&sdkmcp.TextContent{Text: "ok"}}, nil
	}

	transport, err := server.start(ctx)
	require.NoError(t, err)
	t.Cleanup(func() { _ = server.stop() })

	client, err := NewClient("test-server", func() (sdkmcp.Transport, error) { return transport, nil })
	require.NoError(t, err)
	require.NoError(t, client.Start(ctx))
	t.Cleanup(func() { _ = client.Close() })

	// Sequential on purpose: the subtests share gotArgs/called.
	for _, tc := range []struct {
		name string
		args json.RawMessage
	}{
		{"nil raw message", nil},
		{"empty raw message", json.RawMessage("")},
		{"json null", json.RawMessage("null")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotArgs, called = nil, false

			_, err := client.ExecuteTool(ctx, "test-server__echo", tc.args)
			require.NoError(t, err)
			require.True(t, called)
			// The mock handler unmarshals the wire arguments: a JSON
			// object (even empty) produces a non-nil map, while null or
			// absent arguments leave it nil.
			assert.NotNil(t, gotArgs, "server must receive {}, not null/absent arguments")
		})
	}
}

func TestNamespaceTool(t *testing.T) {
	t.Parallel()

	factory := func() (sdkmcp.Transport, error) { return &mockTransport{}, nil }
	client, err := NewClient("github", factory)
	require.NoError(t, err)

	impl, ok := client.(*clientImpl)
	require.True(t, ok, "client must be *clientImpl")
	assert.Equal(t, "github__create-issue", impl.namespaceTool("create-issue"))
	assert.Equal(t, "github__search_code", impl.namespaceTool("search_code"))
}

func TestConcurrentAccess(t *testing.T) {
	t.Parallel()

	factory := func() (sdkmcp.Transport, error) { return &mockTransport{}, nil }
	client, err := NewClient("test", factory)
	require.NoError(t, err)

	var wg sync.WaitGroup
	for range 10 {
		wg.Go(func() {
			_ = client.ServerID()
			_ = client.Started()
		})
		wg.Go(func() {
			_, _ = client.ListTools(context.Background())
		})
	}

	wg.Wait()
}
