package mcp

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestIntegrationClientLifecycle(t *testing.T) { //nolint:paralleltest // spawns MCP server process with shared stdin/stdout/lifecycle
	registry := newMockRegistry()

	env := withTestEnv(t, testEnvOptions{
		serverID:   "test-server",
		clientOpts: []ClientOption{WithRegistry(registry)},
		skipStart:  true,
	})

	client := env.Client

	assert.Equal(t, "test-server", client.ServerID())
	assert.False(t, client.Started())

	require.NoError(t, client.Start(env.Ctx))
	assert.True(t, client.Started())

	tools, err := client.ListTools(env.Ctx)
	require.NoError(t, err)
	assert.Len(t, tools, 2)

	toolNames := make(map[string]string, len(tools))
	for _, tool := range tools {
		assert.Contains(t, tool.Name, "test-server__", "tool name should be namespaced")
		serverName := strings.TrimPrefix(tool.Name, "test-server__")
		toolNames[serverName] = tool.Name
	}

	assert.Contains(t, toolNames, "echo")
	assert.Contains(t, toolNames, "add")
	assert.Equal(t, 2, registry.count())

	echoArgs := json.RawMessage(`{"message":"hello world"}`)
	result, err := client.ExecuteTool(env.Ctx, toolNames["echo"], echoArgs)
	require.NoError(t, err)
	assert.Contains(t, string(result), "hello world")

	addArgs := json.RawMessage(`{"a":5,"b":3}`)
	result, err = client.ExecuteTool(env.Ctx, toolNames["add"], addArgs)
	require.NoError(t, err)
	assert.Contains(t, string(result), "8")

	require.NoError(t, client.SyncTools(env.Ctx))

	require.NoError(t, client.Shutdown(env.Ctx))
	assert.False(t, client.Started())
	assert.Equal(t, 0, registry.count())
}

func TestIntegrationToolFilter(t *testing.T) { //nolint:paralleltest // spawns MCP server process with shared stdin/stdout/lifecycle
	registry := newMockRegistry()

	filter := func(name, _ string) bool {
		return name == "echo"
	}

	env := withTestEnv(t, testEnvOptions{
		serverID: "test-server",
		clientOpts: []ClientOption{
			WithRegistry(registry),
			WithToolFilter(filter),
		},
	})

	client := env.Client

	tools, err := client.ListTools(env.Ctx)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	assert.Equal(t, "test-server__echo", tools[0].Name)
	assert.Equal(t, 1, registry.count())

	echoArgs := json.RawMessage(`{"message":"hello world"}`)
	result, err := client.ExecuteTool(env.Ctx, tools[0].Name, echoArgs)
	require.NoError(t, err)
	assert.Contains(t, string(result), "hello world")
}

func TestIntegrationAutoSync(t *testing.T) { //nolint:paralleltest // spawns MCP server process with shared stdin/stdout/lifecycle
	registry := newMockRegistry()

	env := withTestEnv(t, testEnvOptions{
		serverID: "test-server",
		clientOpts: []ClientOption{
			WithRegistry(registry),
			WithAutoSync(50 * time.Millisecond),
		},
	})

	_ = env // ensure env keeps client/server alive for cleanup

	require.Eventually(t, func() bool {
		return registry.count() == 2
	}, time.Second, 50*time.Millisecond)

	assert.Equal(t, 2, registry.count())
}

func TestIntegrationWithoutRegistry(t *testing.T) { //nolint:paralleltest // spawns MCP server process with shared stdin/stdout/lifecycle
	env := withTestEnv(t, testEnvOptions{serverID: "test-server"})

	client := env.Client

	tools, err := client.ListTools(env.Ctx)
	require.NoError(t, err)
	require.Len(t, tools, 2)

	echoTool := toolNameContaining(t, tools, "echo")

	echoArgs := json.RawMessage(`{"message":"test"}`)
	result, err := client.ExecuteTool(env.Ctx, echoTool, echoArgs)
	require.NoError(t, err)
	assert.Contains(t, string(result), "test")
}

func TestIntegrationShutdownWithInFlightOperations(t *testing.T) { //nolint:paralleltest // spawns MCP server process with shared stdin/stdout/lifecycle
	env := withTestEnv(t, testEnvOptions{serverID: "test-server"})

	client := env.Client
	ctx := env.Ctx

	opStarted := make(chan struct{})
	opCompleted := make(chan struct{})

	var wg sync.WaitGroup

	wg.Go(func() {
		close(opStarted)

		args := json.RawMessage(`{"message":"test"}`)

		_, err := client.ExecuteTool(ctx, "test-server__echo", args)
		if err != nil {
			t.Logf("ExecuteTool returned error: %v", err)
		}

		close(opCompleted)
	})

	wg.Go(func() {
		<-opStarted

		time.Sleep(10 * time.Millisecond)

		err := client.Shutdown(t.Context())
		require.NoError(t, err)

		select {
		case <-opCompleted:
		case <-time.After(100 * time.Millisecond):
			t.Error("Shutdown() returned before in-flight operation completed")
		}
	})

	wg.Wait()
}

func TestIntegrationConcurrentShutdownAndExecute(t *testing.T) { //nolint:paralleltest // spawns MCP server process with shared stdin/stdout/lifecycle
	env := withTestEnv(t, testEnvOptions{serverID: "test-server"})

	client := env.Client
	ctx := env.Ctx

	var wg sync.WaitGroup

	for range 10 {
		wg.Go(func() {
			args := json.RawMessage(`{"message":"test"}`)
			_, _ = client.ExecuteTool(ctx, "test-server__echo", args)
		})
	}

	time.Sleep(5 * time.Millisecond)
	require.NoError(t, client.Shutdown(t.Context()))

	wg.Wait()
}

func TestIntegrationServerRestartRecovers(t *testing.T) { //nolint:paralleltest // spawns MCP server process with shared stdin/stdout/lifecycle and tests server restart
	ctx := t.Context()

	harness := newRestartHarness(ctx)
	t.Cleanup(harness.Close)

	harness.Spawn(t)

	client, err := NewClient("test-server", harness.Factory(t))
	require.NoError(t, err)
	require.NoError(t, client.Start(ctx))
	t.Cleanup(func() { _ = client.Close() })

	tools, err := client.ListTools(ctx)
	require.NoError(t, err)
	echoTool := toolNameContaining(t, tools, "echo")

	firstArgs := json.RawMessage(`{"message":"first"}`)
	result, err := client.ExecuteTool(ctx, echoTool, firstArgs)
	require.NoError(t, err)
	require.Contains(t, string(result), "first")

	harness.StopCurrent()
	harness.Spawn(t)

	secondArgs := json.RawMessage(`{"message":"second"}`)

	require.Eventually(t, func() bool {
		callCtx, cancel := context.WithTimeout(ctx, 80*time.Millisecond)
		defer cancel()

		res, err := client.ExecuteTool(callCtx, echoTool, secondArgs)
		if err != nil {
			return false
		}

		return strings.Contains(string(res), "second")
	}, 2*time.Second, 80*time.Millisecond, "MCP client did not recover after server restart")
}

// testEnv holds common fixtures for integration tests.
type testEnv struct {
	Ctx    context.Context
	Client Client
}

// testEnvOptions configures the shared test environment helper.
type testEnvOptions struct {
	serverID   string
	clientOpts []ClientOption
	skipStart  bool
}

// withTestEnv provisions a mock server and client for an integration test and
// wires cleanup through t.Cleanup.
func withTestEnv(t *testing.T, opts testEnvOptions) *testEnv {
	t.Helper()

	ctx := t.Context()

	server := newMockMCPServer()
	transport, err := server.start(ctx)
	require.NoError(t, err)
	t.Cleanup(func() { _ = server.stop() })

	factory := func() (sdkmcp.Transport, error) {
		return transport, nil
	}

	serverID := opts.serverID
	if serverID == "" {
		serverID = "test-server"
	}

	client, err := NewClient(serverID, factory, opts.clientOpts...)
	require.NoError(t, err)

	if !opts.skipStart {
		require.NoError(t, client.Start(ctx))
	}

	t.Cleanup(func() { _ = client.Close() })

	return &testEnv{
		Ctx:    ctx,
		Client: client,
	}
}

// toolNameContaining finds the first tool whose name includes the provided fragment.
func toolNameContaining(t *testing.T, tools []llm.ToolDefinition, fragment string) string {
	t.Helper()

	for _, tool := range tools {
		if strings.Contains(tool.Name, fragment) {
			return tool.Name
		}
	}

	require.FailNow(t, "tool not found", "wanted fragment %q", fragment)

	return ""
}

// restartHarness manages successive mock servers and exposes fresh transports
// to the MCP client factory.
type restartHarness struct {
	ctx        context.Context
	transports chan sdkmcp.Transport

	mu      sync.Mutex
	current *mockMCPServer
}

// newRestartHarness constructs a harness rooted in the provided context.
func newRestartHarness(ctx context.Context) *restartHarness {
	return &restartHarness{
		ctx:        ctx,
		transports: make(chan sdkmcp.Transport, 1),
	}
}

// Factory returns a TransportFactory that blocks until a fresh transport is available.
func (h *restartHarness) Factory(t *testing.T) TransportFactory {
	t.Helper()

	return func() (sdkmcp.Transport, error) {
		select {
		case transport := <-h.transports:
			return transport, nil
		case <-time.After(2 * time.Second):
			t.Fatal("timeout waiting for transport")
			return nil, context.DeadlineExceeded
		}
	}
}

// Spawn replaces the current mock server with a new instance and publishes its transport.
func (h *restartHarness) Spawn(t *testing.T) {
	t.Helper()

	server := newMockMCPServer()
	transport, err := server.start(h.ctx)
	require.NoError(t, err)

	h.mu.Lock()

	if h.current != nil {
		_ = h.current.stop()
	}

	h.current = server
	h.mu.Unlock()

	select {
	case h.transports <- transport:
	default:
		<-h.transports

		h.transports <- transport
	}
}

// StopCurrent shuts down the active mock server if one exists.
func (h *restartHarness) StopCurrent() {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.current != nil {
		_ = h.current.stop()
		h.current = nil
	}
}

// Close releases resources held by the harness.
func (h *restartHarness) Close() {
	h.StopCurrent()
}
