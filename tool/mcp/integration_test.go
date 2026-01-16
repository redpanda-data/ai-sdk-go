package mcp

import (
	"context"
	"encoding/json"
	"errors"
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

// TestIntegrationInitialConnectionFailureRecovers verifies that when the MCP server
// is unavailable at startup, the client starts successfully and reconnects once
// the server becomes available.
func TestIntegrationInitialConnectionFailureRecovers(t *testing.T) { //nolint:paralleltest // spawns MCP server process
	ctx := t.Context()

	// Create harness but DON'T spawn server yet - simulates server unavailable at startup
	harness := newRestartHarness(ctx)
	t.Cleanup(harness.Close)

	// Create client with a factory that will block until server is available
	client, err := NewClient("test-server", harness.Factory(t))
	require.NoError(t, err)
	t.Cleanup(func() { _ = client.Close() })

	// Start client in background - it will try to connect and retry
	startDone := make(chan error, 1)
	go func() {
		startDone <- client.Start(ctx)
	}()

	// Give client time to attempt initial connection (will block in factory waiting for transport)
	time.Sleep(50 * time.Millisecond)

	// Now spawn the server - client should connect
	harness.Spawn(t)

	// Start should complete successfully
	select {
	case err := <-startDone:
		require.NoError(t, err, "Start() should succeed once server is available")
	case <-time.After(3 * time.Second):
		t.Fatal("Start() timed out waiting for server")
	}

	assert.True(t, client.Started(), "client should be in started state")

	// Verify client can execute tools
	tools, err := client.ListTools(ctx)
	require.NoError(t, err)
	echoTool := toolNameContaining(t, tools, "echo")

	result, err := client.ExecuteTool(ctx, echoTool, json.RawMessage(`{"message":"recovered"}`))
	require.NoError(t, err)
	require.Contains(t, string(result), "recovered")
}

// TestIntegrationShutdownDuringReconnect verifies that calling Close() while the
// client is attempting to reconnect completes cleanly without hanging.
func TestIntegrationShutdownDuringReconnect(t *testing.T) { //nolint:paralleltest // tests client reconnection
	ctx := t.Context()

	// Track connection attempts and control when connections fail
	var mu sync.Mutex
	connectCount := 0
	allowConnect := true

	server := newMockMCPServer()
	transport, err := server.start(ctx)
	require.NoError(t, err)
	t.Cleanup(func() { _ = server.stop() })

	factory := func() (sdkmcp.Transport, error) {
		mu.Lock()
		connectCount++
		allow := allowConnect
		mu.Unlock()

		if !allow {
			return nil, errors.New("connection refused")
		}
		return transport, nil
	}

	client, err := NewClient("test-server", factory)
	require.NoError(t, err)
	require.NoError(t, client.Start(ctx))

	// Verify connection works
	tools, err := client.ListTools(ctx)
	require.NoError(t, err)
	require.NotEmpty(t, tools)

	// Block new connections and stop server - client will start reconnecting
	mu.Lock()
	allowConnect = false
	mu.Unlock()
	_ = server.stop()

	// Give reconnect loop time to start attempting reconnection
	time.Sleep(300 * time.Millisecond)

	// Verify reconnect attempts are happening
	mu.Lock()
	attempts := connectCount
	mu.Unlock()
	assert.Greater(t, attempts, 1, "should have attempted reconnection")

	// Close should complete cleanly even during reconnect
	shutdownDone := make(chan error, 1)
	go func() {
		shutdownDone <- client.Close()
	}()

	select {
	case err := <-shutdownDone:
		assert.NoError(t, err, "Close() should complete cleanly during reconnect")
	case <-time.After(2 * time.Second):
		t.Fatal("Close() hung during reconnect - this is a bug")
	}
}

// TestIntegrationOperationsDuringReconnect verifies that operations properly
// handle the case when no session is available (during reconnection).
func TestIntegrationOperationsDuringReconnect(t *testing.T) { //nolint:paralleltest // spawns MCP server process
	ctx := t.Context()

	harness := newRestartHarness(ctx)
	t.Cleanup(harness.Close)

	harness.Spawn(t)

	client, err := NewClient("test-server", harness.Factory(t))
	require.NoError(t, err)
	require.NoError(t, client.Start(ctx))
	t.Cleanup(func() { _ = client.Close() })

	// Get tool name while connected
	tools, err := client.ListTools(ctx)
	require.NoError(t, err)
	echoTool := toolNameContaining(t, tools, "echo")

	// Stop server - client will lose session and start reconnecting
	harness.StopCurrent()

	// Give time for session to close and reconnect to start
	time.Sleep(100 * time.Millisecond)

	// Operations with short timeout should fail/timeout while reconnecting
	shortCtx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
	defer cancel()

	_, err = client.ExecuteTool(shortCtx, echoTool, json.RawMessage(`{"message":"test"}`))
	// Should get context deadline exceeded or similar error, not panic
	assert.Error(t, err, "ExecuteTool should fail when no session is available")

	// Now bring server back up
	harness.Spawn(t)

	// Should eventually recover and work again
	require.Eventually(t, func() bool {
		callCtx, callCancel := context.WithTimeout(ctx, 100*time.Millisecond)
		defer callCancel()

		_, err := client.ExecuteTool(callCtx, echoTool, json.RawMessage(`{"message":"recovered"}`))
		return err == nil
	}, 3*time.Second, 100*time.Millisecond, "client should recover after server comes back")
}

// TestIntegrationMultipleReconnectCycles verifies that the client handles
// multiple server restarts gracefully.
func TestIntegrationMultipleReconnectCycles(t *testing.T) { //nolint:paralleltest // spawns MCP server process
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

	// Perform multiple restart cycles
	for cycle := 1; cycle <= 3; cycle++ {
		// Verify current connection works
		result, err := client.ExecuteTool(ctx, echoTool, json.RawMessage(`{"message":"cycle`+string(rune('0'+cycle))+`"}`))
		require.NoError(t, err, "cycle %d: should work before restart", cycle)
		require.Contains(t, string(result), "cycle")

		// Restart server
		harness.StopCurrent()
		harness.Spawn(t)

		// Wait for reconnection
		require.Eventually(t, func() bool {
			callCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
			defer cancel()

			_, err := client.ExecuteTool(callCtx, echoTool, json.RawMessage(`{"message":"after"}`))
			return err == nil
		}, 2*time.Second, 50*time.Millisecond, "cycle %d: should reconnect after restart", cycle)
	}
}

// TestIntegrationShutdownBeforeServerAvailable verifies that Close() works
// correctly when called before the server ever became available.
func TestIntegrationShutdownBeforeServerAvailable(t *testing.T) { //nolint:paralleltest // tests client without server
	ctx := t.Context()

	// Factory that always fails - simulates permanently unavailable server
	failCount := 0
	factory := func() (sdkmcp.Transport, error) {
		failCount++
		return nil, errors.New("server unavailable")
	}

	client, err := NewClient("test-server", factory)
	require.NoError(t, err)

	// Start in background - it will keep retrying
	startDone := make(chan error, 1)
	go func() {
		startDone <- client.Start(ctx)
	}()

	// Wait for Start to complete (it should succeed even though connection fails)
	select {
	case err := <-startDone:
		require.NoError(t, err, "Start() should succeed even when server is unavailable")
	case <-time.After(1 * time.Second):
		t.Fatal("Start() should complete quickly even when connection fails")
	}

	// Let it attempt a few reconnects
	time.Sleep(500 * time.Millisecond)
	assert.Greater(t, failCount, 1, "should have attempted multiple connections")

	// Close should work cleanly
	shutdownDone := make(chan error, 1)
	go func() {
		shutdownDone <- client.Close()
	}()

	select {
	case err := <-shutdownDone:
		assert.NoError(t, err, "Close() should complete cleanly")
	case <-time.After(2 * time.Second):
		t.Fatal("Close() hung - reconnect loop didn't terminate properly")
	}
}
