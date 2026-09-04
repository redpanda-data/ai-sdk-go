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
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// denyingMCPServer is a real MCP server over streamable HTTP that can be
// flipped to deny every request, the way a gateway does once a policy stops
// permitting the caller.
type denyingMCPServer struct {
	*httptest.Server

	denying  atomic.Bool
	denials  atomic.Int64
	denyCode int
}

func newDenyingMCPServer(t *testing.T, denyCode int) *denyingMCPServer {
	t.Helper()

	srv := &denyingMCPServer{denyCode: denyCode}

	mcpServer := sdkmcp.NewServer(&sdkmcp.Implementation{
		Name:    "deny-test-server",
		Version: "1.0.0",
	}, &sdkmcp.ServerOptions{HasTools: true})

	mcpServer.AddTool(&sdkmcp.Tool{
		Name:        "ping",
		Description: "Answers pong",
		InputSchema: map[string]any{"type": "object"},
	}, func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "pong"}},
		}, nil
	})

	handler := sdkmcp.NewStreamableHTTPHandler(
		func(*http.Request) *sdkmcp.Server { return mcpServer },
		&sdkmcp.StreamableHTTPOptions{},
	)

	srv.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if srv.denying.Load() {
			srv.denials.Add(1)
			w.WriteHeader(srv.denyCode)

			return
		}

		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	return srv
}

// deny starts refusing every request and severs the established session, so
// the client notices the loss and reconnects into the denial.
func (s *denyingMCPServer) deny() {
	s.denying.Store(true)
	s.CloseClientConnections()
}

// inMemoryTransport serves one connect attempt from an in-process MCP server,
// for tests that only need a client that started successfully.
func inMemoryTransport(ctx context.Context, t *testing.T) TransportFactory {
	t.Helper()

	server := newMockMCPServer()

	t.Cleanup(func() { _ = server.stop() })

	return func() (sdkmcp.Transport, error) {
		return server.start(ctx)
	}
}

// TestReconnectStopsWhenServerDeniesAuthorization is the regression test for a
// client that spun on a permission error: the reconnect loop retried a 403
// handshake like a network blip, at its 30 second cap, indefinitely, while the
// session it had already cleared left every tool call waiting for the caller's
// deadline.
func TestReconnectStopsWhenServerDeniesAuthorization(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	srv := newDenyingMCPServer(t, http.StatusForbidden)
	registry := newMockRegistry()

	client, err := NewClient("denied-server", NewStreamableTransport(srv.URL),
		WithRegistry(registry),
		// A generous tool timeout is the realistic setting, and the point:
		// without the fix a call waits it out instead of failing.
		WithToolTimeout(10*time.Minute),
	)
	require.NoError(t, err)

	t.Cleanup(func() { _ = client.Close() })

	require.NoError(t, client.Start(ctx))
	require.Equal(t, 1, registry.count(), "the healthy session must register its tools")

	srv.deny()

	// The client gives up: tools unregistered, so a model is no longer offered
	// calls that cannot succeed.
	require.Eventually(t, func() bool { return registry.count() == 0 },
		30*time.Second, 25*time.Millisecond,
		"a denied client must unregister its tools instead of advertising them while it reconnects")

	// And it stops asking. Sample after it settles: the old behaviour kept
	// producing one denial per reconnectMaxDelay forever.
	settled := srv.denials.Load()

	time.Sleep(time.Second)
	assert.LessOrEqual(t, srv.denials.Load(), settled,
		"a denied client must stop reconnecting")

	// Calls fail immediately with a reason, not after the tool timeout.
	start := time.Now()
	_, err = client.ExecuteTool(ctx, "denied-server__ping", nil)
	require.Error(t, err)
	assert.Less(t, time.Since(start), 5*time.Second,
		"a call on a denied client must fail fast, not wait out the tool timeout")
	assert.True(t, errors.Is(err, ErrAuthDenied) || errors.Is(err, ErrClosed),
		"want ErrAuthDenied or ErrClosed, got %v", err)
}

// TestStartReportsAuthDenial covers the same classification on the initial
// connect, where it lets a caller tell "this server is unreachable, retry
// later" from "this server will not let me in".
func TestStartReportsAuthDenial(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	srv := newDenyingMCPServer(t, http.StatusForbidden)
	srv.denying.Store(true)

	client, err := NewClient("denied-at-start", NewStreamableTransport(srv.URL))
	require.NoError(t, err)

	t.Cleanup(func() { _ = client.Close() })

	err = client.Start(ctx)
	require.Error(t, err)
	require.ErrorIs(t, err, ErrAuthDenied)
	assert.Contains(t, err.Error(), "403")
}

// TestWaitForSessionIsBounded pins the second half of the hang: while the
// session manager holds no session, an operation must fail with ErrNoSession
// rather than block for the caller's whole deadline.
func TestWaitForSessionIsBounded(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	client, err := NewClient("no-session", inMemoryTransport(ctx, t),
		WithSessionWaitTimeout(200*time.Millisecond))
	require.NoError(t, err)

	impl, ok := client.(*clientImpl)
	require.True(t, ok)

	require.NoError(t, client.Start(ctx))

	t.Cleanup(func() { _ = client.Close() })

	// Stand in for a reconnect in progress: the session manager clears the
	// session before it starts reconnecting.
	impl.mu.Lock()
	impl.replaceSessionLocked(nil, nil)
	impl.mu.Unlock()

	// The caller's deadline is long, as a tool call's is.
	callCtx, cancelCall := context.WithTimeout(ctx, 10*time.Minute)
	defer cancelCall()

	start := time.Now()
	_, _, err = impl.waitForSession(callCtx)
	elapsed := time.Since(start)

	require.ErrorIs(t, err, ErrNoSession)
	assert.Less(t, elapsed, 5*time.Second, "the wait must be bounded by sessionWaitTimeout")
	assert.NoError(t, callCtx.Err(), "the caller's context must be left usable")
}

// TestWaitForSessionUnboundedWhenDisabled keeps the escape hatch honest: with
// the bound switched off the wait belongs to the caller's context again.
func TestWaitForSessionUnboundedWhenDisabled(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	client, err := NewClient("no-session-unbounded", inMemoryTransport(ctx, t),
		WithSessionWaitTimeout(0))
	require.NoError(t, err)

	impl, ok := client.(*clientImpl)
	require.True(t, ok)

	require.NoError(t, client.Start(ctx))

	t.Cleanup(func() { _ = client.Close() })

	impl.mu.Lock()
	impl.replaceSessionLocked(nil, nil)
	impl.mu.Unlock()

	callCtx, cancelCall := context.WithTimeout(ctx, 200*time.Millisecond)
	defer cancelCall()

	_, _, err = impl.waitForSession(callCtx)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	assert.NotErrorIs(t, err, ErrNoSession)
}

func TestAuthStatusProbe(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		codes   []int
		want    int
		wantAny bool
	}{
		{name: "success", codes: []int{http.StatusOK}, want: 0},
		{name: "server error is not a denial", codes: []int{http.StatusServiceUnavailable}, want: 0},
		{name: "unauthorized", codes: []int{http.StatusUnauthorized}, want: http.StatusUnauthorized},
		{name: "forbidden", codes: []int{http.StatusForbidden}, want: http.StatusForbidden},
		{name: "failed dependency", codes: []int{http.StatusFailedDependency}, want: http.StatusFailedDependency},
		{
			name:  "first denial wins",
			codes: []int{http.StatusOK, http.StatusForbidden, http.StatusUnauthorized},
			want:  http.StatusForbidden,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			probe := &authStatusProbe{}
			for _, code := range tc.codes {
				probe.observe(code, "tools/list")
			}

			assert.Equal(t, tc.want, probe.denialStatus())

			if tc.want != 0 {
				assert.Equal(t, "tools/list", probe.denialMethod())
			}
		})
	}
}

func TestAuthDenialStatusFromTransport(t *testing.T) {
	t.Parallel()

	t.Run("streamable transport reports the denial", func(t *testing.T) {
		t.Parallel()

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusForbidden)
		}))
		t.Cleanup(srv.Close)

		transport, err := NewStreamableTransport(srv.URL)()
		require.NoError(t, err)
		require.Equal(t, 0, authDenialStatus(transport), "nothing has been sent yet")

		streamable, ok := transport.(*sdkmcp.StreamableClientTransport)
		require.True(t, ok)

		req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, srv.URL, nil)
		require.NoError(t, err)

		resp, err := streamable.HTTPClient.Do(req)
		require.NoError(t, err)
		require.NoError(t, resp.Body.Close())

		assert.Equal(t, http.StatusForbidden, authDenialStatus(transport))
	})

	t.Run("a denial is attributed to its method", func(t *testing.T) {
		t.Parallel()

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusForbidden)
		}))
		t.Cleanup(srv.Close)

		post := func(transport sdkmcp.Transport, body string) {
			streamable, ok := transport.(*sdkmcp.StreamableClientTransport)
			require.True(t, ok)

			req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
				srv.URL, strings.NewReader(body))
			require.NoError(t, err)

			resp, err := streamable.HTTPClient.Do(req)
			require.NoError(t, err)
			require.NoError(t, resp.Body.Close())
		}

		// A refused listing is the client's own access.
		listing, err := NewStreamableTransport(srv.URL)()
		require.NoError(t, err)

		post(listing, `{"jsonrpc":"2.0","id":1,"method":"tools/list"}`)
		assert.Equal(t, http.StatusForbidden, clientOwnedDenial(listing))

		// A refused call may belong to one caller of a shared session.
		call, err := NewStreamableTransport(srv.URL)()
		require.NoError(t, err)

		post(call, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x"}}`)
		assert.Equal(t, http.StatusForbidden, authDenialStatus(call), "the denial is still recorded")
		assert.Equal(t, 0, clientOwnedDenial(call), "but it is not the client's own")
	})

	t.Run("a transport that cannot observe status reports none", func(t *testing.T) {
		t.Parallel()

		assert.Equal(t, 0, authDenialStatus(&mockTransport{}),
			"a caller's own transport keeps the retry-regardless behaviour")
	})
}

// listDenyingMCPServer permits the handshake and refuses tools/list.
type listDenyingMCPServer struct {
	*httptest.Server

	denyList   atomic.Bool
	denyCode   int
	handshakes atomic.Int64
	listDenied atomic.Int64
	callsOK    atomic.Int64
}

func newListDenyingMCPServer(t *testing.T, denyCode int) *listDenyingMCPServer {
	t.Helper()

	srv := &listDenyingMCPServer{denyCode: denyCode}

	mcpServer := sdkmcp.NewServer(&sdkmcp.Implementation{
		Name:    "list-deny-test-server",
		Version: "1.0.0",
	}, &sdkmcp.ServerOptions{HasTools: true})

	mcpServer.AddTool(&sdkmcp.Tool{
		Name:        "ping",
		Description: "Answers pong",
		InputSchema: map[string]any{"type": "object"},
	}, func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		srv.callsOK.Add(1)

		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "pong"}},
		}, nil
	})

	handler := sdkmcp.NewStreamableHTTPHandler(
		func(*http.Request) *sdkmcp.Server { return mcpServer },
		&sdkmcp.StreamableHTTPOptions{},
	)

	srv.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(body))

		switch {
		case bytes.Contains(body, []byte(`"method":"initialize"`)):
			srv.handshakes.Add(1)
		case srv.denyList.Load() && bytes.Contains(body, []byte(`"method":"tools/list"`)):
			srv.listDenied.Add(1)
			w.WriteHeader(srv.denyCode)

			return
		}

		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	return srv
}

// TestDeniedListingRetiresClient covers the manifestation that survives a fix
// aimed only at the handshake: initialize is permitted, so every reconnect
// succeeds, and it is tools/list that is refused.
//
// The result is a churn rather than a spin. The refused listing fails the
// connection, the session manager reconnects successfully, the post-reconnect
// sync is refused again, and round it goes with attempt=1 every time -- so the
// backoff never engages and the loop is tighter than a backed-off one. A
// client that may not list a server's tools cannot use that server, so the
// rejection has to retire it.
func TestDeniedListingRetiresClient(t *testing.T) {
	t.Parallel()

	for _, code := range []int{http.StatusFailedDependency, http.StatusForbidden} {
		t.Run(http.StatusText(code), func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			srv := newListDenyingMCPServer(t, code)
			registry := newMockRegistry()

			client, err := NewClient("denied-listing", NewStreamableTransport(srv.URL),
				WithRegistry(registry),
				WithAutoSync(100*time.Millisecond),
			)
			require.NoError(t, err)

			t.Cleanup(func() { _ = client.Close() })

			require.NoError(t, client.Start(ctx))
			require.Equal(t, 1, registry.count())

			srv.denyList.Store(true)

			require.Eventually(t, func() bool { return registry.count() == 0 },
				30*time.Second, 25*time.Millisecond,
				"a client that may not list tools must unregister them, not churn through sessions")

			handshakes, denials := srv.handshakes.Load(), srv.listDenied.Load()

			time.Sleep(time.Second)

			assert.LessOrEqual(t, srv.handshakes.Load(), handshakes+1,
				"a retired client must stop reconnecting")
			assert.LessOrEqual(t, srv.listDenied.Load(), denials+1,
				"a retired client must stop asking for the tool list")

			_, err = client.ExecuteTool(ctx, "denied-listing__ping", nil)
			require.Error(t, err)
			assert.True(t, errors.Is(err, ErrAuthDenied) || errors.Is(err, ErrClosed),
				"want ErrAuthDenied or ErrClosed, got %v", err)
		})
	}
}

// TestDeniedToolCallDoesNotRetireClient is the counterpart, and the reason the
// classification is method-aware: one session can be shared by several callers,
// with the server authorizing each call. A refused tools/call may be one
// caller's missing permission, which must not retire a client that works for
// everyone else.
func TestDeniedToolCallDoesNotRetireClient(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var denyCalls atomic.Bool

	mcpServer := sdkmcp.NewServer(&sdkmcp.Implementation{
		Name:    "call-deny-test-server",
		Version: "1.0.0",
	}, &sdkmcp.ServerOptions{HasTools: true})

	mcpServer.AddTool(&sdkmcp.Tool{
		Name:        "ping",
		Description: "Answers pong",
		InputSchema: map[string]any{"type": "object"},
	}, func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "pong"}},
		}, nil
	})

	handler := sdkmcp.NewStreamableHTTPHandler(
		func(*http.Request) *sdkmcp.Server { return mcpServer },
		&sdkmcp.StreamableHTTPOptions{},
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(body))

		if denyCalls.Load() && bytes.Contains(body, []byte(`"method":"tools/call"`)) {
			w.WriteHeader(http.StatusForbidden)

			return
		}

		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(ts.Close)

	registry := newMockRegistry()

	client, err := NewClient("denied-call", NewStreamableTransport(ts.URL),
		WithRegistry(registry))
	require.NoError(t, err)

	t.Cleanup(func() { _ = client.Close() })

	require.NoError(t, client.Start(ctx))
	require.Equal(t, 1, registry.count())

	denyCalls.Store(true)

	_, err = client.ExecuteTool(ctx, "denied-call__ping", nil)
	require.Error(t, err, "the refused call must fail")

	// The client itself is not implicated: nothing says it may not use the
	// server, only that this call was refused.
	require.NotErrorIs(t, err, ErrAuthDenied,
		"a refused tools/call must not be reported as the client being denied")

	impl, ok := client.(*clientImpl)
	require.True(t, ok)
	require.NoError(t, impl.terminal(), "a refused tools/call must not retire the client")

	denyCalls.Store(false)

	// And the client recovers on its own: the session manager reconnects the
	// connection that the refusal killed.
	require.Eventually(t, func() bool {
		_, err := client.ExecuteTool(ctx, "denied-call__ping", nil)

		return err == nil
	}, 30*time.Second, 100*time.Millisecond,
		"the client must keep working once the refusal stops")
}
