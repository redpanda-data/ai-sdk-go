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
	"log/slog"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// errServerRefused stands in for an upstream that answers every reconnect
// with an HTTP 500.
var errServerRefused = errors.New("HTTP 500 Internal Server Error")

// refusingFactory hands each connect a fresh mock MCP server until refuse is
// set, then fails every connect with errServerRefused.
func refusingFactory(ctx context.Context, refuse *atomic.Bool) TransportFactory {
	return func() (sdkmcp.Transport, error) {
		if refuse.Load() {
			return &mockTransport{connectErr: errServerRefused}, nil
		}

		return newMockMCPServer().start(ctx)
	}
}

// startRefusingClient starts a client against a live server, then makes the
// server refuse reconnects and closes the client's session. It returns once
// the client has no session and at least one reconnect attempt has failed.
// Clearing the returned flag lets the next reconnect attempt succeed.
func startRefusingClient(t *testing.T, opts ...ClientOption) (*clientImpl, *atomic.Bool) {
	t.Helper()

	refuse := &atomic.Bool{}

	opts = append([]ClientOption{WithLogger(slog.New(slog.DiscardHandler))}, opts...)

	cl, err := NewClient("refusing", refusingFactory(t.Context(), refuse), opts...)
	require.NoError(t, err)
	require.NoError(t, cl.Start(t.Context()))
	t.Cleanup(func() { _ = cl.Close() })

	c, ok := cl.(*clientImpl)
	require.True(t, ok)

	refuse.Store(true)

	c.mu.RLock()
	session := c.session
	c.mu.RUnlock()
	require.NotNil(t, session)

	_ = session.Close()

	require.Eventually(t, func() bool {
		c.mu.RLock()
		defer c.mu.RUnlock()

		return c.session == nil && c.lastReconnectErr != nil
	}, 5*time.Second, 10*time.Millisecond, "the session must drop and a reconnect attempt must fail")

	return c, refuse
}

func executeEcho(ctx context.Context, c *clientImpl) error {
	_, err := c.ExecuteTool(ctx, "refusing__echo", json.RawMessage(`{"message":"hi"}`))
	return err
}

// TestCallsFailFastWithoutSession pins the bounded wait: when the server
// refuses every reconnect, a call returns ErrNoSession, with the reconnect
// error, once the session wait ends instead of holding the caller until its
// own deadline.
func TestCallsFailFastWithoutSession(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		wait time.Duration
		call func(context.Context, *clientImpl) error
	}{
		{name: "ExecuteTool", wait: 200 * time.Millisecond, call: executeEcho},
		{name: "SyncTools", wait: 200 * time.Millisecond, call: func(ctx context.Context, c *clientImpl) error {
			return c.SyncTools(ctx)
		}},
		{name: "ExecuteTool with no wait", wait: 0, call: executeEcho},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			c, _ := startRefusingClient(t, WithSessionWait(tt.wait))

			// A deadline far beyond the wait, like a 10-minute tool timeout.
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()

			start := time.Now()
			err := tt.call(ctx, c)
			elapsed := time.Since(start)

			require.ErrorIs(t, err, ErrNoSession)
			require.ErrorContains(t, err, errServerRefused.Error(), "the error names the reconnect failure")
			require.ErrorContains(t, err, `server "refusing", reconnecting for`)
			assert.GreaterOrEqual(t, elapsed, tt.wait, "the call waits out the session wait")
			assert.Less(t, elapsed, tt.wait+3*time.Second, "the call returns soon after the session wait")
			assert.NoError(t, ctx.Err(), "the call returns before the caller's deadline")
		})
	}
}

// TestExecuteToolWaitsForReconnect pins what the wait is for: a call made
// while the client reconnects goes through once the session comes back
// within the session wait.
func TestExecuteToolWaitsForReconnect(t *testing.T) {
	t.Parallel()

	c, refuse := startRefusingClient(t, WithSessionWait(10*time.Second))

	result := make(chan error, 1)

	go func() { result <- executeEcho(t.Context(), c) }()

	refuse.Store(false)

	select {
	case err := <-result:
		require.NoError(t, err)
	case <-time.After(15 * time.Second):
		t.Fatal("ExecuteTool did not return after the client reconnected")
	}
}

// TestSessionWaitHonorsCallerDeadline pins that a caller deadline shorter
// than the session wait still ends the call with the context's error.
func TestSessionWaitHonorsCallerDeadline(t *testing.T) {
	t.Parallel()

	c, _ := startRefusingClient(t, WithSessionWait(time.Minute))

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()

	err := executeEcho(ctx, c)

	require.ErrorIs(t, err, context.DeadlineExceeded)
	assert.NotErrorIs(t, err, ErrNoSession)
}

func TestWithSessionWait(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		opts []ClientOption
		want time.Duration
	}{
		{name: "default", want: defaultSessionWait},
		{name: "set", opts: []ClientOption{WithSessionWait(15 * time.Second)}, want: 15 * time.Second},
		{name: "zero", opts: []ClientOption{WithSessionWait(0)}, want: 0},
		{name: "negative", opts: []ClientOption{WithSessionWait(-time.Second)}, want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cl, err := NewClient("test", func() (sdkmcp.Transport, error) { return &mockTransport{}, nil }, tt.opts...)
			require.NoError(t, err)

			c, ok := cl.(*clientImpl)
			require.True(t, ok)
			assert.Equal(t, tt.want, c.sessionWait)
		})
	}

	assert.Equal(t, 5*time.Second, defaultSessionWait)
}
