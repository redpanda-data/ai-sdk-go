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
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

// defaultSessionWait is how long ExecuteTool and SyncTools wait for the
// reconnect loop to bring back a lost session before they fail with
// ErrNoSession. It covers a short drop, such as a server restart, while a
// server that refuses every reconnect fails each call within seconds instead
// of holding it until the caller's own deadline (10 minutes for a tool call
// under a long tool timeout).
const defaultSessionWait = 5 * time.Second

// waitForSession returns the live MCP session. When the client has none
// because it is reconnecting, it waits up to c.sessionWait for the reconnect
// loop to install one, then returns an error wrapping ErrNoSession and the
// last reconnect error. The reconnect loop keeps trying in the background.
// Returns ErrClosed if the client is shutting down, and the caller's context
// error if that context ends first.
func (c *clientImpl) waitForSession(ctx context.Context) (*sdkmcp.ClientSession, context.Context, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}

	session, bgCtx, changed, err := c.sessionState()
	if err != nil || session != nil {
		return session, bgCtx, err
	}

	if c.sessionWait <= 0 {
		return nil, nil, c.noSessionError()
	}

	timer := time.NewTimer(c.sessionWait)
	defer timer.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		case <-bgCtx.Done():
			return nil, nil, ErrClosed
		case <-timer.C:
			return nil, nil, c.noSessionError()
		case <-changed:
			// Session state changed; re-check.
		}

		session, bgCtx, changed, err = c.sessionState()
		if err != nil || session != nil {
			return session, bgCtx, err
		}
	}
}

// sessionState snapshots the session, the client's lifetime context and the
// channel that closes on the next session change. It returns ErrClosed when
// the client is shutting down or has been shut down.
func (c *clientImpl) sessionState() (*sdkmcp.ClientSession, context.Context, <-chan struct{}, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closing.Load() || c.bgCtx == nil {
		return nil, nil, nil, ErrClosed
	}

	return c.session, c.bgCtx, c.sessionChanged, nil
}

// noSessionError describes why there is no session: how long the client has
// been reconnecting and the last reconnect attempt's error, when there is one.
func (c *clientImpl) noSessionError() error {
	c.mu.RLock()
	lostAt := c.sessionLostAt
	lastErr := c.lastReconnectErr
	c.mu.RUnlock()

	detail := fmt.Sprintf("server %q", c.serverID)
	if !lostAt.IsZero() {
		detail += fmt.Sprintf(", reconnecting for %s", time.Since(lostAt).Round(time.Millisecond))
	}

	if lastErr != nil {
		return fmt.Errorf("%w: %s: last reconnect attempt: %w", ErrNoSession, detail, lastErr)
	}

	return fmt.Errorf("%w: %s", ErrNoSession, detail)
}
