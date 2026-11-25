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
	"strings"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

// legacyProtocolVersion is the newest MCP revision that opens a session with
// initialize instead of server/discover. Revision 2026-07-28 replaced the
// handshake with the stateless server/discover request (SEP-2575).
const legacyProtocolVersion = "2025-11-25"

// connectSession opens an MCP session, and retries with initialize only when
// the server/discover handshake leaves the client holding two session IDs.
//
// go-sdk v1.8.0 keeps the Mcp-Session-Id of every response that it accepts,
// the rejection of server/discover included. A server that rejects discover
// with a session ID of its own, and then answers the initialize fallback with
// another session ID, makes the client fail the whole connection with
// "mismatching session IDs". Redpanda AI Gateway puts such a server in front
// of each remote MCP member, so the agent loses every tool on that member and
// stops before its first model call. The retry asks for the newest revision
// that does not send server/discover: the session then opens with initialize
// and only one session ID ever exists.
//
// REMOVE THIS FILE, and its test, after go.mod carries a go-sdk release that
// includes https://github.com/modelcontextprotocol/go-sdk/pull/1523, which is
// v1.9.0 or newer. That release keeps the session ID of initialize only, so
// the first Connect succeeds and this retry never runs. The retry stays
// correct until then and after: it only reacts to the error, so an upgrade
// makes it dead code, not a behavior change.
func (c *clientImpl) connectSession(ctx context.Context, mcpClient *sdkmcp.Client, transport sdkmcp.Transport) (*sdkmcp.ClientSession, error) {
	session, err := mcpClient.Connect(ctx, transport, nil)
	// go-sdk builds this error with fmt.Errorf and exports no sentinel, so match its text.
	if err == nil || !strings.Contains(err.Error(), "mismatching session IDs") {
		return session, err
	}

	// A transport is spent after a failed Connect, so the retry needs its own.
	retryTransport, transportErr := c.transportFactory()
	if transportErr != nil {
		return nil, fmt.Errorf("failed to create transport for the initialize-only retry: %w (after %w)", transportErr, err)
	}

	session, retryErr := mcpClient.Connect(ctx, retryTransport, &sdkmcp.ClientSessionOptions{
		ProtocolVersion: legacyProtocolVersion,
	})
	if retryErr != nil {
		return nil, fmt.Errorf("%w (after retrying with initialize only: %w)", err, retryErr)
	}

	c.logger.Warn("MCP server gave mismatching session IDs for server/discover, connected with initialize only",
		"server_id", c.serverID,
		"err", err,
	)

	return session, nil
}
