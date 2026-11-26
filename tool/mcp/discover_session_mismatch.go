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
	"log/slog"
	"strings"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

// legacyProtocolVersion is the newest MCP revision that opens a session with
// initialize instead of server/discover. Revision 2026-07-28 replaced the
// handshake with the stateless server/discover request (SEP-2575).
const legacyProtocolVersion = "2025-11-25"

// connectWithDiscoverFallback opens an MCP session, and retries with initialize only when
// the server/discover handshake leaves the client holding two session IDs.
//
// go-sdk v1.8.0 keeps the Mcp-Session-Id of every response that it accepts,
// the rejection of server/discover included. A server that rejects discover
// with a session ID of its own, then answers the initialize fallback with
// another one, makes the client fail with "mismatching session IDs". The
// retry asks for the newest revision that does not send server/discover, so
// only one session ID ever exists.
//
// REMOVE THIS FILE, and its test, after go.mod carries go-sdk v1.9.0 or newer,
// which includes https://github.com/modelcontextprotocol/go-sdk/pull/1523.
// Until then the retry is harmless: it only runs on that error.
func connectWithDiscoverFallback(
	ctx context.Context,
	mcpClient *sdkmcp.Client,
	newTransport TransportFactory,
	logger *slog.Logger,
	serverID string,
) (*sdkmcp.ClientSession, error) {
	transport, err := newTransport()
	if err != nil {
		return nil, fmt.Errorf("failed to create transport: %w", err)
	}

	session, err := mcpClient.Connect(ctx, transport, nil)
	// go-sdk builds this error with fmt.Errorf and exports no sentinel, so match its text.
	if err == nil || !strings.Contains(err.Error(), "mismatching session IDs") {
		return session, err
	}

	// A transport is spent after a failed Connect, so the retry needs its own.
	retryTransport, transportErr := newTransport()
	if transportErr != nil {
		return nil, fmt.Errorf("failed to create transport for the initialize-only retry: %w (after %w)", transportErr, err)
	}

	session, retryErr := mcpClient.Connect(ctx, retryTransport, &sdkmcp.ClientSessionOptions{
		// This is the fix. Below 2026-07-28 go-sdk sends no server/discover, so
		// the stray session ID never arrives and initialize gives the only one.
		ProtocolVersion: legacyProtocolVersion,
	})
	if retryErr != nil {
		return nil, fmt.Errorf("%w (after retrying with initialize only: %w)", err, retryErr)
	}

	logger.Warn("MCP server gave mismatching session IDs for server/discover, connected with initialize only",
		"server_id", serverID,
		"err", err,
	)

	return session, nil
}
