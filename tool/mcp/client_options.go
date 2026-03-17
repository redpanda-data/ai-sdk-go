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
	"log/slog"
	"time"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

// ClientOption is a functional option for configuring a Client.
type ClientOption func(*clientImpl)

// WithRegistry sets the tool.Registry for automatic tool registration.
// When provided, tools are automatically registered and kept in sync.
// Without a registry, tools are accessed via ListTools() and ExecuteTool().
func WithRegistry(registry tool.Registry) ClientOption {
	return func(c *clientImpl) {
		c.registry = registry
	}
}

// WithAutoSync enables automatic periodic tool syncing from the MCP server.
// Setting interval to 0 or negative disables auto-sync.
// Requires a registry configured via WithRegistry.
func WithAutoSync(interval time.Duration) ClientOption {
	return func(c *clientImpl) {
		if interval < 0 {
			interval = 0 // Treat negative as disabled
		}

		c.autoSyncInterval = interval
	}
}

// WithLogger sets a custom logger for the client.
// Defaults to a no-op logger.
func WithLogger(logger *slog.Logger) ClientOption {
	return func(c *clientImpl) {
		c.logger = logger
	}
}

// ToolFilterFunc determines whether a tool should be registered.
// It receives the tool's name and description, and returns true if the tool should be included.
type ToolFilterFunc func(name, description string) bool

// WithToolFilter sets a filter function for selective tool registration.
// Only tools returning true from the filter are registered.
// Requires a registry configured via WithRegistry.
//
// Example:
//
//	// Only register tools that start with "search_"
//	filter := func(name, description string) bool {
//	    return strings.HasPrefix(name, "search_")
//	}
//	client, err := NewClient(serverID, transport, WithToolFilter(filter))
func WithToolFilter(filter ToolFilterFunc) ClientOption {
	return func(c *clientImpl) {
		c.toolFilter = filter
	}
}

// WithShutdownTimeout sets the maximum duration to wait for in-flight operations
// during graceful shutdown. If the timeout expires, Stop() proceeds to close the
// session anyway. Defaults to 30 seconds if not set.
//
// Example:
//
//	client, err := NewClient(serverID, transport, WithShutdownTimeout(10*time.Second))
func WithShutdownTimeout(timeout time.Duration) ClientOption {
	return func(c *clientImpl) {
		c.shutdownTimeout = timeout
	}
}

// WithToolTimeout sets the execution timeout for all tools registered by this MCP client.
// This overrides the default 30-second timeout from the tool registry.
// Setting to 0 uses the tool registry's default timeout.
//
// Use this when MCP tools need longer execution times (e.g., complex queries, data processing).
//
// Example:
//
//	client, err := NewClient(serverID, transport, WithToolTimeout(10*time.Minute))
func WithToolTimeout(timeout time.Duration) ClientOption {
	return func(c *clientImpl) {
		c.toolTimeout = timeout
	}
}
