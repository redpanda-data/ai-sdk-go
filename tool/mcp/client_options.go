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

// ElicitationHandler is called when an MCP server requests user input during
// tool execution via the MCP elicitation protocol.
//
// The handler receives the server's elicitation request (message + optional
// schema) and returns the user's response. For synchronous contexts (CLI apps),
// the handler can prompt on stdin. For async contexts (web apps), it can
// integrate with a UI framework.
//
// Returning an error causes the MCP tool call to fail with that error.
type ElicitationHandler func(ctx context.Context, req *ElicitationRequest) (*ElicitationResponse, error)

// ElicitationRequest contains the MCP server's request for user input.
type ElicitationRequest struct {
	// Message is the human-readable message explaining what input is needed.
	Message string `json:"message"`
	// RequestedSchema is an optional JSON schema defining the expected input structure.
	// Only used for "form" elicitation mode.
	RequestedSchema any `json:"requested_schema,omitempty"`
}

// ElicitationResponse contains the user's response to an elicitation request.
type ElicitationResponse struct {
	// Action is the user's decision: "accept", "decline", or "cancel".
	Action string `json:"action"`
	// Content contains the submitted form data when Action is "accept".
	Content map[string]any `json:"content,omitempty"`
}

// WithElicitationHandler sets a handler for MCP server elicitation requests.
// When an MCP server requests user input during tool execution, this handler
// is called to obtain the user's response.
//
// Setting this handler automatically advertises elicitation capability to
// the MCP server during connection.
//
// Example:
//
//	handler := func(ctx context.Context, req *mcp.ElicitationRequest) (*mcp.ElicitationResponse, error) {
//	    fmt.Printf("Server asks: %s\n", req.Message)
//	    // ... prompt user and collect response ...
//	    return &mcp.ElicitationResponse{Action: "accept", Content: response}, nil
//	}
//	client, err := NewClient(serverID, transport, WithElicitationHandler(handler))
func WithElicitationHandler(handler ElicitationHandler) ClientOption {
	return func(c *clientImpl) {
		c.elicitationHandler = handler
	}
}
