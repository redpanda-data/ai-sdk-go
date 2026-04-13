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
	"fmt"
	"strings"
	"sync"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// ListTools returns tool definitions for all available MCP server tools.
//
// Tool names are always namespaced with serverID (e.g., "github__create-issue").
//
// The returned definitions can be:
//   - Passed to an LLM for tool calling
//   - Used with ExecuteTool() using the Definition.Name
func (c *clientImpl) ListTools(context.Context) ([]llm.ToolDefinition, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.bgCtx == nil {
		return nil, ErrNotStarted
	}

	tools := make([]llm.ToolDefinition, 0, len(c.tools))
	for _, wrapper := range c.tools {
		tools = append(tools, wrapper.definition)
	}

	return tools, nil
}

// ExecuteTool executes a tool on the MCP server.
//
// The toolName should be the namespaced name from ListTools() (e.g., "github__create-issue").
// The client automatically strips the namespace prefix before calling the MCP server.
//
// Example:
//
//	tools, _ := client.ListTools(ctx)
//	result, _ := client.ExecuteTool(ctx, tools[0].Name, argsJSON)
func (c *clientImpl) ExecuteTool(ctx context.Context, toolName string, args json.RawMessage) (json.RawMessage, error) {
	end, err := c.beginOp()
	if err != nil {
		return nil, err
	}
	defer end()

	session, bgCtx, err := c.waitForSession(ctx)
	if err != nil {
		return nil, err
	}

	serverID := c.serverID

	// Strip namespace prefix to get the server's tool name
	// "github__create-issue" → "create-issue"
	serverToolName := strings.TrimPrefix(toolName, serverID+"__")

	// Parse arguments directly into map
	var argsMap map[string]any
	if len(args) > 0 {
		err := json.Unmarshal(args, &argsMap)
		if err != nil {
			return nil, fmt.Errorf("tool %q arguments must be a JSON object: %w", toolName, err)
		}
	}

	// Create operation context that respects both client lifetime and caller's deadline
	opCtx, cancel := opContext(bgCtx, ctx)
	defer cancel()

	// Call MCP server with server tool name (prefix stripped)
	result, err := session.CallTool(opCtx, &sdkmcp.CallToolParams{
		Name:      serverToolName,
		Arguments: argsMap,
	})
	if err != nil {
		return nil, fmt.Errorf("tool %q execution failed: %w", toolName, err)
	}

	// Log tool-level errors (IsError indicates tool execution failed)
	if result.IsError {
		c.logger.Debug("tool execution returned error",
			"tool", toolName,
			"serverID", serverID)
	}

	return json.Marshal(result.Content)
}

// SyncTools fetches tools from the MCP server and reconciles with local state
// and registry (if configured). Concurrent calls are collapsed via singleflight.
func (c *clientImpl) SyncTools(ctx context.Context) error {
	end, err := c.beginOp()
	if err != nil {
		return err
	}
	defer end()

	_, err, _ = c.syncGroup.Do("sync", func() (any, error) {
		session, bgCtx, err := c.waitForSession(ctx)
		if err != nil {
			return nil, err
		}

		opCtx, cancel := opContext(bgCtx, ctx)
		defer cancel()

		// Fetch tools from server (network I/O, outside lock)
		fetched, err := c.fetchTools(opCtx, session)
		if err != nil {
			return nil, err
		}

		// Pre-process tools (marshal JSON, apply filters) before acquiring lock
		prepared := c.prepareTools(fetched)

		// Compute diff under lock, then execute registry ops outside lock
		c.mu.Lock()
		ops := c.computeToolDiff(prepared)
		toolCount := len(c.tools)
		c.mu.Unlock()

		// Execute registry operations outside lock
		if c.registry != nil && len(ops) > 0 {
			c.executeRegistryOps(ops)
		}

		c.logger.Info("tool sync completed",
			"serverID", c.serverID,
			"toolCount", toolCount)

		return nil, nil
	})

	return err
}

// fetchTools retrieves the current tool list from the MCP server.
func (*clientImpl) fetchTools(ctx context.Context, session *sdkmcp.ClientSession) (map[string]*sdkmcp.Tool, error) {
	if session == nil {
		return nil, errors.New("no active session")
	}

	out := make(map[string]*sdkmcp.Tool)

	for mcpTool, err := range session.Tools(ctx, nil) {
		if err != nil {
			return nil, fmt.Errorf("failed to list tools: %w", err)
		}

		out[mcpTool.Name] = mcpTool
	}

	return out, nil
}

// registryOp represents a deferred registry operation to execute outside the lock.
type registryOp struct {
	register   tool.Tool // non-nil for register operations
	unregister string    // non-empty for unregister operations
}

// preparedTool holds a tool with its pre-marshalled parameters JSON.
type preparedTool struct {
	mcpTool        *sdkmcp.Tool
	paramsJSON     json.RawMessage
	namespacedName string
}

// prepareTools pre-processes fetched tools by marshalling JSON and applying filters.
// This expensive operation happens before acquiring locks.
func (c *clientImpl) prepareTools(fetched map[string]*sdkmcp.Tool) map[string]*preparedTool {
	prepared := make(map[string]*preparedTool, len(fetched))

	for _, mcpTool := range fetched {
		// Apply filter if configured
		if c.toolFilter != nil && !c.toolFilter(mcpTool.Name, mcpTool.Description) {
			continue
		}

		namespaced := c.namespaceTool(mcpTool.Name)

		// Marshal JSON outside lock
		paramsJSON, err := json.Marshal(mcpTool.InputSchema)
		if err != nil {
			c.logger.Warn("failed to marshal tool parameters",
				"tool", mcpTool.Name, "err", err)

			continue
		}

		prepared[namespaced] = &preparedTool{
			mcpTool:        mcpTool,
			paramsJSON:     paramsJSON,
			namespacedName: namespaced,
		}
	}

	return prepared
}

// computeToolDiff computes the diff between current and prepared tools.
// Returns registry operations to execute. MUST be called with write lock held.
func (c *clientImpl) computeToolDiff(prepared map[string]*preparedTool) []registryOp {
	var ops []registryOp

	// Remove tools that no longer exist on the server
	for namespaced := range c.tools {
		if _, exists := prepared[namespaced]; !exists {
			delete(c.tools, namespaced)

			if c.registry != nil {
				ops = append(ops, registryOp{unregister: namespaced})
			}

			c.logger.Debug("removed tool", "tool", namespaced)
		}
	}

	// Add new tools and update existing tools
	for namespaced, prep := range prepared {
		def := llm.ToolDefinition{
			Name:        namespaced,
			Description: prep.mcpTool.Description,
			Parameters:  prep.paramsJSON,
			Type:        llm.ToolTypeExtension, // MCP tools are remote (extension)
		}

		if w, exists := c.tools[namespaced]; exists {
			// Update existing tool definition
			w.mu.Lock()
			w.definition = def
			w.mu.Unlock()
			c.logger.Debug("updated tool", "tool", namespaced)
		} else {
			// Create new tool wrapper
			w := &toolWrapper{
				client:     c,
				definition: def,
			}
			c.tools[namespaced] = w

			if c.registry != nil {
				ops = append(ops, registryOp{register: w})
			}

			c.logger.Debug("added tool", "tool", namespaced)
		}
	}

	return ops
}

// executeRegistryOps executes registry operations. Called without lock.
func (c *clientImpl) executeRegistryOps(ops []registryOp) {
	for _, op := range ops {
		if op.unregister != "" {
			err := c.registry.Unregister(op.unregister)
			if err != nil {
				c.logger.Warn("failed to unregister tool",
					"tool", op.unregister,
					"err", err)
			}
		} else if op.register != nil {
			// Build registration options with timeout if configured
			var opts []tool.Option
			if c.toolTimeout > 0 {
				opts = append(opts, tool.WithTimeout(c.toolTimeout))
			}

			err := c.registry.Register(op.register, opts...)
			if err != nil {
				c.logger.Warn("failed to register tool",
					"tool", op.register.Definition().Name,
					"err", err)
			}
		}
	}
}

// autoSyncLoop runs in the background and periodically syncs tools.
// Handles both timer-based syncs and server notification-based syncs with debouncing.
func (c *clientImpl) autoSyncLoop() {
	defer c.wg.Done()

	ticker := time.NewTicker(c.autoSyncInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.bgCtx.Done():
			return

		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

			err := c.SyncTools(ctx)
			if err != nil {
				c.logger.Warn("auto-sync failed",
					"serverID", c.serverID,
					"err", err)
			}

			cancel()

		case <-c.notifyCh:
			// Debounce: drain any additional queued notifications
			drain := true
			for drain {
				select {
				case <-c.notifyCh:
				default:
					drain = false
				}
			}

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

			err := c.SyncTools(ctx)
			if err != nil {
				c.logger.Warn("sync after notification failed",
					"serverID", c.serverID,
					"err", err)
			}

			cancel()
		}
	}
}

// namespaceTool creates a namespaced tool name to prevent collisions.
// Format: serverID__toolName (double underscore for LLM API compatibility).
func (c *clientImpl) namespaceTool(name string) string {
	return fmt.Sprintf("%s__%s", c.serverID, name)
}

// toolWrapper wraps an MCP tool and implements the tool.Tool interface.
type toolWrapper struct {
	client *clientImpl

	mu         sync.RWMutex
	definition llm.ToolDefinition
}

// Ensure toolWrapper implements tool.Tool at compile time.
var _ tool.Tool = (*toolWrapper)(nil)

// Definition returns the tool's definition for LLM consumption.
func (w *toolWrapper) Definition() llm.ToolDefinition {
	w.mu.RLock()
	defer w.mu.RUnlock()

	return w.definition
}

// IsAsynchronous implements tool.Tool.
func (*toolWrapper) IsAsynchronous() bool { return false }

// Execute forwards the tool execution to the MCP client.
// Uses the namespaced tool name from the definition.
func (w *toolWrapper) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	w.mu.RLock()
	toolName := w.definition.Name
	w.mu.RUnlock()

	return w.client.ExecuteTool(ctx, toolName, args)
}
