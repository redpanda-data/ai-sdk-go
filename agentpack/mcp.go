package agentpack

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/mcp"
)

// mcpManager starts and manages MCP client lifecycles.
type mcpManager struct {
	clients []mcp.Client
	logger  *slog.Logger
}

// startMCPClients creates and starts MCP clients for each entry in the config.
// Tools from all MCP servers are registered into the shared registry.
func startMCPClients(ctx context.Context, mcps map[string]MCPConfig, registry tool.Registry, logger *slog.Logger) (*mcpManager, error) {
	mgr := &mcpManager{logger: logger}

	for name, cfg := range mcps {
		client, err := mgr.createClient(name, cfg, registry, logger)
		if err != nil {
			mgr.Shutdown(ctx)
			return nil, fmt.Errorf("create MCP client %s: %w", name, err)
		}

		if err := client.Start(ctx); err != nil {
			mgr.Shutdown(ctx)
			return nil, fmt.Errorf("start MCP client %s: %w", name, err)
		}

		tools, _ := client.ListTools(ctx)
		toolNames := make([]string, len(tools))
		for i, t := range tools {
			toolNames[i] = t.Name
		}
		logger.Info("MCP client started", "server", name, "tools", toolNames)
		mgr.clients = append(mgr.clients, client)
	}

	return mgr, nil
}

func (m *mcpManager) createClient(name string, cfg MCPConfig, registry tool.Registry, logger *slog.Logger) (mcp.Client, error) {
	var transport mcp.TransportFactory

	if cfg.IsStdio() {
		// Build environment key=value pairs.
		var env []string
		for k, v := range cfg.Env {
			env = append(env, k+"="+v)
		}
		logger.Debug("MCP stdio transport", "server", name, "command", cfg.Command, "args", cfg.Args, "env_keys", envKeys(env))
		transport = mcp.NewStdioTransport(cfg.Command, cfg.Args, env)
	} else if cfg.URL != "" {
		var opts []mcp.HTTPTransportOption
		if len(cfg.Headers) > 0 {
			opts = append(opts, mcp.WithHTTPHeaders(cfg.Headers))
		}
		logger.Debug("MCP HTTP transport", "server", name, "url", cfg.URL)
		transport = mcp.NewStreamableTransport(cfg.URL, opts...)
	} else {
		return nil, fmt.Errorf("MCP server %s must have either command (stdio) or url (HTTP)", name)
	}

	return mcp.NewClient(name, transport,
		mcp.WithRegistry(registry),
		mcp.WithLogger(logger),
	)
}

// envKeys extracts just the key names from "KEY=value" pairs for safe logging.
func envKeys(env []string) []string {
	keys := make([]string, len(env))
	for i, e := range env {
		for j := range e {
			if e[j] == '=' {
				keys[i] = e[:j]
				break
			}
		}
	}
	return keys
}

// Shutdown gracefully closes all MCP clients.
func (m *mcpManager) Shutdown(ctx context.Context) {
	for _, c := range m.clients {
		if err := c.Shutdown(ctx); err != nil {
			m.logger.Warn("MCP client shutdown error", "server", c.ServerID(), "error", err)
		}
	}
}
