// Package agentpack provides a zero-code agent deployment framework.
//
// It glues together the ai-sdk-go building blocks (providers, agent, runner,
// MCP client, session stores, OTEL, A2A adapter) following 12-factor app
// principles. Deploy an agent by writing an agent.yaml manifest and setting
// environment variables — no Go code required for the common case.
package agentpack

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Run is the main entry point for agentpack. It loads configuration,
// wires all components, and runs the agent in the configured mode.
func Run(ctx context.Context, configPath string) error {
	// 1. Load agent.yaml.
	cfg, err := LoadConfig(configPath)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	// 2. Set up structured JSON logger to stdout.
	logger := newLogger(cfg.Name)
	logger.Info("Starting agent", "name", cfg.Name)

	// 3. Resolve prompt (inline or from file).
	baseDir := filepath.Dir(configPath)
	prompt, err := cfg.ResolvePrompt(baseDir)
	if err != nil {
		return fmt.Errorf("resolve prompt: %w", err)
	}

	// 4. Create provider + model from env vars.
	model, err := newModel(ctx)
	if err != nil {
		return fmt.Errorf("create model: %w", err)
	}

	// 5. Create tool registry.
	registry := tool.NewRegistry(tool.RegistryConfig{})

	// 6. Start MCP clients from config.
	var mcpMgr *mcpManager
	if len(cfg.MCPs) > 0 {
		mcpMgr, err = startMCPClients(ctx, cfg.MCPs, registry, logger)
		if err != nil {
			return fmt.Errorf("start MCP clients: %w", err)
		}
		defer mcpMgr.Shutdown(ctx)
	}

	// 7. Create session store from env vars.
	store, storeCleanup, err := newSessionStore(ctx)
	if err != nil {
		return fmt.Errorf("create session store: %w", err)
	}
	defer storeCleanup()

	// 8. Configure OTEL if enabled.
	otelCfg, err := setupOTEL(ctx, cfg.Name)
	if err != nil {
		return fmt.Errorf("setup OTEL: %w", err)
	}
	if otelCfg != nil {
		defer otelCfg.shutdown(ctx)
	}

	// Log total registered tools.
	tools := registry.List()
	toolNames := make([]string, len(tools))
	for i, t := range tools {
		toolNames[i] = t.Name
	}
	logger.Info("Tool registry ready", "count", len(tools), "tools", toolNames)

	// 9. Create agent.
	agentOpts := buildAgentOpts(cfg, registry, otelCfg)
	ag, err := llmagent.New(cfg.Name, prompt, model, agentOpts...)
	if err != nil {
		return fmt.Errorf("create agent: %w", err)
	}

	// 10. Create runner.
	r, err := runner.New(ag, store, runner.WithLogger(logger))
	if err != nil {
		return fmt.Errorf("create runner: %w", err)
	}

	// 11. Handle OS signals for graceful shutdown.
	ctx, cancel := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	// 12. Run in configured mode.
	mode := os.Getenv("AGENT_MODE")
	if mode == "" {
		mode = "http"
	}

	switch mode {
	case "http":
		return serveHTTP(ctx, ag, r, cfg, logger)
	case "cli":
		return runCLI(ctx, r, logger)
	default:
		return fmt.Errorf("unknown AGENT_MODE: %s (supported: http, cli)", mode)
	}
}

func buildAgentOpts(cfg *AgentConfig, registry tool.Registry, otelCfg *otelSetup) []llmagent.Option {
	var opts []llmagent.Option

	if cfg.Description != "" {
		opts = append(opts, llmagent.WithDescription(cfg.Description))
	}

	opts = append(opts, llmagent.WithTools(registry))

	if cfg.MaxTurns > 0 {
		opts = append(opts, llmagent.WithMaxTurns(cfg.MaxTurns))
	}
	if cfg.ToolConcurrency > 0 {
		opts = append(opts, llmagent.WithToolConcurrency(cfg.ToolConcurrency))
	}

	var interceptors []agent.Interceptor
	if otelCfg != nil {
		interceptors = append(interceptors, otelCfg.interceptor)
	}
	if len(interceptors) > 0 {
		opts = append(opts, llmagent.WithInterceptors(interceptors...))
	}

	return opts
}
