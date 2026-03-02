package agentpack

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"

	"github.com/redpanda-data/ai-sdk-go/agent"
	a2aadapter "github.com/redpanda-data/ai-sdk-go/adapter/a2a"
	"github.com/redpanda-data/ai-sdk-go/runner"
)

// serveHTTP starts the A2A HTTP server.
func serveHTTP(ctx context.Context, ag agent.Agent, r *runner.Runner, cfg *AgentConfig, logger *slog.Logger) error {
	port := os.Getenv("AGENT_PORT")
	if port == "" {
		port = "8080"
	}

	executor := a2aadapter.NewExecutor(ag, r, logger)

	agentCard := &a2a.AgentCard{
		Name:               cfg.Name,
		Description:        cfg.Description,
		URL:                fmt.Sprintf("http://localhost:%s/", port),
		PreferredTransport: a2a.TransportProtocolJSONRPC,
		DefaultInputModes:  []string{"text"},
		DefaultOutputModes: []string{"text"},
		Capabilities:       a2a.AgentCapabilities{Streaming: true},
	}

	requestHandler := a2asrv.NewHandler(executor, a2asrv.WithLogger(logger))

	mux := http.NewServeMux()
	mux.Handle("/", a2asrv.NewJSONRPCHandler(requestHandler))
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(agentCard))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	})

	listener, err := net.Listen("tcp", ":"+port)
	if err != nil {
		return fmt.Errorf("listen on port %s: %w", port, err)
	}

	logger.Info("HTTP server starting", "port", port)

	server := &http.Server{Handler: mux}

	// Shut down gracefully when context is cancelled.
	go func() {
		<-ctx.Done()
		logger.Info("Shutting down HTTP server")
		server.Shutdown(context.Background())
	}()

	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("http server: %w", err)
	}

	return nil
}
