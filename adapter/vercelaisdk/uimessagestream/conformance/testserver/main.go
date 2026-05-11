// testserver starts an HTTP server that uses the aisdk adapter with fakellm
// to serve canned responses for conformance testing against the real Vercel
// AI SDK TypeScript client.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/redpanda-data/ai-sdk-go/adapter/vercelaisdk/uimessagestream"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

func main() {
	port := flag.Int("port", 0, "port to listen on (0 = random)")

	flag.Parse()

	mux := http.NewServeMux()

	// POST /api/simple -- single text response
	simpleModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenStreamText("Hello, world!", fakellm.StreamConfig{ChunkSize: 100})
	mux.Handle("POST /api/simple", uimessagestream.Handler(simpleModel))

	// POST /api/streaming -- small chunks
	streamModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenStreamText("Hello streaming world", fakellm.StreamConfig{ChunkSize: 4})
	mux.Handle("POST /api/streaming", uimessagestream.Handler(streamModel))

	// POST /api/error -- rate limit error
	errorModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenError(llm.ErrRateLimitExceeded)
	mux.Handle("POST /api/error", uimessagestream.Handler(errorModel))

	// POST /api/echo-context -- echoes back the received messages as JSON text
	echoModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			type msgInfo struct {
				Role string `json:"role"`
				Text string `json:"text"`
			}

			msgs := make([]msgInfo, 0, len(req.Messages))
			for _, m := range req.Messages {
				msgs = append(msgs, msgInfo{
					Role: string(m.Role),
					Text: m.TextContent(),
				})
			}

			data, err := json.Marshal(msgs)
			if err != nil {
				return nil, err
			}

			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(string(data))),
				FinishReason: llm.FinishReasonStop,
			}, nil
		})
	mux.Handle("POST /api/echo-context", uimessagestream.Handler(echoModel))

	// POST /api/system -- has system prompt "You are a pirate", echoes it back
	systemModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			var systemText string

			for _, m := range req.Messages {
				if m.Role == llm.RoleSystem {
					systemText = m.TextContent()
					break
				}
			}

			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("System: "+systemText)),
				FinishReason: llm.FinishReasonStop,
			}, nil
		})
	mux.Handle("POST /api/system", uimessagestream.Handler(systemModel, uimessagestream.WithSystem("You are a pirate")))

	// Health check.
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)

		if _, err := w.Write([]byte("ok\n")); err != nil {
			slog.Error("health check write failed", "error", err)
		}
	})

	addr := fmt.Sprintf("127.0.0.1:%d", *port)

	ln, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", addr)
	if err != nil {
		slog.Error("listen failed", "error", err)
		os.Exit(1)
	}

	// The conformance runner parses this line from stdout.
	_, _ = os.Stdout.WriteString("LISTEN_ADDR=" + ln.Addr().String() + "\n")

	server := &http.Server{
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	if err := server.Serve(ln); err != nil {
		slog.Error("server error", "error", err)
		os.Exit(1)
	}
}
