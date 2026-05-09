// testserver starts an HTTP server that uses the aisdk adapter with fakellm
// to serve canned responses for conformance testing against the real Vercel
// AI SDK TypeScript client.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
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
			var msgs []msgInfo
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

	// Health check
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "ok")
	})

	addr := fmt.Sprintf("127.0.0.1:%d", *port)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}

	// Print the actual address (important when port=0)
	fmt.Printf("LISTEN_ADDR=%s\n", ln.Addr().String())

	server := &http.Server{
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
	}
	if err := server.Serve(ln); err != nil {
		log.Fatal(err)
	}
}
