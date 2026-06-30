// testserver starts an HTTP server that uses the aisdk adapter with fakellm
// to serve canned responses for conformance testing against the real Vercel
// AI SDK TypeScript client.
package main

import (
	"context"
	"encoding/json"
	"errors"
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

	// POST /api/error -- rate limit error, surfaced to the client via WithOnError
	// (mirrors the reference onError option that maps an error to client text).
	errorModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenError(llm.ErrRateLimitExceeded)
	mux.Handle("POST /api/error", uimessagestream.Handler(errorModel,
		uimessagestream.WithOnError(func(err error) string { return err.Error() })))

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

	// POST /api/reasoning -- emits a reasoning trace followed by text.
	reasoningModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			return &llm.Response{
				Message: llm.NewMessage(llm.RoleAssistant,
					llm.NewReasoningPart("Thinking about the question."),
					llm.NewTextPart("The answer is 42."),
				),
				FinishReason: llm.FinishReasonStop,
			}, nil
		})
	mux.Handle("POST /api/reasoning", uimessagestream.Handler(reasoningModel))

	// Tool definition shared by the tool-calling endpoints.
	weatherTool := llm.ToolDefinition{
		Name:        "getWeather",
		Description: "Get the current weather for a city.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}

	// POST /api/tools -- single tool call, then a final text answer.
	toolModel := fakellm.NewFakeModel(fakellm.WithLatency(fakellm.LatencyProfile{}))
	toolModel.When(fakellm.LastMessageHasToolResponse("getWeather")).
		ThenStreamText("It is sunny and 72F in San Francisco.", fakellm.StreamConfig{ChunkSize: 100})
	toolModel.When(fakellm.Any()).
		ThenRespondWithToolCall("getWeather", map[string]any{"city": "San Francisco"})

	weatherExec := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"temperature":"72F","conditions":"sunny"}`), nil
	}
	mux.Handle("POST /api/tools", uimessagestream.Handler(toolModel,
		uimessagestream.WithTools([]llm.ToolDefinition{weatherTool}, weatherExec)))

	// POST /api/tool-error -- tool call whose executor fails, then a final text.
	toolErrModel := fakellm.NewFakeModel(fakellm.WithLatency(fakellm.LatencyProfile{}))
	toolErrModel.When(fakellm.LastMessageHasToolResponse("getWeather")).
		ThenStreamText("I could not fetch the weather.", fakellm.StreamConfig{ChunkSize: 100})
	toolErrModel.When(fakellm.Any()).
		ThenRespondWithToolCall("getWeather", map[string]any{"city": "San Francisco"})

	failingExec := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return nil, errors.New("weather service unavailable")
	}
	mux.Handle("POST /api/tool-error", uimessagestream.Handler(toolErrModel,
		uimessagestream.WithTools([]llm.ToolDefinition{weatherTool}, failingExec),
		uimessagestream.WithOnError(func(err error) string { return err.Error() })))

	// POST /api/multistep -- two sequential tool calls, then a final text answer.
	// Exercises text-id reset across multiple finish-step boundaries.
	stepOne := llm.ToolDefinition{Name: "stepOne", Description: "first step", Parameters: json.RawMessage(`{"type":"object"}`)}
	stepTwo := llm.ToolDefinition{Name: "stepTwo", Description: "second step", Parameters: json.RawMessage(`{"type":"object"}`)}
	multiModel := fakellm.NewFakeModel(fakellm.WithLatency(fakellm.LatencyProfile{}))
	multiModel.When(fakellm.LastMessageHasToolResponse("stepTwo")).
		ThenStreamText("Both steps are complete.", fakellm.StreamConfig{ChunkSize: 4})
	multiModel.When(fakellm.LastMessageHasToolResponse("stepOne")).
		ThenRespondWithToolCall("stepTwo", map[string]any{})
	multiModel.When(fakellm.Any()).
		ThenRespondWithToolCall("stepOne", map[string]any{})

	stepExec := func(_ context.Context, name string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"step":"` + name + `","ok":true}`), nil
	}
	mux.Handle("POST /api/multistep", uimessagestream.Handler(multiModel,
		uimessagestream.WithTools([]llm.ToolDefinition{stepOne, stepTwo}, stepExec)))

	// POST /api/midstream-error -- streams partial text, then fails mid-stream.
	// The open text part must be closed (text-end) before the error chunk so the
	// client does not throw "missing text part".
	midModel := fakellm.NewFakeModel(fakellm.WithLatency(fakellm.LatencyProfile{})).
		When(fakellm.Any()).
		ThenStreamText("Hello mid stream world", fakellm.StreamConfig{
			ChunkSize:        4,
			ErrorAfterChunks: 2,
			MidStreamError:   llm.ErrServerError,
		})
	mux.Handle("POST /api/midstream-error", uimessagestream.Handler(midModel,
		uimessagestream.WithOnError(func(err error) string { return err.Error() })))

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
