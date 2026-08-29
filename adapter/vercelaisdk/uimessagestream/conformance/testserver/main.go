// testserver starts an HTTP server that serves canned agent responses for
// conformance testing against the real Vercel AI SDK TypeScript client. Each
// endpoint wires uimessagestream.Handler to an llmagent backed by a scripted
// fakellm model and an in-memory session store — the same server-session setup
// production users run, minus the network.
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
	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// funcTool adapts a plain function to the tool.Tool interface.
type funcTool struct {
	def llm.ToolDefinition
	fn  func(ctx context.Context, args json.RawMessage) (json.RawMessage, error)
}

func (t *funcTool) Definition() llm.ToolDefinition { return t.def }

func (t *funcTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	return t.fn(ctx, args)
}

// mustAgent builds an llmagent over the given fake model, exiting on error.
func mustAgent(name string, model llm.Model, opts ...llmagent.Option) agent.Agent {
	ag, err := llmagent.New(name, "You are a test agent.", model, opts...)
	if err != nil {
		slog.Error("failed to build agent", "name", name, "error", err)
		os.Exit(1)
	}

	return ag
}

// mustRegistry builds a tool registry from the given tools, exiting on error.
func mustRegistry(tools ...tool.Tool) tool.Registry {
	reg := tool.NewRegistry(tool.RegistryConfig{})

	for _, tl := range tools {
		if err := reg.Register(tl); err != nil {
			slog.Error("failed to register tool", "error", err)
			os.Exit(1)
		}
	}

	return reg
}

func main() {
	port := flag.Int("port", 0, "port to listen on (0 = random)")

	flag.Parse()

	mux := http.NewServeMux()

	// mount exposes a chat handler at path: the exact path serves POST (run a
	// turn) and GET (list); path/{id} serves history and delete. Both lines
	// strip the mount prefix; the handler normalizes the resulting empty path.
	mount := func(path string, h http.Handler) {
		mux.Handle(path, http.StripPrefix(path, h))
		mux.Handle(path+"/", http.StripPrefix(path, h))
	}

	// /api/simple -- single text response
	simpleModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenStreamText("Hello, world!", fakellm.StreamConfig{ChunkSize: 100})
	mount("/api/simple", uimessagestream.Handler(mustAgent("simple", simpleModel), session.NewInMemoryStore()))

	// /api/streaming -- small chunks
	streamModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenStreamText("Hello streaming world", fakellm.StreamConfig{ChunkSize: 4})
	mount("/api/streaming", uimessagestream.Handler(mustAgent("streaming", streamModel), session.NewInMemoryStore()))

	// /api/error -- rate limit error, surfaced to the client via WithOnError
	// (mirrors the reference onError option that maps an error to client text).
	errorModel := fakellm.NewFakeModel(
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	).When(fakellm.Any()).
		ThenError(llm.ErrRateLimitExceeded)
	mount("/api/error", uimessagestream.Handler(mustAgent("error", errorModel), session.NewInMemoryStore(),
		uimessagestream.WithOnError(func(err error) string { return err.Error() })))

	// POST /api/echo-context -- echoes back the received messages as JSON text.
	// The agent's own system prompt is filtered so the echo reflects only the
	// conversation the client sent.
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
				if m.Role == llm.RoleSystem {
					continue
				}

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
	mount("/api/echo-context", uimessagestream.Handler(mustAgent("echo-context", echoModel), session.NewInMemoryStore()))

	// POST /api/system -- the agent owns the system prompt ("You are a pirate");
	// the fake model echoes back the system message it received, proving the
	// agent-supplied prompt (not anything client-sent) reaches the model.
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

	systemAgent, err := llmagent.New("system", "You are a pirate", systemModel)
	if err != nil {
		slog.Error("failed to build agent", "error", err)
		os.Exit(1)
	}

	mount("/api/system", uimessagestream.Handler(systemAgent, session.NewInMemoryStore()))

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
	mount("/api/reasoning", uimessagestream.Handler(mustAgent("reasoning", reasoningModel), session.NewInMemoryStore()))

	// Tool definition shared by the tool-calling endpoints. Agent tools are
	// runtime-discovered, so they surface to the client as dynamic-tool parts.
	weatherDef := llm.ToolDefinition{
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

	weatherTool := &funcTool{def: weatherDef, fn: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"temperature":"72F","conditions":"sunny"}`), nil
	}}
	mount("/api/tools", uimessagestream.Handler(
		mustAgent("tools", toolModel, llmagent.WithTools(mustRegistry(weatherTool))),
		session.NewInMemoryStore()))

	// POST /api/tool-error -- tool call whose executor fails, then a final text.
	toolErrModel := fakellm.NewFakeModel(fakellm.WithLatency(fakellm.LatencyProfile{}))
	toolErrModel.When(fakellm.LastMessageHasToolResponse("getWeather")).
		ThenStreamText("I could not fetch the weather.", fakellm.StreamConfig{ChunkSize: 100})
	toolErrModel.When(fakellm.Any()).
		ThenRespondWithToolCall("getWeather", map[string]any{"city": "San Francisco"})

	failingTool := &funcTool{def: weatherDef, fn: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
		return nil, errors.New("weather service unavailable")
	}}
	mount("/api/tool-error", uimessagestream.Handler(
		mustAgent("tool-error", toolErrModel, llmagent.WithTools(mustRegistry(failingTool))),
		session.NewInMemoryStore(),
		uimessagestream.WithOnError(func(err error) string { return err.Error() })))

	// POST /api/multistep -- two sequential tool calls, then a final text answer.
	// Exercises text-id reset across multiple finish-step boundaries.
	stepDef := func(name string) llm.ToolDefinition {
		return llm.ToolDefinition{Name: name, Description: name, Parameters: json.RawMessage(`{"type":"object"}`)}
	}
	multiModel := fakellm.NewFakeModel(fakellm.WithLatency(fakellm.LatencyProfile{}))
	multiModel.When(fakellm.LastMessageHasToolResponse("stepTwo")).
		ThenStreamText("Both steps are complete.", fakellm.StreamConfig{ChunkSize: 4})
	multiModel.When(fakellm.LastMessageHasToolResponse("stepOne")).
		ThenRespondWithToolCall("stepTwo", map[string]any{})
	multiModel.When(fakellm.Any()).
		ThenRespondWithToolCall("stepOne", map[string]any{})

	stepTool := func(name string) tool.Tool {
		return &funcTool{def: stepDef(name), fn: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"step":"` + name + `","ok":true}`), nil
		}}
	}
	mount("/api/multistep", uimessagestream.Handler(
		mustAgent("multistep", multiModel, llmagent.WithTools(mustRegistry(stepTool("stepOne"), stepTool("stepTwo")))),
		session.NewInMemoryStore()))

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
	mount("/api/midstream-error", uimessagestream.Handler(mustAgent("midstream-error", midModel),
		session.NewInMemoryStore(),
		uimessagestream.WithOnError(func(err error) string { return err.Error() })))

	// POST /api/max-turns -- the model requests a tool on every turn, so the
	// agent loop hits its turn budget. The client must see the fixed
	// "maximum iterations reached" error followed by finish{error}, with every
	// tool call resolved (no dynamic tool part stuck in input-available).
	loopModel := fakellm.NewFakeModel(fakellm.WithLatency(fakellm.LatencyProfile{}))
	loopModel.When(fakellm.Any()).
		ThenRespondWithToolCall("getWeather", map[string]any{"city": "San Francisco"})
	mount("/api/max-turns", uimessagestream.Handler(
		mustAgent("max-turns", loopModel, llmagent.WithTools(mustRegistry(weatherTool)), llmagent.WithMaxTurns(2)),
		session.NewInMemoryStore()))

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
