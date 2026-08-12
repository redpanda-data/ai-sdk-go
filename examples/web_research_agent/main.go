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

// Command web_research_agent is an interactive research assistant that answers
// questions by issuing parallel web searches (Tavily) and page fetches, then
// synthesizing an answer with citations.
//
// It demonstrates three things the other examples don't: several tool calls
// issued in a single turn so they execute in parallel, a multi-turn REPL over
// one session, and the plugins/otel interceptor tracing all of it.
//
// Tracing is optional and configured entirely through the standard OpenTelemetry
// environment variables. With no OTEL_EXPORTER_OTLP_ENDPOINT set the agent runs
// with tracing disabled; point that variable at any OTLP/HTTP collector to get
// invoke_agent → chat → parallel execute_tool spans.
package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	pluginotel "github.com/redpanda-data/ai-sdk-go/plugins/otel"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const systemPrompt = `You are a diligent research assistant.

Your process for any non-trivial question:
1. Decompose it into 2-4 focused sub-topics.
2. Call web_search for EACH sub-topic IN THE SAME TURN so the searches run in
   parallel. Do not search one at a time.
3. Read the most promising results with fetch_url (again, issue the fetches
   together when you need more than one).
4. Synthesize a concise, well-structured answer that cites the sources you used.

Always use the tools — never answer research questions from memory alone.`

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// web_search hits the Tavily API (https://tavily.com — free tier available).
	tavilyKey := os.Getenv("TAVILY_API_KEY")
	if tavilyKey == "" {
		log.Fatal("TAVILY_API_KEY is required for web_search (get a key at https://app.tavily.com)")
	}

	ctx := context.Background()

	// Optional: export spans to whatever OTLP/HTTP collector the standard
	// OpenTelemetry environment variables point at. tp is nil when tracing is
	// not configured, in which case the interceptor is simply not installed.
	tp, err := setupTracing(ctx)
	if err != nil {
		log.Fatalf("Failed to set up OTLP tracing: %v", err)
	}
	if tp != nil {
		// Flush spans on exit; surfaces any export/HTTP error.
		defer func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			if err := tp.Shutdown(shutdownCtx); err != nil {
				log.Printf("Failed to flush traces: %v", err)
			}
		}()
	}

	provider, err := openai.NewProvider(apiKey)
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}
	// Latest GPT flagship (GPT-5.6 "Sol").
	model, err := provider.NewModel(openai.ModelGPT5_6)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	registry := tool.NewRegistry(tool.RegistryConfig{})
	for _, t := range []tool.Tool{NewWebSearchTool(tavilyKey), NewFetchURLTool()} {
		if err := registry.Register(t); err != nil {
			log.Fatalf("Failed to register tool: %v", err)
		}
	}

	opts := []llmagent.Option{
		llmagent.WithTools(registry),
		llmagent.WithMaxTurns(10),
	}
	if tp != nil {
		// The OTel tracing interceptor records invoke_agent / chat / execute_tool
		// spans. Inputs and outputs are recorded here because seeing the tool
		// arguments and results is the point of the example — in production
		// that's a privacy decision, since prompts and results land in spans.
		opts = append(opts, llmagent.WithInterceptors(pluginotel.New(
			pluginotel.WithTracerProvider(tp),
			pluginotel.WithRecordInputs(true),
			pluginotel.WithRecordOutputs(true),
		)))
	}

	ag, err := llmagent.New("research-assistant", systemPrompt, model, opts...)
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	r, err := runner.New(ag, session.NewInMemoryStore())
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	printBanner(tp != nil)

	// One session across the whole REPL so multi-turn context (and one
	// gen_ai.conversation.id) is preserved. Each question is its own trace.
	const sessionID = "research-session"
	const userID = "local-user"
	reader := bufio.NewReader(os.Stdin)
	turn := 0
	for {
		fmt.Print("\n\033[1myou ›\033[0m ")
		line, rerr := reader.ReadString('\n')
		q := strings.TrimSpace(line)
		if q == "exit" || q == "quit" {
			break
		}
		if q != "" {
			turn++
			runTurn(ctx, r, userID, sessionID, q)
			if tp != nil {
				// Push this turn's spans now (instead of waiting for the batch
				// timer) so any export error lands with the answer.
				flushCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
				if err := tp.ForceFlush(flushCtx); err != nil {
					log.Printf("\033[31motlp: flush after turn failed: %v\033[0m", err)
				}
				cancel()
			}
		}
		if rerr != nil {
			if !errors.Is(rerr, io.EOF) {
				log.Printf("input error: %v", rerr)
			}
			break
		}
	}
	fmt.Printf("\nBye — %d question(s) answered\n", turn)
}

// runTurn runs one question and streams the agent's tool activity + answer.
func runTurn(ctx context.Context, r *runner.Runner, userID, sessionID, question string) {
	userMessage := llm.NewMessage(llm.RoleUser, llm.NewTextPart(question))
	var answer string
	for evt, err := range r.Run(ctx, userID, sessionID, userMessage) {
		if err != nil {
			log.Printf("run error: %v", err)
			return
		}
		switch e := evt.(type) {
		case agent.ToolRequestEvent:
			fmt.Printf("  \033[36m→ %s\033[0m %s\n", e.Request.Name, compact(e.Request.Arguments))
		case agent.ToolResponseEvent:
			fmt.Printf("  \033[32m✓ %s\033[0m\n", e.Response.Name)
		case agent.MessageEvent:
			if txt := e.Response.Message.TextContent(); strings.TrimSpace(txt) != "" {
				answer = txt
			}
		case agent.InvocationEndEvent:
			if e.Usage != nil {
				fmt.Printf("  \033[90m[%s · in=%d out=%d]\033[0m\n",
					e.FinishReason, e.Usage.InputTokens, e.Usage.OutputTokens)
			}
		}
	}
	if answer != "" {
		fmt.Printf("\n\033[1massistant ›\033[0m %s\n", answer)
	}
}

func compact(raw []byte) string {
	s := strings.Join(strings.Fields(string(raw)), " ")
	if len(s) > 120 {
		s = s[:117] + "…"
	}
	return s
}

func printBanner(tracing bool) {
	fmt.Println("=================================================================")
	fmt.Println("  Web Research Agent — parallel tool calls over a multi-turn session")
	fmt.Println("=================================================================")
	if tracing {
		fmt.Printf("  Tracing     : on → %s\n", otlpTarget())
		fmt.Println("  Each answer emits invoke_agent → chat → parallel execute_tool spans.")
	} else {
		fmt.Println("  Tracing     : off (set OTEL_EXPORTER_OTLP_ENDPOINT to export spans)")
	}
	fmt.Println("  Try: \"Compare Redpanda and Apache Kafka on performance, architecture,")
	fmt.Println("        and operational cost.\"   (type 'exit' to quit)")
	fmt.Println("=================================================================")
}

// otlpTarget reports the configured OTLP endpoint, for display only.
func otlpTarget() string {
	if v := os.Getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"); v != "" {
		return v
	}
	return strings.TrimRight(os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"), "/") + "/v1/traces"
}

// setupTracing builds a TracerProvider exporting over OTLP/HTTP, configured
// purely from the standard OpenTelemetry environment variables — chiefly
// OTEL_EXPORTER_OTLP_ENDPOINT (or ..._TRACES_ENDPOINT), OTEL_EXPORTER_OTLP_HEADERS
// for auth (e.g. "Authorization=Bearer <token>"), and OTEL_SERVICE_NAME. Passing
// no options to otlptracehttp.New is what makes the exporter read them.
//
// It returns (nil, nil) when no endpoint is configured, so tracing stays opt-in
// and the example runs with nothing but an OpenAI and a Tavily key.
func setupTracing(ctx context.Context) (*sdktrace.TracerProvider, error) {
	if os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT") == "" &&
		os.Getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") == "" {
		return nil, nil
	}

	exp, err := otlptracehttp.New(ctx)
	if err != nil {
		return nil, err
	}

	res := resource.NewWithAttributes("",
		attribute.String("service.name", getenv("OTEL_SERVICE_NAME", "web-research-agent")),
	)
	return sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(&loggingExporter{SpanExporter: exp, target: otlpTarget()}),
		sdktrace.WithResource(res),
	), nil
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// loggingExporter wraps the OTLP exporter to log every batch it ships (and any
// export failure), which makes this example usable as a collector smoke test.
type loggingExporter struct {
	sdktrace.SpanExporter

	target string
}

func (e *loggingExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	if err := e.SpanExporter.ExportSpans(ctx, spans); err != nil {
		log.Printf("\033[31motlp: FAILED to export %d span(s) to %s: %v\033[0m", len(spans), e.target, err)
		return err
	}
	log.Printf("\033[90motlp: exported %d span(s) to %s\033[0m", len(spans), e.target)
	return nil
}
