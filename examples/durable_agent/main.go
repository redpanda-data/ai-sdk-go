// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

// Command durable_agent demonstrates durable execution for agents with Redpanda
// as the only durable log. Run each piece in its own terminal:
//
//	go run . engine                         # controller + read API on :8080
//	go run . worker                         # hosts the "support" agent
//	go run . start refund-1 "refund order 42 for $120"
//	go run . describe refund-1              # status: suspended, awaiting approval
//	go run . input refund-1 approval '{"approved":true,"by":"ops"}'
//	go run . wait refund-1                  # prints the final assistant message
//	go run . continue refund-1 "thanks!"    # durable multi-turn
//	go run . trigger orders.placed          # one run per record on a topic
//
// With OPENAI_API_KEY set the worker uses gpt; otherwise a scripted fake model
// that asks for approval whenever the request mentions a refund.
//
// Environment: DX_BROKERS (localhost:19092), DX_PREFIX (durable),
// DX_ENGINE_URL (http://localhost:8080).
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/durable"
	"github.com/redpanda-data/ai-sdk-go/durable/engine"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const (
	agentName = "support"
	taskQueue = "support"
)

var errUsage = errors.New("usage: durable_agent engine|worker|start|continue|input|cancel|describe|wait|trigger ...")

func main() {
	if err := run(os.Args[1:]); err != nil {
		slog.Error("durable_agent failed", "error", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return errUsage
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cfg := durable.Config{
		Brokers:     strings.Split(env("DX_BROKERS", "localhost:19092"), ","),
		TopicPrefix: env("DX_PREFIX", durable.DefaultTopicPrefix),
		EngineURL:   env("DX_ENGINE_URL", "http://localhost:8080"),
	}

	switch args[0] {
	case "engine":
		eng := engine.New(engine.Options{
			Brokers:       cfg.Brokers,
			TopicPrefix:   cfg.TopicPrefix,
			HTTPAddr:      env("DX_HTTP_ADDR", ":8080"),
			LeaseDuration: 2 * time.Minute,
		})

		return eng.Run(ctx)

	case "worker":
		return runWorker(ctx, cfg)

	case "trigger":
		if len(args) < 2 {
			return errUsage
		}

		tr := &engine.Trigger{Config: cfg, Source: args[1], Agent: agentName, TaskQueue: taskQueue}

		return tr.Run(ctx)
	}

	client, err := durable.NewClient(cfg)
	if err != nil {
		return err
	}
	defer client.Close()

	switch args[0] {
	case "start":
		if len(args) < 3 {
			return errUsage
		}

		return client.Start(ctx, durable.StartOptions{
			RunID: args[1], Agent: agentName, TaskQueue: taskQueue, Text: strings.Join(args[2:], " "),
		})

	case "continue":
		if len(args) < 3 {
			return errUsage
		}

		return client.Continue(ctx, args[1], llm.NewMessage(llm.RoleUser, llm.NewTextPart(strings.Join(args[2:], " "))))

	case "input":
		if len(args) < 4 {
			return errUsage
		}

		return client.SendInput(ctx, args[1], args[2], json.RawMessage(args[3]))

	case "cancel":
		if len(args) < 2 {
			return errUsage
		}

		return client.Cancel(ctx, args[1], "cancelled from cli")

	case "describe":
		if len(args) < 2 {
			return errUsage
		}

		st, err := client.Describe(ctx, args[1])
		if err != nil {
			return err
		}

		return printJSON(st)

	case "wait":
		if len(args) < 2 {
			return errUsage
		}

		st, err := client.WaitResult(ctx, args[1], 500*time.Millisecond)
		if err != nil {
			return err
		}

		for i := len(st.Messages) - 1; i >= 0; i-- {
			if st.Messages[i].Role == llm.RoleAssistant {
				fmt.Fprintln(os.Stdout, st.Messages[i].TextContent()) //nolint:forbidigo // CLI output

				break
			}
		}

		if st.Status != durable.StatusCompleted {
			return fmt.Errorf("run %s ended with status %s: %s", st.RunID, st.Status, st.Error) //nolint:err113 // CLI
		}

		return nil
	}

	return errUsage
}

func runWorker(ctx context.Context, cfg durable.Config) error {
	reg := tool.NewRegistry(tool.RegistryConfig{})

	if err := reg.Register(durable.InputTool("request_approval",
		"Ask a human operator to approve the action you are about to take. Use it before any refund. "+
			"The response is the operator's decision.", "approval")); err != nil {
		return err
	}

	if err := reg.Register(durable.SleepTool()); err != nil {
		return err
	}

	model, err := pickModel()
	if err != nil {
		return err
	}

	ag, err := llmagent.New(agentName,
		"You are a customer-support agent. For refunds, call request_approval first and only proceed if approved. Be brief.",
		model,
		llmagent.WithTools(reg),
		llmagent.WithInterceptors(durable.NewInterceptor()),
	)
	if err != nil {
		return err
	}

	w, err := durable.NewWorker(cfg, taskQueue, durable.WorkerOptions{MaxConcurrent: 4})
	if err != nil {
		return err
	}

	w.Register(agentName, "v1", ag)

	return w.Run(ctx)
}

func pickModel() (llm.Model, error) {
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		provider, err := openai.NewProvider(key)
		if err != nil {
			return nil, err
		}

		return provider.NewModel(openai.ModelGPT5_4)
	}

	slog.Info("OPENAI_API_KEY not set; using a scripted fake model")

	m := fakellm.NewFakeModel()
	m.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, cc *fakellm.CallContext) (*llm.Response, error) {
		last := req.Messages[len(req.Messages)-1]

		if resps := last.ToolResponses(); len(resps) > 0 {
			return text("Operator decision received: " + string(resps[0].Result)), nil
		}

		if strings.Contains(strings.ToLower(last.TextContent()), "refund") {
			return &llm.Response{
				Message: llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart(
					fmt.Sprintf("call_%d", cc.TotalCalls), "request_approval",
					json.RawMessage(`{"request":"approve refund"}`))),
				FinishReason: llm.FinishReasonToolCalls,
			}, nil
		}

		return text("Noted: " + last.TextContent()), nil
	})

	return m, nil
}

func text(s string) *llm.Response {
	return &llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(s)), FinishReason: llm.FinishReasonStop}
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")

	return enc.Encode(v)
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}

	return def
}
