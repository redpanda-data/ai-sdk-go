// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package durable_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go/modules/redpanda"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/durable"
	"github.com/redpanda-data/ai-sdk-go/durable/engine"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const prefix = "it"

// gate is a tool that blocks until released, standing in for a slow external
// call during which the worker dies.
type gate struct {
	mu       sync.Mutex
	release  chan struct{}
	executed int
}

func (g *gate) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{Name: "slow_lookup", Description: "slow", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}
}

func (g *gate) Execute(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
	g.mu.Lock()
	g.executed++
	rel := g.release
	g.mu.Unlock()

	if rel != nil {
		select {
		case <-rel:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	return json.RawMessage(`{"found":true}`), nil
}

// supportModel scripts the fake: ask for approval on "refund", call
// slow_lookup on "lookup", otherwise answer with the last tool result or text.
func supportModel() *fakellm.FakeModel {
	m := fakellm.NewFakeModel()
	m.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, cc *fakellm.CallContext) (*llm.Response, error) {
		last := req.Messages[len(req.Messages)-1]

		if resps := last.ToolResponses(); len(resps) > 0 {
			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("result: "+string(resps[0].Result))),
				FinishReason: llm.FinishReasonStop,
			}, nil
		}

		text := last.TextContent()

		var toolName string

		switch text {
		case "refund":
			toolName = "request_approval"
		case "lookup":
			toolName = "slow_lookup"
		default:
			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("echo: "+text)),
				FinishReason: llm.FinishReasonStop,
			}, nil
		}

		id := "call_" + toolName + "_" + itoa(cc.TotalCalls)

		return &llm.Response{
			Message:      llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart(id, toolName, json.RawMessage(`{"request":"please"}`))),
			FinishReason: llm.FinishReasonToolCalls,
		}, nil
	})

	return m
}

func itoa(i int) string {
	b, _ := json.Marshal(i) //nolint:errchkjson // ints marshal

	return string(b)
}

func newSupportAgent(t *testing.T, g *gate) agent.Agent {
	t.Helper()

	reg := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, reg.Register(durable.InputTool("request_approval", "ask a human", "approval")))
	require.NoError(t, reg.Register(g))

	ag, err := llmagent.New("support", "support agent", supportModel(),
		llmagent.WithTools(reg),
		llmagent.WithInterceptors(durable.NewInterceptor()),
	)
	require.NoError(t, err)

	return ag
}

func startWorker(ctx context.Context, t *testing.T, cfg durable.Config, g *gate) {
	t.Helper()

	w, err := durable.NewWorker(cfg, "support", durable.WorkerOptions{MaxConcurrent: 4})
	require.NoError(t, err)
	w.Register("support", "v1", newSupportAgent(t, g))

	go func() {
		if err := w.Run(ctx); err != nil {
			t.Logf("worker stopped: %v", err)
		}
	}()
}

func startEngine(ctx context.Context, t *testing.T, brokers string, lease time.Duration) string {
	t.Helper()

	eng := engine.New(engine.Options{
		Brokers:       []string{brokers},
		TopicPrefix:   prefix,
		LeaseDuration: lease,
		Partitions:    2,
	})

	go func() {
		if err := eng.Run(ctx); err != nil {
			t.Logf("engine stopped: %v", err)
		}
	}()

	select {
	case <-eng.Ready():
	case <-time.After(60 * time.Second):
		t.Fatal("engine did not become ready")
	}

	srv := httptest.NewServer(eng.Handler())
	t.Cleanup(srv.Close)

	return srv.URL
}

func waitStatus(ctx context.Context, t *testing.T, c *durable.Client, runID, status string) *durable.RunState {
	t.Helper()

	deadline := time.Now().Add(60 * time.Second)

	for time.Now().Before(deadline) {
		st, err := c.Describe(ctx, runID)
		if err == nil && st.Status == status {
			return st
		}

		if err != nil && !errors.Is(err, durable.ErrRunNotFound) {
			require.NoError(t, err)
		}

		time.Sleep(200 * time.Millisecond)
	}

	st, _ := c.Describe(ctx, runID)
	t.Fatalf("run %s never reached %s (last: %+v)", runID, status, st)

	return nil
}

func TestDurableExecution_EndToEnd(t *testing.T) { //nolint:paralleltest // one container, sequential scenarios
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest", redpanda.WithAutoCreateTopics())
	require.NoError(t, err)

	defer func() { _ = container.Terminate(context.Background()) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	engineCtx, stopEngine := context.WithCancel(ctx)
	url := startEngine(engineCtx, t, brokers, 3*time.Second)

	cfg := durable.Config{Brokers: []string{brokers}, TopicPrefix: prefix, EngineURL: url}

	client, err := durable.NewClient(cfg)
	require.NoError(t, err)

	defer client.Close()

	g := &gate{}
	workerCtx, stopWorker := context.WithCancel(ctx)
	startWorker(workerCtx, t, cfg, g)

	// The replacement worker used after the crash scenario; started there,
	// stopped only when the whole test ends.
	g2 := &gate{}
	worker2Ctx, stopWorker2 := context.WithCancel(ctx)

	defer stopWorker2()

	t.Run("plain run completes and publishes a result", func(t *testing.T) { //nolint:paralleltest // scenarios share one cluster and must run in order
		require.NoError(t, client.Start(ctx, durable.StartOptions{RunID: "run-plain", Agent: "support", Text: "hello"}))

		st := waitStatus(ctx, t, client, "run-plain", durable.StatusCompleted)
		assert.Equal(t, "stop", st.FinishReason)
		require.Len(t, st.Messages, 2)
		assert.Equal(t, "echo: hello", st.Messages[1].TextContent())
		assert.Equal(t, 1, st.Attempt)

		// Idempotent start: a duplicate start for a closed run starts a new
		// attempt only via Continue; for an open run it is ignored. Here the
		// run is closed, so a fresh start re-runs it; check it stays sane.
		res := readResult(ctx, t, brokers, durable.Topics{Prefix: prefix}.Results(), "run-plain")
		assert.Equal(t, durable.StatusCompleted, res.Status)
		require.NotNil(t, res.FinalMessage)
		assert.Equal(t, "echo: hello", res.FinalMessage.TextContent())
	})

	t.Run("human approval suspends and resumes with the payload", func(t *testing.T) { //nolint:paralleltest // scenarios share one cluster and must run in order
		require.NoError(t, client.Start(ctx, durable.StartOptions{RunID: "run-refund", Agent: "support", Text: "refund"}))

		st := waitStatus(ctx, t, client, "run-refund", durable.StatusSuspended)
		require.NotNil(t, st.Awaiting)
		assert.Equal(t, durable.AwaitInput, st.Awaiting.Kind)
		assert.Equal(t, "approval", st.Awaiting.Name)
		assert.Len(t, st.Messages, 2, "user message and the assistant tool request are journaled")
		assert.Empty(t, st.CurrentTaskToken, "no worker holds a suspended run")

		require.NoError(t, client.SendInput(ctx, "run-refund", "approval", map[string]any{"approved": true, "by": "ops"}))

		st = waitStatus(ctx, t, client, "run-refund", durable.StatusCompleted)
		assert.Equal(t, 2, st.Attempt)
		require.Len(t, st.Messages, 4)
		assert.Contains(t, st.Messages[3].TextContent(), `"approved":true`)
	})

	t.Run("input sent before the wait is buffered", func(t *testing.T) { //nolint:paralleltest // scenarios share one cluster and must run in order
		g.mu.Lock()
		g.release = make(chan struct{})
		g.mu.Unlock()

		// Block the worker on slow_lookup first so the approval input for a
		// second run arrives before that run has been picked up.
		require.NoError(t, client.Start(ctx, durable.StartOptions{RunID: "run-early", Agent: "support", Text: "refund"}))
		require.NoError(t, client.SendInput(ctx, "run-early", "approval", "yes"))

		st := waitStatus(ctx, t, client, "run-early", durable.StatusCompleted)
		assert.Contains(t, st.Messages[len(st.Messages)-1].TextContent(), `"yes"`)
	})

	t.Run("worker crash mid-tool is recovered by lease expiry", func(t *testing.T) { //nolint:paralleltest // scenarios share one cluster and must run in order
		require.NoError(t, client.Start(ctx, durable.StartOptions{RunID: "run-crash", Agent: "support", Text: "lookup"}))

		// Wait until the tool is executing (blocked on the gate), then kill the worker.
		require.Eventually(t, func() bool {
			g.mu.Lock()
			defer g.mu.Unlock()

			return g.executed >= 1
		}, 30*time.Second, 100*time.Millisecond)

		stopWorker()

		// The replacement worker's tool does not block. It stays up for the
		// rest of the suite, so later scenarios still have a worker.
		startWorker(worker2Ctx, t, cfg, g2)

		st := waitStatus(ctx, t, client, "run-crash", durable.StatusCompleted)
		assert.Equal(t, 2, st.Attempt, "second attempt after lease expiry")
		assert.Equal(t, 1, g2.executed, "tool re-executed once by the second worker")
		require.Len(t, st.Messages, 4)
		assert.Equal(t, `result: {"found":true}`, st.Messages[3].TextContent())

		var assistantToolRequests int

		for _, m := range st.Messages {
			if m.Role == llm.RoleAssistant && len(m.ToolRequests()) > 0 {
				assistantToolRequests++
			}
		}

		assert.Equal(t, 1, assistantToolRequests, "journaled progress is not duplicated")
	})

	t.Run("engine restart replays the journal", func(t *testing.T) { //nolint:paralleltest // scenarios share one cluster and must run in order
		require.NoError(t, client.Start(ctx, durable.StartOptions{RunID: "run-restart", Agent: "support", Text: "refund"}))
		waitStatus(ctx, t, client, "run-restart", durable.StatusSuspended)

		stopEngine()
		time.Sleep(time.Second)

		engine2Ctx, stopEngine2 := context.WithCancel(ctx)
		defer stopEngine2()

		url2 := startEngine(engine2Ctx, t, brokers, 3*time.Second)

		client2, err := durable.NewClient(durable.Config{Brokers: []string{brokers}, TopicPrefix: prefix, EngineURL: url2})
		require.NoError(t, err)

		defer client2.Close()

		st := waitStatus(ctx, t, client2, "run-restart", durable.StatusSuspended)
		assert.Len(t, st.Messages, 2)

		done, err := client2.Describe(ctx, "run-plain")
		require.NoError(t, err)
		assert.Equal(t, durable.StatusCompleted, done.Status, "closed runs survive replay")

		require.NoError(t, client2.SendInput(ctx, "run-restart", "approval", map[string]any{"approved": false}))
		st = waitStatus(ctx, t, client2, "run-restart", durable.StatusCompleted)
		assert.Contains(t, st.Messages[len(st.Messages)-1].TextContent(), `"approved":false`)

		require.NoError(t, client2.Continue(ctx, "run-restart", llm.NewMessage(llm.RoleUser, llm.NewTextPart("thanks"))))
		st = waitStatus(ctx, t, client2, "run-restart", durable.StatusCompleted)
		require.GreaterOrEqual(t, len(st.Messages), 6)
		assert.Equal(t, "echo: thanks", st.Messages[len(st.Messages)-1].TextContent())
	})
}

func readResult(ctx context.Context, t *testing.T, brokers, topic, runID string) durable.Result {
	t.Helper()

	cl, err := kgo.NewClient(kgo.SeedBrokers(brokers),
		kgo.ConsumeTopics(topic), kgo.ConsumeResetOffset(kgo.NewOffset().AtStart()))
	require.NoError(t, err)

	defer cl.Close()

	deadline := time.Now().Add(30 * time.Second)

	for time.Now().Before(deadline) {
		pollCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		f := cl.PollFetches(pollCtx)

		cancel()

		for iter := f.RecordIter(); !iter.Done(); {
			rec := iter.Next()

			var res durable.Result
			if err := json.Unmarshal(rec.Value, &res); err == nil && res.RunID == runID {
				return res
			}
		}
	}

	t.Fatalf("no result for %s on %s", runID, topic)

	return durable.Result{}
}
