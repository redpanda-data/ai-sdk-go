// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package durable

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"
	"sync"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/redpanda-data/ai-sdk-go/agent"
)

// DefaultVersion is used when an agent is registered or a run is started
// without an explicit version.
const DefaultVersion = "v1"

// WorkerOptions tunes a Worker.
type WorkerOptions struct {
	// MaxConcurrent bounds in-flight tasks. Default 8. Anything beyond it stays
	// in the task topic as consumer lag, which is the intended backpressure.
	MaxConcurrent int
	// ConsumerGroup defaults to "durable-worker-<queue>".
	ConsumerGroup string
	Logger        *slog.Logger
}

// Worker hosts agents and executes tasks from one task queue.
type Worker struct {
	cfg    Config
	queue  string
	opts   WorkerOptions
	topics Topics
	log    *slog.Logger

	mu     sync.RWMutex
	agents map[string]agent.Agent // key: name@version

	prod producer
	cl   *kgo.Client
}

// NewWorker creates a worker for taskQueue. Call Register, then Run.
func NewWorker(cfg Config, taskQueue string, opts WorkerOptions) (*Worker, error) {
	if taskQueue == "" {
		return nil, errors.New("durable: task queue is required")
	}

	if opts.MaxConcurrent <= 0 {
		opts.MaxConcurrent = 8
	}

	if opts.ConsumerGroup == "" {
		opts.ConsumerGroup = "durable-worker-" + taskQueue
	}

	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}

	return &Worker{
		cfg:    cfg,
		queue:  taskQueue,
		opts:   opts,
		topics: cfg.topics(),
		log:    opts.Logger.With("task_queue", taskQueue),
		agents: map[string]agent.Agent{},
	}, nil
}

// Register hosts ag under name and version. Version "" means DefaultVersion.
// The agent must include NewInterceptor() in its interceptors for tool-level
// durability and suspension; message-level journaling works regardless.
func (w *Worker) Register(name, version string, ag agent.Agent) {
	if version == "" {
		version = DefaultVersion
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	w.agents[name+"@"+version] = ag
}

// Run consumes tasks until ctx is cancelled. It returns nil on a clean stop.
func (w *Worker) Run(ctx context.Context) error {
	cl, err := kgo.NewClient(w.cfg.kafkaOpts(
		kgo.ClientID("durable-worker"),
		kgo.ConsumerGroup(w.opts.ConsumerGroup),
		kgo.ConsumeTopics(w.topics.Tasks(w.queue)),
		kgo.ConsumeResetOffset(kgo.NewOffset().AtStart()),
		kgo.AllowAutoTopicCreation(),
	)...)
	if err != nil {
		return fmt.Errorf("durable: worker kafka client: %w", err)
	}

	w.cl = cl
	w.prod = &kgoProducer{cl: cl}

	defer cl.Close()

	sem := make(chan struct{}, w.opts.MaxConcurrent)

	var wg sync.WaitGroup

	defer wg.Wait()

	w.log.Info("durable worker started", "group", w.opts.ConsumerGroup)

	for {
		fetches := cl.PollFetches(ctx)
		if fetches.IsClientClosed() || ctx.Err() != nil {
			return nil
		}

		fetches.EachError(func(topic string, p int32, err error) {
			w.log.Error("fetch error", "topic", topic, "partition", p, "error", err)
		})

		for iter := fetches.RecordIter(); !iter.Done(); {
			rec := iter.Next()

			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return nil
			}

			wg.Add(1)

			go func(rec *kgo.Record) {
				defer wg.Done()
				defer func() { <-sem }()

				w.handle(ctx, rec.Value)
			}(rec)
		}
	}
}

func (w *Worker) lookup(name, version string) (agent.Agent, bool) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	ag, ok := w.agents[name+"@"+version]

	return ag, ok
}

func (w *Worker) handle(ctx context.Context, value []byte) {
	task, err := decodeTask(value)
	if err != nil {
		w.log.Error("dropping undecodable task", "error", err)

		return
	}

	logger := w.log.With("run_id", task.RunID, "task_token", task.TaskToken, "attempt", task.Attempt)

	rc := &runContext{task: task, topics: w.topics, prod: w.prod, log: logger}

	ag, ok := w.lookup(task.Agent, task.Version)
	if !ok {
		logger.Error("no agent registered for task", "agent", task.Agent, "version", task.Version)

		if err := rc.report(ctx, unsupported(task)); err != nil {
			logger.Error("report failed", "error", err)
		}

		return
	}

	o := w.execute(ctx, ag, rc)

	logger.Info("attempt finished",
		"completed", o.completed, "suspended", o.suspended, "failed", o.failed,
		"interrupted", o.interrupted, "finish_reason", o.finishReason, "error", o.err)

	if err := rc.report(ctx, o); err != nil {
		logger.Error("report failed; lease expiry will re-dispatch", "error", err)
	}
}

// execute runs one attempt, turning a panic in agent or tool code into a
// retryable failure rather than taking the worker process down.
func (*Worker) execute(ctx context.Context, ag agent.Agent, rc *runContext) outcome {
	var o outcome

	func() {
		defer func() {
			if r := recover(); r != nil {
				rc.log.Error("agent panicked", "panic", r, "stack", string(debug.Stack()))

				o = outcome{failed: true, retryable: true, err: fmt.Sprintf("panic: %v", r)}
			}
		}()

		o = executeAttempt(ctx, ag, rc)
	}()

	return o
}
