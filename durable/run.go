// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package durable

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// runContext is the per-attempt state the worker and the interceptor share.
// It travels in the context so the interceptor can be installed once on the
// agent and still address the current attempt.
type runContext struct {
	task   Task
	topics Topics
	prod   producer
	log    *slog.Logger
	cancel context.CancelFunc

	mu        sync.Mutex
	persisted int       // messages already journaled
	awaiting  *Awaiting // set by the first suspending tool call
	suspended bool
	sendErr   error
}

type runContextKey struct{}

func withRunContext(ctx context.Context, rc *runContext) context.Context {
	return context.WithValue(ctx, runContextKey{}, rc)
}

func runContextFrom(ctx context.Context) *runContext {
	rc, _ := ctx.Value(runContextKey{}).(*runContext)

	return rc
}

func (rc *runContext) command(typ string) Command {
	cmd := NewCommand(typ, rc.task.RunID)
	cmd.TaskToken = rc.task.TaskToken

	return cmd
}

func (rc *runContext) send(ctx context.Context, cmd Command) error {
	// Use a context detached from the run's cancellation so a suspension can
	// still be reported after the run context was cancelled.
	sendCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
	defer cancel()

	if err := rc.prod.Produce(sendCtx, rc.topics.Commands(), rc.task.RunID, cmd); err != nil {
		rc.mu.Lock()
		if rc.sendErr == nil {
			rc.sendErr = err
		}
		rc.mu.Unlock()

		return err
	}

	return nil
}

// flushMessages journals every session message not yet persisted. Called after
// each agent event; the tool-results message has no event of its own, so this
// is the only place it gets captured.
//
// Once the worker's context is cancelled it journals nothing more. A shutting
// down worker reports no outcome, so the engine re-dispatches on lease expiry;
// anything written after cancellation would be progress no attempt owns. This
// matters most for the tool-results message, which the agent fills with
// "interrupted, outcome unknown" placeholders for calls that never finished.
func (rc *runContext) flushMessages(ctx context.Context, sess *session.State) error {
	if ctx.Err() != nil {
		return nil //nolint:nilerr // cancellation is not an attempt failure: reporting nothing lets the lease expire
	}

	rc.mu.Lock()
	if rc.suspended {
		rc.mu.Unlock()

		return nil
	}

	start := rc.persisted
	rc.mu.Unlock()

	for i := start; i < len(sess.Messages); i++ {
		if ctx.Err() != nil {
			return nil //nolint:nilerr // see above: stop journaling, do not fail the attempt
		}

		msg := llm.CloneMessage(sess.Messages[i])
		cmd := rc.command(CommandAppendMessage)
		cmd.MessageIndex = i
		cmd.Message = &msg

		if err := rc.send(ctx, cmd); err != nil {
			return err
		}

		rc.mu.Lock()
		rc.persisted = i + 1
		rc.mu.Unlock()
	}

	return nil
}

// resetMessages journals the whole message list after compaction rewrote it.
func (rc *runContext) resetMessages(ctx context.Context, sess *session.State) error {
	if ctx.Err() != nil {
		return nil //nolint:nilerr // see flushMessages: cancellation is not an attempt failure
	}

	cmd := rc.command(CommandResetMessages)
	cmd.Messages = make([]llm.Message, len(sess.Messages))

	for i, m := range sess.Messages {
		cmd.Messages[i] = llm.CloneMessage(m)
	}

	if err := rc.send(ctx, cmd); err != nil {
		return err
	}

	rc.mu.Lock()
	rc.persisted = len(sess.Messages)
	rc.mu.Unlock()

	return nil
}

// Interceptor makes tool execution durable. Install it on the agent with
// llmagent.WithInterceptors(durable.NewInterceptor()).
//
//   - A tool call whose result was journaled by a previous attempt returns that
//     result without executing again.
//   - A tool call that a previous attempt suspended on returns the delivered
//     input (or timer) payload without executing.
//   - A tool that returns WaitForInput/SleepUntil suspends the run: the attempt
//     ends, the engine parks the run, and no worker is held.
//   - Every other result is journaled before it is handed back to the agent.
type Interceptor struct{}

var _ agent.ToolInterceptor = Interceptor{}

// NewInterceptor returns the durable tool interceptor.
func NewInterceptor() Interceptor { return Interceptor{} }

// InterceptToolExecution implements agent.ToolInterceptor.
func (Interceptor) InterceptToolExecution(
	ctx context.Context,
	info *agent.ToolCallInfo,
	next agent.ToolExecutionNext,
) (*llm.ToolResponsePart, error) {
	rc := runContextFrom(ctx)
	if rc == nil {
		return next(ctx, info)
	}

	id := info.Req.ID

	if payload, ok := rc.task.Delivered[id]; ok {
		return &llm.ToolResponsePart{ID: id, Name: info.Req.Name, Result: payload}, nil
	}

	if prev, ok := rc.task.ToolResults[id]; ok {
		cp := *prev

		return &cp, nil
	}

	resp, err := next(ctx, info)

	if resp != nil && !resp.IsError {
		if s, ok := ParseSuspension(resp.Result); ok {
			rc.suspend(&Awaiting{
				Kind:       s.Kind,
				ToolCallID: id,
				ToolName:   info.Req.Name,
				Name:       s.Name,
				FireAt:     fireAt(s),
			})

			return resp, nil
		}
	}

	if err != nil || resp == nil || ctx.Err() != nil {
		// Tool errors are the agent's business (it sees an error result and
		// decides what to do). Nothing durable happened, so nothing to journal.
		return resp, err
	}

	cmd := rc.command(CommandRecordToolResult)
	cp := *resp
	cmd.ToolResult = &cp

	if sendErr := rc.send(ctx, cmd); sendErr != nil {
		return nil, sendErr
	}

	return resp, nil
}

func fireAt(s *Suspension) *time.Time {
	if s.Kind != AwaitTimer {
		return nil
	}

	t := s.FireAt

	return &t
}

func (rc *runContext) suspend(aw *Awaiting) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	if rc.suspended {
		return
	}

	rc.suspended = true
	rc.awaiting = aw

	if rc.cancel != nil {
		rc.cancel()
	}
}

// outcome is what one attempt reports back to the engine.
type outcome struct {
	completed    bool
	finishReason agent.FinishReason
	usage        *llm.TokenUsage
	failed       bool
	err          string
	retryable    bool
	suspended    bool
	awaiting     *Awaiting
	interrupted  bool // worker shutdown: report nothing, lease expiry re-dispatches
}

// executeAttempt runs one attempt of the agent against the task's session and
// returns what to report. It never returns before every message produced by
// the agent is journaled (or the journal write failed).
func executeAttempt(ctx context.Context, ag agent.Agent, rc *runContext) outcome {
	sess := &session.State{
		ID:       rc.task.RunID,
		Messages: rc.task.Messages,
		Metadata: rc.task.Metadata,
	}
	if sess.Metadata == nil {
		sess.Metadata = map[string]any{}
	}

	rc.persisted = len(sess.Messages)

	runCtx, cancel := context.WithCancel(withRunContext(ctx, rc))
	defer cancel()

	rc.cancel = cancel
	inv := agent.NewInvocationMetadata(sess, ag.Info())

	var (
		end     *agent.InvocationEndEvent
		termErr error
	)

	for evt, err := range ag.Run(runCtx, inv) {
		if err != nil {
			termErr = err

			break
		}

		if _, ok := evt.(agent.CompactionEvent); ok {
			if rErr := rc.resetMessages(ctx, sess); rErr != nil {
				return outcome{failed: true, retryable: true, err: rErr.Error()}
			}

			continue
		}

		if fErr := rc.flushMessages(ctx, sess); fErr != nil {
			return outcome{failed: true, retryable: true, err: fErr.Error()}
		}

		if e, ok := evt.(agent.InvocationEndEvent); ok {
			end = &e

			break
		}
	}

	rc.mu.Lock()
	suspended, awaiting, sendErr := rc.suspended, rc.awaiting, rc.sendErr
	rc.mu.Unlock()

	switch {
	case suspended:
		return outcome{suspended: true, awaiting: awaiting}
	case sendErr != nil:
		return outcome{failed: true, retryable: true, err: sendErr.Error()}
	case termErr != nil:
		if ctx.Err() != nil {
			return outcome{interrupted: true}
		}

		return outcome{failed: true, retryable: true, err: termErr.Error()}
	case end == nil:
		if ctx.Err() != nil {
			return outcome{interrupted: true}
		}

		return outcome{failed: true, retryable: true, err: "agent ended without a terminal event"}
	}

	switch end.FinishReason {
	case agent.FinishReasonInterrupted:
		return outcome{interrupted: true}
	case agent.FinishReasonError:
		return outcome{failed: true, retryable: false, err: "agent finished with error"}
	case agent.FinishReasonStop, agent.FinishReasonMaxTurns, agent.FinishReasonLength,
		agent.FinishReasonContextOverflow, agent.FinishReasonInputRequired, agent.FinishReasonTransfer:
		return outcome{completed: true, finishReason: end.FinishReason, usage: end.Usage}
	default:
		return outcome{completed: true, finishReason: end.FinishReason, usage: end.Usage}
	}
}

// report sends the attempt's outcome to the engine.
func (rc *runContext) report(ctx context.Context, o outcome) error {
	switch {
	case o.interrupted:
		return nil
	case o.suspended:
		cmd := rc.command(CommandRunSuspended)
		cmd.Awaiting = o.awaiting

		return rc.send(ctx, cmd)
	case o.failed:
		cmd := rc.command(CommandAttemptFailed)
		cmd.Error = o.err
		cmd.Retryable = new(o.retryable)

		return rc.send(ctx, cmd)
	default:
		cmd := rc.command(CommandRunCompleted)
		cmd.FinishReason = string(o.finishReason)
		cmd.Usage = o.usage

		return rc.send(ctx, cmd)
	}
}

// errUnsupportedAgent is reported when a worker receives a task for an agent
// or version it does not host.
var errUnsupportedAgent = errors.New("durable: unsupported agent version")

func unsupported(task Task) outcome {
	return outcome{
		failed:    true,
		retryable: false,
		err:       fmt.Sprintf("%s: %s@%s", errUnsupportedAgent.Error(), task.Agent, task.Version),
	}
}

func decodeTask(b []byte) (Task, error) {
	var t Task
	if err := json.Unmarshal(b, &t); err != nil {
		return Task{}, fmt.Errorf("durable: decode task: %w", err)
	}

	return t, nil
}
