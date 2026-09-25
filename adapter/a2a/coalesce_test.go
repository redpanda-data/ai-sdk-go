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

package a2a

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// newCoalesceTestExecutor builds an Executor configured with cfg for tests
// that drive processEvents directly, without a real agent or runner.
func newCoalesceTestExecutor(cfg DeltaCoalescing) *Executor {
	return &Executor{
		log:      slog.Default(),
		coalesce: cfg,
	}
}

// testReqCtx returns a RequestContext for tests that call processEvents
// directly rather than through Execute.
func testReqCtx() *a2asrv.RequestContext {
	return &a2asrv.RequestContext{
		ContextID: "test-context",
		TaskID:    a2a.TaskID("test-task"),
	}
}

// recordingQueue is a minimal eventqueue.Queue that only records what is
// written to it, in order, behind a mutex. Tests that drive processEvents
// directly do not need a real queue's subscriber fan-out.
type recordingQueue struct {
	mu     sync.Mutex
	events []a2a.Event
}

func newRecordingQueue() *recordingQueue {
	return &recordingQueue{}
}

func (q *recordingQueue) Write(ctx context.Context, event a2a.Event) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	q.events = append(q.events, event)

	return nil
}

func (q *recordingQueue) WriteVersioned(ctx context.Context, event a2a.Event, _ a2a.TaskVersion) error {
	return q.Write(ctx, event)
}

func (q *recordingQueue) Read(_ context.Context) (a2a.Event, a2a.TaskVersion, error) {
	return nil, 0, errors.New("recordingQueue: Read is not supported")
}

func (q *recordingQueue) Close() error {
	return nil
}

func (q *recordingQueue) snapshot() []a2a.Event {
	q.mu.Lock()
	defer q.mu.Unlock()

	out := make([]a2a.Event, len(q.events))
	copy(out, q.events)

	return out
}

// flakyQueue wraps a recordingQueue and fails writes for which shouldFail
// returns true, without recording them. attempts counts every call,
// successful or not, so a test can prove that nothing was even attempted
// after some point, not just that nothing new was recorded.
type flakyQueue struct {
	*recordingQueue

	shouldFail func(a2a.Event) bool
	attempts   atomic.Int64
}

func newFlakyQueue(shouldFail func(a2a.Event) bool) *flakyQueue {
	return &flakyQueue{recordingQueue: newRecordingQueue(), shouldFail: shouldFail}
}

func (q *flakyQueue) Write(ctx context.Context, event a2a.Event) error {
	q.attempts.Add(1)

	if q.shouldFail(event) {
		return errors.New("simulated write failure")
	}

	return q.recordingQueue.Write(ctx, event)
}

// seqStep is one step of a hand-built agent event sequence: an optional
// sleep before the step, then the event (or error) to yield.
type seqStep struct {
	sleep time.Duration
	event agent.Event
	err   error
}

// seq builds an iter.Seq2[agent.Event, error] out of steps, sleeping
// between them as requested. A sleep inside a synctest bubble advances the
// bubble's fake clock instead of real time.
func seq(steps ...seqStep) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		for _, step := range steps {
			if step.sleep > 0 {
				time.Sleep(step.sleep)
			}

			if !yield(step.event, step.err) {
				return
			}
		}
	}
}

// events builds a sequence of events with no delays and no error.
func events(evs ...agent.Event) iter.Seq2[agent.Event, error] {
	steps := make([]seqStep, len(evs))
	for i, ev := range evs {
		steps[i] = seqStep{event: ev}
	}

	return seq(steps...)
}

// eventsThenErr builds a sequence of events followed by a runner error, the
// shape processEvents sees when the model stream fails or is canceled.
func eventsThenErr(err error, evs ...agent.Event) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		for _, ev := range evs {
			if !yield(ev, nil) {
				return
			}
		}

		yield(nil, err)
	}
}

func deltaEvent(text string) agent.AssistantDeltaEvent {
	return agent.AssistantDeltaEvent{Delta: llm.ContentPartEvent{Part: &llm.TextPart{Text: text}}}
}

func messageEvent(text string) agent.MessageEvent {
	return agent.MessageEvent{Response: llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(text))}}
}

func modelCallStatus() agent.StatusEvent {
	return agent.StatusEvent{Stage: agent.StatusStageModelCall}
}

func streamReset() agent.StreamResetEvent {
	return agent.StreamResetEvent{Attempt: 1, Reason: "retry"}
}

func toolResponseEvent(name string) agent.ToolResponseEvent {
	return agent.ToolResponseEvent{Response: llm.ToolResponsePart{ID: "1", Name: name}}
}

func invocationEnd() agent.InvocationEndEvent {
	return agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}
}

func filterArtifacts(evs []a2a.Event) []*a2a.TaskArtifactUpdateEvent {
	var out []*a2a.TaskArtifactUpdateEvent

	for _, ev := range evs {
		if a, ok := ev.(*a2a.TaskArtifactUpdateEvent); ok {
			out = append(out, a)
		}
	}

	return out
}

func filterStatuses(evs []a2a.Event) []*a2a.TaskStatusUpdateEvent {
	var out []*a2a.TaskStatusUpdateEvent

	for _, ev := range evs {
		if s, ok := ev.(*a2a.TaskStatusUpdateEvent); ok {
			out = append(out, s)
		}
	}

	return out
}

func artifactText(a *a2a.TaskArtifactUpdateEvent) string {
	var text strings.Builder

	for _, part := range a.Artifact.Parts {
		if tp, ok := part.(a2a.TextPart); ok {
			text.WriteString(tp.Text)
		}
	}

	return text.String()
}

func concatText(as []*a2a.TaskArtifactUpdateEvent) string {
	var text strings.Builder

	for _, a := range as {
		text.WriteString(artifactText(a))
	}

	return text.String()
}

// TestDeltaCoalescing_DisabledIsByteIdentical pins the zero-value contract:
// with coalescing off, the executor must send the exact same, fully
// ordered event sequence it always has, byte for byte, for every event
// kind processEvents and Cancel can produce.
func TestDeltaCoalescing_DisabledIsByteIdentical(t *testing.T) {
	t.Parallel()

	t.Run("deltas and message", func(t *testing.T) {
		t.Parallel()

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{})
		reqCtx := testReqCtx()

		require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
			deltaEvent("a"),
			deltaEvent("b"),
			deltaEvent("c"),
			messageEvent("abc"),
			invocationEnd(),
		)))

		got := queue.snapshot()
		artifacts := filterArtifacts(got)
		require.Len(t, artifacts, 4, "3 delta events plus the empty LastChunk event")

		assert.False(t, artifacts[0].Append)
		assert.False(t, artifacts[0].LastChunk)
		require.Len(t, artifacts[0].Artifact.Parts, 1)
		assert.Equal(t, "a", artifactText(artifacts[0]))

		for i, want := range []string{"b", "c"} {
			a := artifacts[i+1]
			assert.True(t, a.Append)
			assert.False(t, a.LastChunk)
			require.Len(t, a.Artifact.Parts, 1)
			assert.Equal(t, want, artifactText(a))
			assert.Equal(t, artifacts[0].Artifact.ID, a.Artifact.ID)
		}

		last := artifacts[3]
		assert.True(t, last.Append)
		assert.True(t, last.LastChunk)
		assert.Empty(t, last.Artifact.Parts)

		statuses := filterStatuses(got)
		require.Len(t, statuses, 2)
		assert.Equal(t, a2a.TaskStateWorking, statuses[0].Status.State)
		assert.False(t, statuses[0].Final)
		assert.Equal(t, a2a.TaskStateCompleted, statuses[1].Status.State)
		assert.True(t, statuses[1].Final)
	})

	t.Run("stream reset", func(t *testing.T) {
		t.Parallel()

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{})
		reqCtx := testReqCtx()

		require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
			deltaEvent("a"),
			deltaEvent("b"),
			streamReset(),
			deltaEvent("c"),
			invocationEnd(),
		)))

		got := queue.snapshot()
		artifacts := filterArtifacts(got)
		require.Len(t, artifacts, 4, "a, b, the empty LastChunk from reset, c")

		assert.False(t, artifacts[0].Append)
		assert.Equal(t, "a", artifactText(artifacts[0]))

		assert.True(t, artifacts[1].Append)
		assert.False(t, artifacts[1].LastChunk)
		assert.Equal(t, "b", artifactText(artifacts[1]))
		assert.Equal(t, artifacts[0].Artifact.ID, artifacts[1].Artifact.ID)

		assert.True(t, artifacts[2].Append)
		assert.True(t, artifacts[2].LastChunk)
		assert.Empty(t, artifacts[2].Artifact.Parts)
		assert.Equal(t, artifacts[0].Artifact.ID, artifacts[2].Artifact.ID)

		assert.False(t, artifacts[3].Append)
		assert.Equal(t, "c", artifactText(artifacts[3]))
		assert.NotEqual(t, artifacts[0].Artifact.ID, artifacts[3].Artifact.ID)

		statuses := filterStatuses(got)
		require.Len(t, statuses, 1)
		assert.True(t, statuses[0].Final)
	})

	t.Run("model_call status writes nothing by itself", func(t *testing.T) {
		t.Parallel()

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{})
		reqCtx := testReqCtx()

		require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
			deltaEvent("a"),
			modelCallStatus(),
			deltaEvent("b"),
			invocationEnd(),
		)))

		got := queue.snapshot()
		artifacts := filterArtifacts(got)
		require.Len(t, artifacts, 2, "model_call only resets the artifact ID; it writes nothing")

		assert.False(t, artifacts[0].Append)
		assert.Equal(t, "a", artifactText(artifacts[0]))
		assert.False(t, artifacts[1].Append)
		assert.Equal(t, "b", artifactText(artifacts[1]))
		assert.NotEqual(t, artifacts[0].Artifact.ID, artifacts[1].Artifact.ID)

		statuses := filterStatuses(got)
		require.Len(t, statuses, 1)
		assert.True(t, statuses[0].Final)
	})

	t.Run("tool response writes history directly", func(t *testing.T) {
		t.Parallel()

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{})
		reqCtx := testReqCtx()

		require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
			deltaEvent("a"),
			toolResponseEvent("get_weather"),
			invocationEnd(),
		)))

		got := queue.snapshot()
		require.Len(t, got, 3, "artifact, tool-response history status, final status - no extra flush")

		_, ok := got[0].(*a2a.TaskArtifactUpdateEvent)
		require.True(t, ok)

		history, ok := got[1].(*a2a.TaskStatusUpdateEvent)
		require.True(t, ok)
		assert.Equal(t, a2a.TaskStateWorking, history.Status.State)
		assert.False(t, history.Final)

		final, ok := got[2].(*a2a.TaskStatusUpdateEvent)
		require.True(t, ok)
		assert.True(t, final.Final)
	})

	errCases := []struct {
		name      string
		err       error
		wantState a2a.TaskState
	}{
		{name: "error: generic failure", err: errors.New("boom"), wantState: a2a.TaskStateFailed},
		{name: "error: context overflow", err: llm.ErrContextOverflow, wantState: a2a.TaskStateFailed},
		{name: "error: canceled via background context", err: context.Canceled, wantState: a2a.TaskStateCanceled},
	}

	for _, tc := range errCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			queue := newRecordingQueue()
			exec := newCoalesceTestExecutor(DeltaCoalescing{})
			reqCtx := testReqCtx()

			require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, eventsThenErr(tc.err, deltaEvent("a"))))

			got := queue.snapshot()
			require.Len(t, got, 2, "off mode never buffers, so there is no tail to flush before the terminal status")

			_, ok := got[0].(*a2a.TaskArtifactUpdateEvent)
			require.True(t, ok)

			status, ok := got[1].(*a2a.TaskStatusUpdateEvent)
			require.True(t, ok)
			assert.Equal(t, tc.wantState, status.Status.State)
			assert.True(t, status.Final)
		})
	}

	t.Run("missing invocation end", func(t *testing.T) {
		t.Parallel()

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{})
		reqCtx := testReqCtx()

		err := exec.processEvents(context.Background(), reqCtx, queue, events(deltaEvent("a")))
		require.Error(t, err)

		got := queue.snapshot()
		require.Len(t, got, 2)

		_, ok := got[0].(*a2a.TaskArtifactUpdateEvent)
		require.True(t, ok)

		status, ok := got[1].(*a2a.TaskStatusUpdateEvent)
		require.True(t, ok)
		assert.Equal(t, a2a.TaskStateFailed, status.Status.State)
		assert.True(t, status.Final)
	})

	t.Run("cancel never registers a writer and does not touch the running loop", func(t *testing.T) {
		t.Parallel()

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{})
		reqCtx := testReqCtx()

		stalled := make(chan struct{})
		release := make(chan struct{})

		seqFn := func(yield func(agent.Event, error) bool) {
			if !yield(deltaEvent("a"), nil) {
				return
			}

			close(stalled)
			<-release

			if !yield(deltaEvent("b"), nil) {
				return
			}

			yield(invocationEnd(), nil)
		}

		done := make(chan struct{})

		go func() {
			defer close(done)

			_ = exec.processEvents(context.Background(), reqCtx, queue, seqFn)
		}()

		<-stalled

		_, registered := exec.activeWriters.Load(reqCtx.TaskID)
		assert.False(t, registered, "coalescing off must never register a writer")

		require.NoError(t, exec.Cancel(context.Background(), reqCtx, queue))

		close(release)
		<-done

		// Cancel and the still-running loop write independently, exactly as
		// the base code always has: Cancel has no writer to go through, so
		// it never touches the loop, which keeps sending its own deltas and
		// its own completion status.
		got := queue.snapshot()
		artifacts := filterArtifacts(got)
		require.Len(t, artifacts, 2)
		assert.Equal(t, "a", artifactText(artifacts[0]))
		assert.Equal(t, "b", artifactText(artifacts[1]))
		assert.True(t, artifacts[1].Append, "nothing between a and b resets the artifact ID, so b appends to a's artifact")
		assert.Equal(t, artifacts[0].Artifact.ID, artifacts[1].Artifact.ID)

		var sawCanceled, sawFinal bool

		for _, status := range filterStatuses(got) {
			if status.Status.State == a2a.TaskStateCanceled {
				sawCanceled = true
			}

			if status.Final {
				sawFinal = true
			}
		}

		assert.True(t, sawCanceled, "Cancel's own write still lands on the queue")
		assert.True(t, sawFinal, "the loop's own completion status still lands too")
	})
}

// TestDeltaCoalescing_FirstDeltaImmediate checks that the leading edge of a
// streamed artifact goes out at once, before any time-based trigger could
// fire, so time to first token does not change.
func TestDeltaCoalescing_FirstDeltaImmediate(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: 100 * time.Millisecond, MaxBytes: 512})
		reqCtx := testReqCtx()

		release := make(chan struct{})
		seqFn := func(yield func(agent.Event, error) bool) {
			if !yield(deltaEvent("hello"), nil) {
				return
			}

			<-release

			yield(invocationEnd(), nil)
		}

		done := make(chan struct{})

		go func() {
			defer close(done)

			_ = exec.processEvents(context.Background(), reqCtx, queue, seqFn)
		}()

		synctest.Wait()

		artifacts := filterArtifacts(queue.snapshot())
		require.Len(t, artifacts, 1)
		assert.False(t, artifacts[0].Append)
		assert.Equal(t, "hello", artifactText(artifacts[0]))

		close(release)
		<-done
	})
}

// TestDeltaCoalescing_IntervalFlush drives 100 deltas 10ms apart with a
// 100ms interval, and checks that the number of artifact events is bounded
// by time rather than by the delta count, while every byte still arrives.
func TestDeltaCoalescing_IntervalFlush(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		const n = 100

		var want strings.Builder

		steps := make([]seqStep, 0, n+2)

		for i := range n {
			chunk := strconv.Itoa(i % 10)
			want.WriteString(chunk)
			steps = append(steps, seqStep{sleep: 10 * time.Millisecond, event: deltaEvent(chunk)})
		}

		steps = append(steps, seqStep{event: messageEvent(want.String())})
		steps = append(steps, seqStep{event: invocationEnd()})

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: 100 * time.Millisecond})
		reqCtx := testReqCtx()

		require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, seq(steps...)))

		artifacts := filterArtifacts(queue.snapshot())
		assert.GreaterOrEqual(t, len(artifacts), 10)
		assert.LessOrEqual(t, len(artifacts), 12)
		assert.Equal(t, want.String(), concatText(artifacts))

		assert.False(t, artifacts[0].Append)

		for _, a := range artifacts[1:] {
			assert.True(t, a.Append)

			if len(a.Artifact.Parts) > 0 {
				assert.Len(t, a.Artifact.Parts, 1)
			}
		}
	})
}

// TestDeltaCoalescing_SizeFlush checks the byte-size trigger in isolation,
// with an interval long enough that it never fires.
func TestDeltaCoalescing_SizeFlush(t *testing.T) {
	t.Parallel()

	queue := newRecordingQueue()
	exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour, MaxBytes: 16})
	reqCtx := testReqCtx()

	require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
		deltaEvent("aaaaa"), // leading edge, sent at once
		deltaEvent("bbbbb"), // buf: 5 bytes
		deltaEvent("ccccc"), // buf: 10 bytes
		deltaEvent("ddddd"), // buf: 15 bytes
		deltaEvent("eeeee"), // buf: 20 bytes >= 16, flushes
		messageEvent("aaaaabbbbbcccccdddddeeeee"),
		invocationEnd(),
	)))

	artifacts := filterArtifacts(queue.snapshot())
	require.Len(t, artifacts, 3, "leading edge, one size-triggered flush, one empty LastChunk")

	assert.False(t, artifacts[0].Append)
	assert.Equal(t, "aaaaa", artifactText(artifacts[0]))

	assert.True(t, artifacts[1].Append)
	assert.False(t, artifacts[1].LastChunk)
	assert.Equal(t, "bbbbbcccccdddddeeeee", artifactText(artifacts[1]))

	assert.True(t, artifacts[2].LastChunk)
	assert.Empty(t, artifacts[2].Artifact.Parts)
}

// TestDeltaCoalescing_TimerFlushOnStall checks that a stalled model does not
// hold buffered text past Interval: the timer flushes it on its own, before
// the next real event arrives.
func TestDeltaCoalescing_TimerFlushOnStall(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: 100 * time.Millisecond})
		reqCtx := testReqCtx()

		seqFn := func(yield func(agent.Event, error) bool) {
			for _, text := range []string{"one", "two", "three"} {
				if !yield(deltaEvent(text), nil) {
					return
				}
			}

			time.Sleep(500 * time.Millisecond)

			yield(invocationEnd(), nil)
		}

		done := make(chan struct{})

		go func() {
			defer close(done)

			_ = exec.processEvents(context.Background(), reqCtx, queue, seqFn)
		}()

		// Sleep well past Interval but short of the 500ms stall, so the fake
		// clock is forced through the flush timer's deadline before we look.
		// A bare synctest.Wait() here would return immediately: the goroutine
		// is already durably blocked in its own 500ms sleep, and Wait does
		// not itself advance the clock to fire timers that nothing is
		// waiting on.
		time.Sleep(200 * time.Millisecond)
		synctest.Wait()

		artifacts := filterArtifacts(queue.snapshot())
		require.Len(t, artifacts, 2, "leading delta plus one interval flush of the stalled buffer")

		assert.False(t, artifacts[0].Append)
		assert.Equal(t, "one", artifactText(artifacts[0]))

		assert.True(t, artifacts[1].Append)
		assert.False(t, artifacts[1].LastChunk)
		assert.Equal(t, "twothree", artifactText(artifacts[1]))

		<-done
	})
}

// TestDeltaCoalescing_MessageEventCarriesTail checks that the pending text
// rides in the existing LastChunk event at MessageEvent, rather than
// getting its own event in addition to an empty LastChunk.
func TestDeltaCoalescing_MessageEventCarriesTail(t *testing.T) {
	t.Parallel()

	queue := newRecordingQueue()
	exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
	reqCtx := testReqCtx()

	require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
		deltaEvent("hello"),
		deltaEvent(" world"),
		messageEvent("hello world"),
		invocationEnd(),
	)))

	got := queue.snapshot()
	require.Len(t, got, 4, "leading delta, tail LastChunk, working status, final status")

	tail, ok := got[1].(*a2a.TaskArtifactUpdateEvent)
	require.True(t, ok)
	assert.True(t, tail.LastChunk)
	require.Len(t, tail.Artifact.Parts, 1)
	assert.Equal(t, " world", artifactText(tail))

	working, ok := got[2].(*a2a.TaskStatusUpdateEvent)
	require.True(t, ok)
	assert.Equal(t, a2a.TaskStateWorking, working.Status.State)
}

// TestDeltaCoalescing_StreamReset checks that a retry flushes the pending
// text into the abandoned artifact's LastChunk event, and that the next
// delta opens a fresh artifact.
func TestDeltaCoalescing_StreamReset(t *testing.T) {
	t.Parallel()

	queue := newRecordingQueue()
	exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
	reqCtx := testReqCtx()

	require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
		deltaEvent("partial"),
		deltaEvent(" more"), // buffered, still pending when the reset arrives
		streamReset(),
		deltaEvent("restart"),
		invocationEnd(),
	)))

	artifacts := filterArtifacts(queue.snapshot())
	require.Len(t, artifacts, 3)

	assert.False(t, artifacts[0].Append)
	assert.Equal(t, "partial", artifactText(artifacts[0]))

	tail := artifacts[1]
	assert.True(t, tail.LastChunk)
	require.Len(t, tail.Artifact.Parts, 1, "the pending ' more' text must ride in the LastChunk event")
	assert.Equal(t, " more", artifactText(tail))
	assert.Equal(t, artifacts[0].Artifact.ID, tail.Artifact.ID)

	restarted := artifacts[2]
	assert.False(t, restarted.Append)
	assert.NotEqual(t, artifacts[0].Artifact.ID, restarted.Artifact.ID)
	assert.Equal(t, "restart", artifactText(restarted))
}

// TestDeltaCoalescing_ModelCallStatusFlushes checks that a model_call status
// flushes the pending text as a non-final append to the OLD artifact,
// before the artifact ID resets for the next model call.
func TestDeltaCoalescing_ModelCallStatusFlushes(t *testing.T) {
	t.Parallel()

	queue := newRecordingQueue()
	exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
	reqCtx := testReqCtx()

	require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
		deltaEvent("lead"),
		deltaEvent("tail"),
		modelCallStatus(),
		deltaEvent("newlead"),
		invocationEnd(),
	)))

	artifacts := filterArtifacts(queue.snapshot())
	require.Len(t, artifacts, 3)

	lead, tail, newLead := artifacts[0], artifacts[1], artifacts[2]

	assert.False(t, lead.Append)
	assert.Equal(t, "lead", artifactText(lead))

	assert.True(t, tail.Append)
	assert.False(t, tail.LastChunk)
	assert.Equal(t, lead.Artifact.ID, tail.Artifact.ID, "the flush before the reset targets the old artifact")
	assert.Equal(t, "tail", artifactText(tail))

	assert.False(t, newLead.Append)
	assert.NotEqual(t, lead.Artifact.ID, newLead.Artifact.ID)
	assert.Equal(t, "newlead", artifactText(newLead))
}

// TestDeltaCoalescing_ToolResponseFlushesFirst checks that a tool-response
// history status is preceded by a flush of any pending text.
func TestDeltaCoalescing_ToolResponseFlushesFirst(t *testing.T) {
	t.Parallel()

	queue := newRecordingQueue()
	exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
	reqCtx := testReqCtx()

	require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
		deltaEvent("lead"),
		deltaEvent("buffered"),
		toolResponseEvent("get_weather"),
		invocationEnd(),
	)))

	got := queue.snapshot()
	require.Len(t, got, 4)

	_, ok := got[0].(*a2a.TaskArtifactUpdateEvent)
	require.True(t, ok)

	flush, ok := got[1].(*a2a.TaskArtifactUpdateEvent)
	require.True(t, ok, "the buffered tail must flush before the tool-response history status")
	assert.Equal(t, "buffered", artifactText(flush))

	history, ok := got[2].(*a2a.TaskStatusUpdateEvent)
	require.True(t, ok)
	assert.Equal(t, a2a.TaskStateWorking, history.Status.State)
}

// TestDeltaCoalescing_ErrorPaths checks that every terminal error path
// flushes pending text before writing its status.
func TestDeltaCoalescing_ErrorPaths(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		err       error
		wantState a2a.TaskState
	}{
		{name: "generic failure", err: errors.New("boom"), wantState: a2a.TaskStateFailed},
		{name: "context overflow", err: llm.ErrContextOverflow, wantState: a2a.TaskStateFailed},
		{name: "canceled", err: context.Canceled, wantState: a2a.TaskStateCanceled},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			queue := newRecordingQueue()
			exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
			reqCtx := testReqCtx()

			require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, eventsThenErr(tc.err,
				deltaEvent("lead"),
				deltaEvent("buffered"),
			)))

			got := queue.snapshot()
			require.Len(t, got, 3, "leading delta, buffered-tail flush, terminal status")

			_, ok := got[0].(*a2a.TaskArtifactUpdateEvent)
			require.True(t, ok)

			flush, ok := got[1].(*a2a.TaskArtifactUpdateEvent)
			require.True(t, ok, "the buffered tail must flush before the terminal status")
			assert.Equal(t, "buffered", artifactText(flush))

			status, ok := got[2].(*a2a.TaskStatusUpdateEvent)
			require.True(t, ok)
			assert.Equal(t, tc.wantState, status.Status.State)
			assert.True(t, status.Final)
		})
	}
}

// TestDeltaCoalescing_ExternalCancel checks that Executor.Cancel flushes a
// running task's buffered text before writing the canceled status when it
// can find the task's writer, and falls back to writing the status alone
// when it cannot.
func TestDeltaCoalescing_ExternalCancel(t *testing.T) {
	t.Parallel()

	t.Run("writer registered", func(t *testing.T) {
		t.Parallel()

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
		reqCtx := testReqCtx()

		stalled := make(chan struct{})
		release := make(chan struct{})

		seqFn := func(yield func(agent.Event, error) bool) {
			if !yield(deltaEvent("lead"), nil) {
				return
			}

			if !yield(deltaEvent("buffered"), nil) {
				return
			}

			close(stalled)
			<-release

			// The model keeps streaming after Cancel already reached this
			// writer; a closed writer must drop these silently.
			if !yield(deltaEvent("ignored-1"), nil) {
				return
			}

			yield(deltaEvent("ignored-2"), nil)
		}

		done := make(chan struct{})

		go func() {
			defer close(done)

			_ = exec.processEvents(context.Background(), reqCtx, queue, seqFn)
		}()

		<-stalled

		require.NoError(t, exec.Cancel(context.Background(), reqCtx, queue))

		close(release)
		<-done

		// processEvents keeps running its own loop after Cancel writes the
		// canceled Final: nothing in this package stops it, since that job
		// belongs to a2a-go's consumer, which stops reading at the first
		// Final event. A raw queue like this one keeps whatever lands after
		// it too, so only the prefix up to the canceled Final is checked.
		got := queue.snapshot()
		require.GreaterOrEqual(t, len(got), 3)

		_, ok := got[0].(*a2a.TaskArtifactUpdateEvent)
		require.True(t, ok)

		flush, ok := got[1].(*a2a.TaskArtifactUpdateEvent)
		require.True(t, ok)
		assert.Equal(t, "buffered", artifactText(flush))

		canceled, ok := got[2].(*a2a.TaskStatusUpdateEvent)
		require.True(t, ok)
		assert.Equal(t, a2a.TaskStateCanceled, canceled.Status.State)
		assert.True(t, canceled.Final)

		// The deltas sent after Cancel must produce nothing at all: the
		// writer is closed, so delta() is a no-op for both of them.
		artifacts := filterArtifacts(got)
		require.Len(t, artifacts, 2, "ignored-1 and ignored-2 must not appear as artifact events")
	})

	t.Run("no writer registered", func(t *testing.T) {
		t.Parallel()

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
		reqCtx := testReqCtx()
		reqCtx.TaskID = "unregistered-task"

		require.NoError(t, exec.Cancel(context.Background(), reqCtx, queue))

		got := queue.snapshot()
		require.Len(t, got, 1)

		canceled, ok := got[0].(*a2a.TaskStatusUpdateEvent)
		require.True(t, ok)
		assert.Equal(t, a2a.TaskStateCanceled, canceled.Status.State)
	})

	t.Run("different queue than the running task", func(t *testing.T) {
		t.Parallel()

		// a2a-go's distributed (cluster) mode hands Execute and Cancel
		// different queues for the same task (work_queue_handler.go).
		// queueA is what the running task is writing to; queueB is what
		// Cancel is handed - a different pipe, as in that mode.
		queueA := newRecordingQueue()
		queueB := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
		reqCtx := testReqCtx()

		stalled := make(chan struct{})
		release := make(chan struct{})

		seqFn := func(yield func(agent.Event, error) bool) {
			if !yield(deltaEvent("lead"), nil) {
				return
			}

			if !yield(deltaEvent("buffered"), nil) {
				return
			}

			close(stalled)
			<-release
		}

		done := make(chan struct{})

		go func() {
			defer close(done)

			_ = exec.processEvents(context.Background(), reqCtx, queueA, seqFn)
		}()

		<-stalled

		require.NoError(t, exec.Cancel(context.Background(), reqCtx, queueB))

		close(release)
		<-done

		// queueB gets exactly the canceled status, written directly: dw
		// belongs to queueA, a different queue, so Cancel must not touch it.
		gotB := queueB.snapshot()
		require.Len(t, gotB, 1)

		canceled, ok := gotB[0].(*a2a.TaskStatusUpdateEvent)
		require.True(t, ok)
		assert.Equal(t, a2a.TaskStateCanceled, canceled.Status.State)
		assert.True(t, canceled.Final)

		// queueA is untouched by the mismatched Cancel: no canceled status
		// lands there, and the running task's own writer is free to flush
		// "buffered" there on its own terms.
		for _, ev := range queueA.snapshot() {
			if status, ok := ev.(*a2a.TaskStatusUpdateEvent); ok {
				assert.NotEqual(t, a2a.TaskStateCanceled, status.Status.State)
			}
		}
	})
}

// TestDeltaCoalescing_CanceledFlushesWithBackgroundCtx proves that the
// context-canceled branch flushes and writes its status through the
// background context, not the already-canceled loop ctx. A queue that
// honors ctx (like recordingQueue) would silently drop both writes if the
// code used the wrong one, so this would fail if that regressed.
func TestDeltaCoalescing_CanceledFlushesWithBackgroundCtx(t *testing.T) {
	t.Parallel()

	queue := newRecordingQueue()
	exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
	reqCtx := testReqCtx()

	ctx, cancelCtx := context.WithCancel(context.Background())

	seqFn := func(yield func(agent.Event, error) bool) {
		if !yield(deltaEvent("lead"), nil) {
			return
		}

		if !yield(deltaEvent("buffered"), nil) {
			return
		}

		cancelCtx() // the parent ctx is now done, as on a real cancellation

		yield(nil, context.Canceled)
	}

	require.NoError(t, exec.processEvents(ctx, reqCtx, queue, seqFn))

	got := queue.snapshot()
	require.Len(t, got, 3, "leading delta, buffered-tail flush, canceled status - all via the background context")

	_, ok := got[0].(*a2a.TaskArtifactUpdateEvent)
	require.True(t, ok)

	flush, ok := got[1].(*a2a.TaskArtifactUpdateEvent)
	require.True(t, ok, "the flush must use the background context to get through the already-canceled loop ctx")
	assert.Equal(t, "buffered", artifactText(flush))

	status, ok := got[2].(*a2a.TaskStatusUpdateEvent)
	require.True(t, ok)
	assert.Equal(t, a2a.TaskStateCanceled, status.Status.State)
	assert.True(t, status.Final)
}

// TestDeltaCoalescing_MissingInvocationEnd checks the fallback path taken
// when the event stream ends without an InvocationEndEvent: the pending
// text still flushes before the failed status, and processEvents still
// reports its error.
func TestDeltaCoalescing_MissingInvocationEnd(t *testing.T) {
	t.Parallel()

	queue := newRecordingQueue()
	exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: time.Hour})
	reqCtx := testReqCtx()

	err := exec.processEvents(context.Background(), reqCtx, queue, events(
		deltaEvent("lead"),
		deltaEvent("buffered"),
	))
	require.Error(t, err)

	got := queue.snapshot()
	require.Len(t, got, 3)

	_, ok := got[0].(*a2a.TaskArtifactUpdateEvent)
	require.True(t, ok)

	flush, ok := got[1].(*a2a.TaskArtifactUpdateEvent)
	require.True(t, ok)
	assert.Equal(t, "buffered", artifactText(flush))

	status, ok := got[2].(*a2a.TaskStatusUpdateEvent)
	require.True(t, ok)
	assert.Equal(t, a2a.TaskStateFailed, status.Status.State)
	assert.True(t, status.Final)
}

// TestDeltaCoalescing_NoWriteAfterReturn checks that close() stops a still
// -armed flush timer when processEvents returns, so it cannot write
// anything once the writer is closed, even after its Interval passes. The
// queue fails every buffered-tail flush, so InvocationEnd's own flush
// attempt does not itself stop the timer first: without close() actually
// stopping it, the timer would still be armed, and a later fire would be
// observable as another write attempt.
func TestDeltaCoalescing_NoWriteAfterReturn(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		queue := newFlakyQueue(func(ev a2a.Event) bool {
			a, ok := ev.(*a2a.TaskArtifactUpdateEvent)
			return ok && a.Append && !a.LastChunk
		})
		exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: 100 * time.Millisecond})
		reqCtx := testReqCtx()

		require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, events(
			deltaEvent("lead"),
			deltaEvent("buffered"), // arms the flush timer
			invocationEnd(),        // its own flush attempt fails; Final still gets written
		)))

		_, stillActive := exec.activeWriters.Load(reqCtx.TaskID)
		assert.False(t, stillActive, "processEvents must remove its writer from activeWriters on return")

		before := queue.attempts.Load()

		time.Sleep(200 * time.Millisecond) // advance well past Interval
		synctest.Wait()

		after := queue.attempts.Load()
		assert.Equal(t, before, after, "close() must stop the pending timer so no further write is even attempted")

		got := queue.snapshot()
		last := got[len(got)-1]
		status, ok := last.(*a2a.TaskStatusUpdateEvent)
		require.True(t, ok)
		assert.True(t, status.Final)
	})
}

// TestDeltaCoalescing_TimerVsLoop stresses the main loop and the flush
// timer against each other: deltas arrive faster than Interval, so some
// flushes come from delta() noticing its own trigger and some come from
// onTimer on a different goroutine. Every write goes through the same
// mutex, so the result must still be well-ordered and complete.
//
// A manual `go test -race -count=20 ./adapter/a2a/...` run is the receipt
// that this ordering actually needs the mutex; `task test:unit` runs
// without -race.
func TestDeltaCoalescing_TimerVsLoop(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		const (
			n        = 50
			interval = 10 * time.Millisecond
			delay    = 3 * time.Millisecond // faster than interval, so some deltas race the timer
		)

		var want strings.Builder

		steps := make([]seqStep, 0, n+2)

		for i := range n {
			chunk := fmt.Sprintf("[%02d]", i)
			want.WriteString(chunk)
			steps = append(steps, seqStep{sleep: delay, event: deltaEvent(chunk)})
		}

		steps = append(steps, seqStep{event: messageEvent(want.String())})
		steps = append(steps, seqStep{event: invocationEnd()})

		queue := newRecordingQueue()
		exec := newCoalesceTestExecutor(DeltaCoalescing{Interval: interval})
		reqCtx := testReqCtx()

		require.NoError(t, exec.processEvents(context.Background(), reqCtx, queue, seq(steps...)))

		got := queue.snapshot()
		artifacts := filterArtifacts(got)
		require.NotEmpty(t, artifacts)
		assert.Equal(t, want.String(), concatText(artifacts))

		for _, a := range artifacts {
			if len(a.Artifact.Parts) > 0 {
				assert.Len(t, a.Artifact.Parts, 1)
			}
		}

		lastArtifactIdx, firstStatusIdx := -1, -1

		for i, ev := range got {
			switch ev.(type) {
			case *a2a.TaskArtifactUpdateEvent:
				lastArtifactIdx = i
			case *a2a.TaskStatusUpdateEvent:
				if firstStatusIdx == -1 {
					firstStatusIdx = i
				}
			}
		}

		require.NotEqual(t, -1, firstStatusIdx)
		assert.Less(t, lastArtifactIdx, firstStatusIdx, "every artifact event precedes the status events")

		final, ok := got[len(got)-1].(*a2a.TaskStatusUpdateEvent)
		require.True(t, ok)
		assert.True(t, final.Final)
	})
}
