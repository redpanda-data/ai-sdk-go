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
	"log/slog"
	"strings"
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

func testReqCtx() *a2asrv.RequestContext {
	return &a2asrv.RequestContext{ContextID: "test-context", TaskID: a2a.TaskID("test-task")}
}

// recordingQueue records successful writes. It fails write number failAt
// (1-based, 0 means never) and rejects writes on a done ctx.
type recordingQueue struct {
	failAt int
	writes int
	events []a2a.Event
}

func (q *recordingQueue) Write(ctx context.Context, event a2a.Event) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	q.writes++
	if q.writes == q.failAt {
		return errors.New("simulated write failure")
	}

	q.events = append(q.events, event)

	return nil
}

func (q *recordingQueue) WriteVersioned(ctx context.Context, event a2a.Event, _ a2a.TaskVersion) error {
	return q.Write(ctx, event)
}

func (q *recordingQueue) Read(context.Context) (a2a.Event, a2a.TaskVersion, error) {
	return nil, 0, errors.New("recordingQueue: Read is not supported")
}

func (q *recordingQueue) Close() error { return nil }

func deltaEvent(text string) agent.AssistantDeltaEvent {
	return agent.AssistantDeltaEvent{Delta: llm.ContentPartEvent{Part: &llm.TextPart{Text: text}}}
}

func toolCallDelta() agent.AssistantDeltaEvent {
	return agent.AssistantDeltaEvent{Delta: llm.ContentPartEvent{Part: llm.NewToolRequestPart("1", "get_weather", nil)}}
}

func messageEvent(text string) agent.MessageEvent {
	return agent.MessageEvent{Response: llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(text))}}
}

func statusEvent(stage agent.StatusStage) agent.StatusEvent { return agent.StatusEvent{Stage: stage} }

func streamReset() agent.StreamResetEvent { return agent.StreamResetEvent{Attempt: 1, Reason: "retry"} }

func toolResponseEvent() agent.ToolResponseEvent {
	return agent.ToolResponseEvent{Response: llm.ToolResponsePart{ID: "1", Name: "get_weather"}}
}

func invocationEnd() agent.InvocationEndEvent {
	return agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}
}

// probeMark is a step that records how many events the queue holds.
type probeMark struct{}

var probe = probeMark{}

// seqStep is one runner step: a sleep, a probe, an event, or an error.
type seqStep struct {
	sleep time.Duration
	probe bool
	event agent.Event
	err   error
}

// steps turns agent events, errors, durations (sleeps) and probe into steps.
func steps(items ...any) []seqStep {
	out := make([]seqStep, 0, len(items))

	for _, it := range items {
		switch v := it.(type) {
		case time.Duration:
			out = append(out, seqStep{sleep: v})
		case probeMark:
			out = append(out, seqStep{probe: true})
		case error:
			out = append(out, seqStep{err: v})
		case agent.Event:
			out = append(out, seqStep{event: v})
		}
	}

	return out
}

type coalesceCase struct {
	name       string
	cfg        DeltaCoalescing
	steps      []seqStep
	failWrite  int // 1-based queue write to fail
	wantErr    bool
	want       []wantEvent
	wantProbes []int
}

// run drives processEvents in a synctest bubble, so sleeps move a fake
// clock. A context.Canceled step cancels the parent ctx first, as a real
// cancellation does.
func (tc coalesceCase) run(t *testing.T) {
	t.Helper()

	synctest.Test(t, func(t *testing.T) {
		queue := &recordingQueue{failAt: tc.failWrite}
		exec := &Executor{log: slog.Default(), coalesce: tc.cfg}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		var probes []int

		seq := func(yield func(agent.Event, error) bool) {
			for _, s := range tc.steps {
				switch {
				case s.sleep > 0:
					time.Sleep(s.sleep)
				case s.probe:
					probes = append(probes, len(queue.events))
				default:
					if errors.Is(s.err, context.Canceled) {
						cancel()
					}

					if !yield(s.event, s.err) {
						return
					}
				}
			}
		}

		err := exec.processEvents(ctx, testReqCtx(), queue, seq)
		if tc.wantErr {
			require.Error(t, err)
		} else {
			require.NoError(t, err)
		}

		assertEvents(t, queue.events, tc.want)
		assert.Equal(t, tc.wantProbes, probes)
	})
}

// wantEvent is one expected queued event: an artifact or a status.
type wantEvent struct {
	text      string
	append    bool
	lastChunk bool

	isStatus bool
	state    a2a.TaskState
	final    bool
}

func wantArtifact(text string, isAppend, lastChunk bool) wantEvent {
	return wantEvent{text: text, append: isAppend, lastChunk: lastChunk}
}

func wantStatus(state a2a.TaskState, final bool) wantEvent {
	return wantEvent{isStatus: true, state: state, final: final}
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

// assertEvents checks got against want. Each artifact has a non-empty ID and
// at most one part. An append targets the previous artifact, which a recorded
// create made. A create uses a new ID.
func assertEvents(t *testing.T, got []a2a.Event, want []wantEvent) {
	t.Helper()

	require.Len(t, got, len(want))

	var lastID a2a.ArtifactID

	created := map[a2a.ArtifactID]bool{}

	for i, w := range want {
		if w.isStatus {
			s, ok := got[i].(*a2a.TaskStatusUpdateEvent)
			require.True(t, ok, "event %d", i)
			assert.Equal(t, w.state, s.Status.State, "event %d", i)
			assert.Equal(t, w.final, s.Final, "event %d", i)

			continue
		}

		a, ok := got[i].(*a2a.TaskArtifactUpdateEvent)
		require.True(t, ok, "event %d", i)
		require.NotEmpty(t, a.Artifact.ID, "event %d", i)
		assert.Equal(t, w.append, a.Append, "event %d", i)
		assert.Equal(t, w.lastChunk, a.LastChunk, "event %d", i)
		assert.Equal(t, w.text, artifactText(a), "event %d", i)
		assert.LessOrEqual(t, len(a.Artifact.Parts), 1, "event %d", i)

		if a.Append {
			assert.True(t, created[a.Artifact.ID], "event %d appends to an artifact that was never created", i)
			assert.Equal(t, lastID, a.Artifact.ID, "event %d", i)
		} else {
			assert.False(t, created[a.Artifact.ID], "event %d", i)
			created[a.Artifact.ID] = true
		}

		lastID = a.Artifact.ID
	}
}

var (
	working   = wantStatus(a2a.TaskStateWorking, false)
	completed = wantStatus(a2a.TaskStateCompleted, true)
	failed    = wantStatus(a2a.TaskStateFailed, true)
	canceled  = wantStatus(a2a.TaskStateCanceled, true)
)

// TestDeltaCoalescing_DisabledIsByteIdentical pins the Interval == 0 event
// sequence to the one processEvents sent before coalescing existed.
func TestDeltaCoalescing_DisabledIsByteIdentical(t *testing.T) {
	t.Parallel()

	a := wantArtifact("a", false, false)
	cases := []coalesceCase{
		{
			name:  "deltas and message",
			steps: steps(deltaEvent("a"), deltaEvent("b"), deltaEvent("c"), messageEvent("abc"), invocationEnd()),
			want: []wantEvent{
				a, wantArtifact("b", true, false), wantArtifact("c", true, false),
				wantArtifact("", true, true), working, completed,
			},
		},
		{
			name:  "stream reset",
			steps: steps(deltaEvent("a"), deltaEvent("b"), streamReset(), deltaEvent("c"), invocationEnd()),
			want: []wantEvent{
				a, wantArtifact("b", true, false), wantArtifact("", true, true),
				wantArtifact("c", false, false), completed,
			},
		},
		{
			name:  "model_call writes nothing",
			steps: steps(deltaEvent("a"), statusEvent(agent.StatusStageModelCall), deltaEvent("b"), invocationEnd()),
			want:  []wantEvent{a, wantArtifact("b", false, false), completed},
		},
		{
			name:  "tool response",
			steps: steps(deltaEvent("a"), toolResponseEvent(), invocationEnd()),
			want:  []wantEvent{a, working, completed},
		},
		{name: "generic failure", steps: steps(deltaEvent("a"), errors.New("boom")), want: []wantEvent{a, failed}},
		{name: "context overflow", steps: steps(deltaEvent("a"), llm.ErrContextOverflow), want: []wantEvent{a, failed}},
		{name: "canceled", steps: steps(deltaEvent("a"), context.Canceled), want: []wantEvent{a, canceled}},
		{name: "missing invocation end", steps: steps(deltaEvent("a")), wantErr: true, want: []wantEvent{a, failed}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			tc.run(t)
		})
	}
}

func TestDeltaCoalescing(t *testing.T) {
	t.Parallel()

	hour := DeltaCoalescing{Interval: time.Hour}
	ms100 := DeltaCoalescing{Interval: 100 * time.Millisecond}
	lead := wantArtifact("lead", false, false)
	tail := wantArtifact("tail", true, false)

	cases := []coalesceCase{
		{
			name:  "message event carries the tail into LastChunk",
			cfg:   hour,
			steps: steps(deltaEvent("lead"), deltaEvent("tail"), messageEvent("leadtail"), invocationEnd()),
			want:  []wantEvent{lead, wantArtifact("tail", true, true), working, completed},
		},
		{
			name:  "stream reset flushes the tail, then restarts",
			cfg:   hour,
			steps: steps(deltaEvent("lead"), deltaEvent("tail"), streamReset(), deltaEvent("new"), invocationEnd()),
			want:  []wantEvent{lead, wantArtifact("tail", true, true), wantArtifact("new", false, false), completed},
		},
		{
			name:  "model_call flushes the old artifact, non-final",
			cfg:   hour,
			steps: steps(deltaEvent("lead"), deltaEvent("tail"), statusEvent(agent.StatusStageModelCall), deltaEvent("new"), invocationEnd()),
			want:  []wantEvent{lead, tail, wantArtifact("new", false, false), completed},
		},
		{
			name:  "tool response flushes before the history status",
			cfg:   hour,
			steps: steps(deltaEvent("lead"), deltaEvent("tail"), toolResponseEvent(), invocationEnd()),
			want:  []wantEvent{lead, tail, working, completed},
		},
		{name: "generic failure flushes first", cfg: hour, steps: steps(deltaEvent("lead"), deltaEvent("tail"), errors.New("boom")), want: []wantEvent{lead, tail, failed}},
		{name: "context overflow flushes first", cfg: hour, steps: steps(deltaEvent("lead"), deltaEvent("tail"), llm.ErrContextOverflow), want: []wantEvent{lead, tail, failed}},
		// The parent ctx is done, so these writes pass only on the background ctx.
		{name: "canceled flushes with the background ctx", cfg: hour, steps: steps(deltaEvent("lead"), deltaEvent("tail"), context.Canceled), want: []wantEvent{lead, tail, canceled}},
		{name: "missing invocation end flushes first", cfg: hour, steps: steps(deltaEvent("lead"), deltaEvent("tail")), wantErr: true, want: []wantEvent{lead, tail, failed}},
		{
			name: "size trigger",
			cfg:  DeltaCoalescing{Interval: time.Hour, MaxBytes: 16},
			steps: steps(deltaEvent("aaaaa"), deltaEvent("bbbbb"), deltaEvent("ccccc"), deltaEvent("ddddd"),
				deltaEvent("eeeee"), messageEvent("x"), invocationEnd()),
			want: []wantEvent{
				wantArtifact("aaaaa", false, false), wantArtifact("bbbbbcccccdddddeeeee", true, false),
				wantArtifact("", true, true), working, completed,
			},
		},
		{
			name:       "first delta is sent before the next event",
			cfg:        DeltaCoalescing{Interval: time.Hour, MaxBytes: 512},
			steps:      steps(deltaEvent("lead"), probe, invocationEnd()),
			want:       []wantEvent{lead, completed},
			wantProbes: []int{1},
		},
		{
			// Deltas at 0, 40, 80, 120, 160 and 200ms: the one at 120ms fires the age trigger.
			name: "age trigger on a text delta",
			cfg:  ms100,
			steps: steps(deltaEvent("a"), 40*time.Millisecond, deltaEvent("b"), 40*time.Millisecond, deltaEvent("c"),
				40*time.Millisecond, deltaEvent("d"), 40*time.Millisecond, deltaEvent("e"), 40*time.Millisecond,
				deltaEvent("f"), messageEvent("abcdef"), invocationEnd()),
			want: []wantEvent{
				wantArtifact("a", false, false), wantArtifact("bcd", true, false), wantArtifact("ef", true, true),
				working, completed,
			},
		},
		{
			name:       "age trigger on a tool-call delta",
			cfg:        ms100,
			steps:      steps(deltaEvent("lead"), deltaEvent("tail"), 200*time.Millisecond, toolCallDelta(), probe, invocationEnd()),
			want:       []wantEvent{lead, tail, completed},
			wantProbes: []int{2},
		},
		{
			name: "age trigger on a status event that writes nothing",
			cfg:  ms100,
			steps: steps(deltaEvent("lead"), deltaEvent("tail"), 200*time.Millisecond,
				statusEvent(agent.StatusStageToolExec), probe, invocationEnd()),
			want:       []wantEvent{lead, tail, completed},
			wantProbes: []int{2},
		},
		{
			name:       "text waits while no event arrives",
			cfg:        ms100,
			steps:      steps(deltaEvent("lead"), deltaEvent("tail"), 500*time.Millisecond, probe, invocationEnd()),
			want:       []wantEvent{lead, tail, completed},
			wantProbes: []int{1},
		},
		{
			// Write 2 is the LastChunk "tail". The next response must start a new artifact.
			name: "failed LastChunk, then model_call and more deltas",
			cfg:  hour,
			steps: steps(deltaEvent("lead"), deltaEvent("tail"), messageEvent("leadtail"),
				statusEvent(agent.StatusStageModelCall), deltaEvent("next"), deltaEvent("more"),
				messageEvent("nextmore"), invocationEnd()),
			failWrite: 2,
			want: []wantEvent{
				lead, working, wantArtifact("next", false, false), wantArtifact("more", true, true), working, completed,
			},
		},
		{
			name:      "failed first write, next delta creates a new artifact",
			cfg:       hour,
			steps:     steps(deltaEvent("lost"), deltaEvent("lead"), deltaEvent("tail"), messageEvent("x"), invocationEnd()),
			failWrite: 1,
			want:      []wantEvent{lead, wantArtifact("tail", true, true), working, completed},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			tc.run(t)
		})
	}
}

// discardQueue drops every event, so the benchmark measures the writer only.
type discardQueue struct{}

func (discardQueue) Write(context.Context, a2a.Event) error { return nil }

func (discardQueue) WriteVersioned(context.Context, a2a.Event, a2a.TaskVersion) error { return nil }

func (discardQueue) Read(context.Context) (a2a.Event, a2a.TaskVersion, error) {
	return nil, 0, errors.New("discardQueue: Read is not supported")
}

func (discardQueue) Close() error { return nil }

func BenchmarkDeltaCoalescing(b *testing.B) {
	reqCtx := testReqCtx()
	ctx := context.Background()

	cases := []struct {
		name string
		cfg  DeltaCoalescing
	}{
		{name: "off", cfg: DeltaCoalescing{}},
		{name: "on", cfg: DeltaCoalescing{Interval: time.Hour, MaxBytes: 256}},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()

			dw := newDeltaWriter(reqCtx, discardQueue{}, slog.Default(), tc.cfg)

			for range b.N {
				dw.delta(ctx, "the quick brown fox jumps over the lazy dog")
			}
		})
	}
}
