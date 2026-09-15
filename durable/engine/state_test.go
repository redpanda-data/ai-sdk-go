// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package engine

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/durable"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

var t0 = time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)

// drive applies a command through transition+apply the way the engine does,
// stamping seq/at, and returns the records. It mirrors Engine.commit without I/O.
func drive(t *testing.T, r *run, cmd durable.Command, now time.Time) (*run, []durable.Record) {
	t.Helper()

	recs, err := transition(r, cmd, now, nil)
	require.NoError(t, err)

	if r == nil {
		r = newRun(0)
	}

	for i := range recs {
		recs[i].Seq = r.st.JournalLen + 1
		recs[i].RunID = cmd.RunID
		recs[i].At = now
		recs[i].CommandID = cmd.CommandID
		r.apply(recs[i])
	}

	return r, recs
}

func dispatch(t *testing.T, r *run, now time.Time) string {
	t.Helper()

	require.True(t, r.needsDispatch(), "run should need dispatch, status=%s token=%q", r.st.Status, r.st.CurrentTaskToken)

	attempt := r.st.Attempt + 1
	token := durable.TaskToken(r.st.RunID, attempt)
	lease := now.Add(time.Minute)
	r.apply(durable.Record{Seq: r.st.JournalLen + 1, Type: durable.RecordTaskDispatched, RunID: r.st.RunID, At: now, TaskToken: token, Attempt: attempt, LeaseExpiresAt: &lease})

	return token
}

func userMsg(text string) *llm.Message {
	m := llm.NewMessage(llm.RoleUser, llm.NewTextPart(text))

	return &m
}

func startCmd(id string) durable.Command {
	cmd := durable.NewCommand(durable.CommandStartRun, id)
	cmd.Agent = "support"
	cmd.TaskQueue = "q"
	cmd.Message = userMsg("hello")

	return cmd
}

func TestStartRun_IsIdempotentWhileOpen(t *testing.T) {
	t.Parallel()

	r, recs := drive(t, nil, startCmd("run-1"), t0)
	require.Len(t, recs, 2)
	assert.Equal(t, durable.RecordRunStarted, recs[0].Type)
	assert.Equal(t, durable.RecordMessageAppended, recs[1].Type)
	assert.Equal(t, durable.StatusRunning, r.st.Status)
	assert.Equal(t, "v1", r.st.Version, "default version")
	assert.Len(t, r.st.Messages, 1)
	assert.True(t, r.needsDispatch())

	_, err := transition(r, startCmd("run-1"), t0, nil)
	require.ErrorIs(t, err, errRunOpen)
}

func TestStartRun_RolloutPicksTarget(t *testing.T) {
	t.Parallel()

	ro := &durable.Rollout{Agent: "support", Targets: []durable.RolloutTarget{
		{Version: "v1", TaskQueue: "q-v1", Weight: 50},
		{Version: "v2", TaskQueue: "q-v2", Weight: 50},
	}}

	seen := map[string]int{}

	for i := range 200 {
		cmd := startCmd("run-" + string(rune('a'+i%26)) + string(rune('0'+i/26)))
		cmd.TaskQueue = ""

		recs, err := transition(nil, cmd, t0, ro)
		require.NoError(t, err)

		seen[recs[0].TaskQueue]++

		again, err := transition(nil, cmd, t0, ro)
		require.NoError(t, err)
		assert.Equal(t, recs[0].TaskQueue, again[0].TaskQueue, "selection must be deterministic per run id")
	}

	assert.Positive(t, seen["q-v1"])
	assert.Positive(t, seen["q-v2"])
}

func TestWorkerCommands_RejectStaleToken(t *testing.T) {
	t.Parallel()

	r, _ := drive(t, nil, startCmd("run-2"), t0)
	token := dispatch(t, r, t0)

	cmd := durable.NewCommand(durable.CommandRunCompleted, "run-2")
	cmd.TaskToken = "run-2:99"

	_, err := transition(r, cmd, t0, nil)
	require.ErrorIs(t, err, errStaleToken)

	cmd.TaskToken = token
	r, recs := drive(t, r, cmd, t0.Add(time.Second))
	require.Len(t, recs, 1)
	assert.Equal(t, durable.StatusCompleted, r.st.Status)
	assert.Empty(t, r.st.CurrentTaskToken)
	assert.True(t, r.st.Closed())
}

func TestAppendMessage_OrderingAndDuplicates(t *testing.T) {
	t.Parallel()

	r, _ := drive(t, nil, startCmd("run-3"), t0)
	token := dispatch(t, r, t0)

	assistant := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("hi"))
	cmd := durable.NewCommand(durable.CommandAppendMessage, "run-3")
	cmd.TaskToken = token
	cmd.MessageIndex = 1
	cmd.Message = &assistant

	r, recs := drive(t, r, cmd, t0)
	require.Len(t, recs, 1)
	assert.Len(t, r.st.Messages, 2)

	// Retried send of the same index is a silent no-op.
	dup := cmd
	dup.CommandID = "other"
	recs, err := transition(r, dup, t0, nil)
	require.NoError(t, err)
	assert.Empty(t, recs)

	// A gap is an error.
	gap := cmd
	gap.MessageIndex = 5
	_, err = transition(r, gap, t0, nil)
	require.ErrorIs(t, err, errMessageIndex)
}

func TestSuspendAndInput_DirectAndBuffered(t *testing.T) {
	t.Parallel()

	// Direct: run waits, then input arrives.
	r, _ := drive(t, nil, startCmd("run-4"), t0)
	token := dispatch(t, r, t0)

	suspend := durable.NewCommand(durable.CommandRunSuspended, "run-4")
	suspend.TaskToken = token
	suspend.Awaiting = &durable.Awaiting{Kind: durable.AwaitInput, ToolCallID: "call_1", ToolName: "approve", Name: "approval"}

	r, recs := drive(t, r, suspend, t0)
	require.Len(t, recs, 1)
	assert.Equal(t, durable.StatusSuspended, r.st.Status)
	assert.False(t, r.needsDispatch())

	wrongName := durable.NewCommand(durable.CommandSendInput, "run-4")
	wrongName.Name = "other"
	wrongName.Payload = json.RawMessage(`1`)
	r, recs = drive(t, r, wrongName, t0)
	assert.Equal(t, durable.RecordInputReceived, recs[0].Type, "non-matching input is buffered")
	assert.Equal(t, durable.StatusSuspended, r.st.Status)

	input := durable.NewCommand(durable.CommandSendInput, "run-4")
	input.Name = "approval"
	input.Payload = json.RawMessage(`{"approved":true}`)
	r, recs = drive(t, r, input, t0)
	assert.Equal(t, durable.RecordInputDelivered, recs[0].Type)
	assert.Equal(t, durable.StatusRunning, r.st.Status)
	assert.JSONEq(t, `{"approved":true}`, string(r.st.Delivered["call_1"]))
	assert.True(t, r.needsDispatch())
	assert.Len(t, r.st.PendingInputs, 1, "unrelated buffered input stays")

	// Buffered: input arrives before the run waits for it.
	r2, _ := drive(t, nil, startCmd("run-5"), t0)
	token2 := dispatch(t, r2, t0)

	early := durable.NewCommand(durable.CommandSendInput, "run-5")
	early.Name = "approval"
	early.Payload = json.RawMessage(`"yes"`)
	r2, recs = drive(t, r2, early, t0)
	assert.Equal(t, durable.RecordInputReceived, recs[0].Type)

	suspend2 := durable.NewCommand(durable.CommandRunSuspended, "run-5")
	suspend2.TaskToken = token2
	suspend2.Awaiting = &durable.Awaiting{Kind: durable.AwaitInput, ToolCallID: "call_7", Name: "approval"}
	r2, recs = drive(t, r2, suspend2, t0)
	require.Len(t, recs, 2)
	assert.Equal(t, durable.RecordInputDelivered, recs[1].Type)
	assert.Equal(t, durable.StatusRunning, r2.st.Status)
	assert.Empty(t, r2.st.PendingInputs)
	assert.JSONEq(t, `"yes"`, string(r2.st.Delivered["call_7"]))
}

func TestToolResult_PrunedWhenMessageLands(t *testing.T) {
	t.Parallel()

	r, _ := drive(t, nil, startCmd("run-6"), t0)
	token := dispatch(t, r, t0)

	rec := durable.NewCommand(durable.CommandRecordToolResult, "run-6")
	rec.TaskToken = token
	rec.ToolResult = &llm.ToolResponsePart{ID: "call_1", Name: "lookup", Result: json.RawMessage(`{"ok":true}`)}
	r, _ = drive(t, r, rec, t0)
	require.Contains(t, r.st.ToolResults, "call_1")

	toolMsg := llm.NewMessage(llm.RoleUser, rec.ToolResult)
	app := durable.NewCommand(durable.CommandAppendMessage, "run-6")
	app.TaskToken = token
	app.MessageIndex = 1
	app.Message = &toolMsg
	r, _ = drive(t, r, app, t0)
	assert.Nil(t, r.st.ToolResults, "journaled result folded into the session is dropped from the task payload")
}

func TestAttemptFailed_RetriesThenExhausts(t *testing.T) {
	t.Parallel()

	cmd := startCmd("run-7")
	cmd.RetryPolicy = &durable.RetryPolicy{InitialInterval: time.Second, BackoffCoefficient: 2, MaxInterval: time.Minute, MaxAttempts: 2}

	r, _ := drive(t, nil, cmd, t0)
	token := dispatch(t, r, t0)

	fail := durable.NewCommand(durable.CommandAttemptFailed, "run-7")
	fail.TaskToken = token
	fail.Error = "model 503"

	r, recs := drive(t, r, fail, t0)
	require.Len(t, recs, 1)
	assert.Equal(t, durable.StatusRetrying, r.st.Status)
	assert.False(t, recs[0].Exhausted)
	require.NotNil(t, r.st.RetryAt)
	assert.Equal(t, t0.Add(time.Second), *r.st.RetryAt)
	assert.False(t, r.needsDispatch(), "not before the backoff elapses")

	// Retry timer: engine flips to running and dispatches attempt 2.
	r.st.Status = durable.StatusRunning
	r.st.RetryAt = nil
	token = dispatch(t, r, t0.Add(time.Second))
	assert.Equal(t, 2, r.st.Attempt)

	fail.CommandID = "second"
	fail.TaskToken = token
	r, recs = drive(t, r, fail, t0.Add(2*time.Second))
	assert.True(t, recs[0].Exhausted)
	assert.Equal(t, durable.StatusFailed, r.st.Status)
	assert.Equal(t, "model 503", r.st.Error)
	assert.True(t, r.st.Closed())

	// Non-retryable failures exhaust immediately.
	r2, _ := drive(t, nil, startCmd("run-8"), t0)
	token2 := dispatch(t, r2, t0)
	nr := durable.NewCommand(durable.CommandAttemptFailed, "run-8")
	nr.TaskToken = token2
	nr.Error = "unsupported agent version"
	f := false
	nr.Retryable = &f
	r2, _ = drive(t, r2, nr, t0)
	assert.Equal(t, durable.StatusFailed, r2.st.Status)
}

func TestContinueRun_ReopensClosedRun(t *testing.T) {
	t.Parallel()

	r, _ := drive(t, nil, startCmd("run-9"), t0)
	token := dispatch(t, r, t0)

	cont := durable.NewCommand(durable.CommandContinueRun, "run-9")
	cont.Message = userMsg("more")
	_, err := transition(r, cont, t0, nil)
	require.ErrorIs(t, err, errRunOpen)

	done := durable.NewCommand(durable.CommandRunCompleted, "run-9")
	done.TaskToken = token
	done.FinishReason = "stop"
	r, _ = drive(t, r, done, t0)

	r, recs := drive(t, r, cont, t0.Add(time.Hour))
	require.Len(t, recs, 1)
	assert.Equal(t, durable.RecordRunContinued, recs[0].Type)
	assert.Equal(t, durable.StatusRunning, r.st.Status)
	assert.Len(t, r.st.Messages, 2)
	assert.True(t, r.needsDispatch())
	assert.Nil(t, r.st.ClosedAt)
}

func TestReplay_ReproducesState(t *testing.T) {
	t.Parallel()

	// Build a run live, capturing every record, then fold the records into a
	// fresh run and compare. This is the engine's restart guarantee.
	journal := make([]durable.Record, 0, 8)

	r, recs := drive(t, nil, startCmd("run-10"), t0)
	journal = append(journal, recs...)

	token := dispatch(t, r, t0)
	journal = append(journal, durable.Record{Seq: r.st.JournalLen, Type: durable.RecordTaskDispatched, RunID: "run-10", At: t0, TaskToken: token, Attempt: 1, LeaseExpiresAt: r.st.LeaseExpiresAt})

	suspend := durable.NewCommand(durable.CommandRunSuspended, "run-10")
	suspend.TaskToken = token
	fire := t0.Add(time.Hour)
	suspend.Awaiting = &durable.Awaiting{Kind: durable.AwaitTimer, ToolCallID: "call_3", FireAt: &fire}
	r, recs = drive(t, r, suspend, t0)
	journal = append(journal, recs...)

	replayed := newRun(0)
	for _, rec := range journal {
		replayed.apply(rec)
	}

	assert.Equal(t, r.st, replayed.st)
	assert.True(t, replayed.hasSeen(suspend.CommandID), "dedup window survives replay")
}

func TestRetryPolicy_Backoff(t *testing.T) {
	t.Parallel()

	p := durable.RetryPolicy{InitialInterval: time.Second, BackoffCoefficient: 2, MaxInterval: 10 * time.Second, MaxAttempts: 4}
	assert.Equal(t, time.Second, p.Backoff(1))
	assert.Equal(t, 2*time.Second, p.Backoff(2))
	assert.Equal(t, 4*time.Second, p.Backoff(3))
	assert.Equal(t, 10*time.Second, p.Backoff(10))
	assert.False(t, p.Exhausted(3))
	assert.True(t, p.Exhausted(4))
	assert.False(t, durable.RetryPolicy{}.Exhausted(1000), "zero max attempts is unlimited")
}
