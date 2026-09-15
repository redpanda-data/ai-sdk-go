// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

// Package engine is the durable execution controller. It folds the journal
// topic into per-run state, dispatches tasks to workers, enforces leases,
// retries failed attempts, fires timers, matches external input to waiting
// tool calls and publishes results. See durable's package documentation.
package engine

import (
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"time"

	"github.com/redpanda-data/ai-sdk-go/durable"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// maxSeenCommands bounds the per-run dedup window.
const maxSeenCommands = 512

// run is the engine's in-memory record of one run.
type run struct {
	st        durable.RunState
	partition int32
	seen      map[string]struct{}
	seenOrder []string
}

func newRun(partition int32) *run {
	return &run{partition: partition, seen: map[string]struct{}{}}
}

func (r *run) hasSeen(id string) bool {
	_, ok := r.seen[id]

	return ok
}

func (r *run) markSeen(id string) {
	if id == "" || r.hasSeen(id) {
		return
	}

	r.seen[id] = struct{}{}
	r.seenOrder = append(r.seenOrder, id)

	if len(r.seenOrder) > maxSeenCommands {
		delete(r.seen, r.seenOrder[0])
		r.seenOrder = r.seenOrder[1:]
	}
}

// apply folds one journal record into the state. It is the only place state
// changes, both live and during replay.
func (r *run) apply(rec durable.Record) {
	st := &r.st
	st.JournalLen = rec.Seq
	st.UpdatedAt = rec.At
	r.markSeen(rec.CommandID)

	switch rec.Type {
	case durable.RecordRunStarted:
		*st = durable.RunState{
			RunID:            rec.RunID,
			Agent:            rec.Agent,
			Version:          rec.Version,
			TaskQueue:        rec.TaskQueue,
			Status:           durable.StatusRunning,
			Metadata:         rec.Metadata,
			RetryPolicy:      derefPolicy(rec.RetryPolicy),
			ExecutionTimeout: rec.ExecutionTimeout,
			ResultTopic:      rec.ResultTopic,
			Messages:         []llm.Message{},
			JournalLen:       rec.Seq,
			StartedAt:        rec.At,
			UpdatedAt:        rec.At,
		}

	case durable.RecordRunContinued:
		st.Status = durable.StatusRunning
		st.CurrentTaskToken = ""
		st.Awaiting = nil
		st.RetryAt = nil
		st.LeaseExpiresAt = nil
		st.FinishReason = ""
		st.Error = ""
		st.ClosedAt = nil
		st.Failures = 0

		if rec.Message != nil {
			st.Messages = append(st.Messages, *rec.Message)
		}

	case durable.RecordMessageAppended:
		if rec.Message != nil && rec.MessageIndex == len(st.Messages) {
			st.Messages = append(st.Messages, *rec.Message)
			r.pruneToolState(*rec.Message)
		}

	case durable.RecordMessagesReset:
		st.Messages = rec.Messages
		if st.Messages == nil {
			st.Messages = []llm.Message{}
		}

	case durable.RecordTaskDispatched:
		st.Status = durable.StatusRunning
		st.CurrentTaskToken = rec.TaskToken
		st.Attempt = rec.Attempt
		st.LeaseExpiresAt = rec.LeaseExpiresAt
		st.RetryAt = nil
		st.Awaiting = nil

	case durable.RecordLeaseExpired:
		st.CurrentTaskToken = ""
		st.LeaseExpiresAt = nil

	case durable.RecordToolResultRecorded:
		if rec.ToolResult != nil {
			if st.ToolResults == nil {
				st.ToolResults = map[string]*llm.ToolResponsePart{}
			}

			st.ToolResults[rec.ToolResult.ID] = rec.ToolResult
		}

	case durable.RecordRunSuspended:
		st.Status = durable.StatusSuspended
		st.Awaiting = rec.Awaiting
		st.CurrentTaskToken = ""
		st.LeaseExpiresAt = nil

	case durable.RecordInputReceived:
		st.PendingInputs = append(st.PendingInputs, durable.PendingInput{Name: rec.Name, Payload: rec.Payload})

	case durable.RecordInputDelivered:
		r.deliver(rec.ToolCallID, rec.Payload)
		st.PendingInputs = popPending(st.PendingInputs, rec.Name)

	case durable.RecordTimerFired:
		r.deliver(rec.ToolCallID, rec.Payload)

	case durable.RecordAttemptFailed:
		st.Failures = rec.Failures
		st.CurrentTaskToken = ""
		st.LeaseExpiresAt = nil

		if rec.Exhausted {
			r.close(durable.StatusFailed, rec)
			st.Error = rec.Error
		} else {
			st.Status = durable.StatusRetrying
			st.RetryAt = rec.RetryAt
			st.Error = rec.Error
		}

	case durable.RecordRunCompleted:
		r.close(durable.StatusCompleted, rec)
		st.FinishReason = rec.FinishReason
		st.Usage = rec.Usage

	case durable.RecordRunFailed:
		r.close(durable.StatusFailed, rec)
		st.Error = rec.Error

	case durable.RecordRunCancelled:
		r.close(durable.StatusCancelled, rec)
		st.Error = rec.Reason
	}
}

func (r *run) close(status string, rec durable.Record) {
	st := &r.st
	st.Status = status
	st.CurrentTaskToken = ""
	st.LeaseExpiresAt = nil
	st.Awaiting = nil
	st.RetryAt = nil
	at := rec.At
	st.ClosedAt = &at
}

func (r *run) deliver(toolCallID string, payload json.RawMessage) {
	st := &r.st
	if st.Delivered == nil {
		st.Delivered = map[string]json.RawMessage{}
	}

	st.Delivered[toolCallID] = payload
	st.Awaiting = nil
	st.Status = durable.StatusRunning
	st.CurrentTaskToken = ""
	st.LeaseExpiresAt = nil
}

// pruneToolState drops journaled tool results and delivered inputs once the
// tool message that carries them is part of the session.
func (r *run) pruneToolState(msg llm.Message) {
	for _, resp := range msg.ToolResponses() {
		delete(r.st.ToolResults, resp.ID)
		delete(r.st.Delivered, resp.ID)
	}

	if len(r.st.ToolResults) == 0 {
		r.st.ToolResults = nil
	}

	if len(r.st.Delivered) == 0 {
		r.st.Delivered = nil
	}
}

func popPending(in []durable.PendingInput, name string) []durable.PendingInput {
	for i, p := range in {
		if p.Name == name {
			out := make([]durable.PendingInput, 0, len(in)-1)
			out = append(out, in[:i]...)

			return append(out, in[i+1:]...)
		}
	}

	return in
}

func findPending(in []durable.PendingInput, name string) (durable.PendingInput, bool) {
	for _, p := range in {
		if p.Name == name {
			return p, true
		}
	}

	return durable.PendingInput{}, false
}

func derefPolicy(p *durable.RetryPolicy) durable.RetryPolicy {
	if p == nil {
		return durable.DefaultRetryPolicy()
	}

	return *p
}

// needsDispatch reports whether the run should be handed to a worker now.
func (r *run) needsDispatch() bool {
	return r.st.Status == durable.StatusRunning && r.st.CurrentTaskToken == ""
}

// transition errors surfaced to the log; none are fatal.
var (
	errStaleToken   = errors.New("stale task token")
	errRunClosed    = errors.New("run is closed")
	errRunOpen      = errors.New("run is open")
	errUnknownRun   = errors.New("unknown run")
	errBadCommand   = errors.New("malformed command")
	errNoTaskQueue  = errors.New("no task queue: set StartOptions.TaskQueue or a rollout for the agent")
	errMessageIndex = errors.New("message index out of order")
)

// transition computes the journal records a command produces against the
// current state. It is pure: no I/O, no mutation. r is nil for unknown runs.
func transition(r *run, cmd durable.Command, now time.Time, rollout *durable.Rollout) ([]durable.Record, error) {
	switch cmd.Type {
	case durable.CommandStartRun:
		return startRun(r, cmd, rollout)
	case durable.CommandContinueRun:
		return continueRun(r, cmd)
	case durable.CommandSendInput:
		return sendInput(r, cmd)
	case durable.CommandCancelRun:
		if r == nil {
			return nil, errUnknownRun
		}

		if r.st.Closed() {
			return nil, errRunClosed
		}

		return []durable.Record{{Type: durable.RecordRunCancelled, Reason: cmd.Reason}}, nil
	}

	// Worker commands.
	if r == nil {
		return nil, errUnknownRun
	}

	if cmd.TaskToken == "" || cmd.TaskToken != r.st.CurrentTaskToken {
		return nil, errStaleToken
	}

	switch cmd.Type {
	case durable.CommandAppendMessage:
		return appendMessage(r, cmd)
	case durable.CommandResetMessages:
		return []durable.Record{{Type: durable.RecordMessagesReset, Messages: cmd.Messages}}, nil
	case durable.CommandRecordToolResult:
		if cmd.ToolResult == nil {
			return nil, errBadCommand
		}

		return []durable.Record{{Type: durable.RecordToolResultRecorded, ToolResult: cmd.ToolResult}}, nil
	case durable.CommandRunSuspended:
		return suspendRun(r, cmd)
	case durable.CommandAttemptFailed:
		return attemptFailed(r, cmd, now)
	case durable.CommandRunCompleted:
		return []durable.Record{{Type: durable.RecordRunCompleted, FinishReason: cmd.FinishReason, Usage: cmd.Usage}}, nil
	default:
		return nil, fmt.Errorf("%w: type %q", errBadCommand, cmd.Type)
	}
}

func startRun(r *run, cmd durable.Command, rollout *durable.Rollout) ([]durable.Record, error) {
	if r != nil && !r.st.Closed() {
		return nil, errRunOpen
	}

	if cmd.Agent == "" {
		return nil, fmt.Errorf("%w: start_run requires agent", errBadCommand)
	}

	version, queue := resolveTarget(cmd, rollout)
	if queue == "" {
		return nil, errNoTaskQueue
	}

	recs := []durable.Record{{
		Type:             durable.RecordRunStarted,
		Agent:            cmd.Agent,
		Version:          version,
		TaskQueue:        queue,
		Metadata:         cmd.Metadata,
		RetryPolicy:      cmd.RetryPolicy,
		ExecutionTimeout: cmd.ExecutionTimeout,
		ResultTopic:      cmd.ResultTopic,
	}}

	if cmd.Message != nil {
		recs = append(recs, durable.Record{Type: durable.RecordMessageAppended, MessageIndex: 0, Message: cmd.Message})
	}

	return recs, nil
}

// resolveTarget pins the (version, task queue) a new run executes on: explicit
// values win, then the agent's rollout record, then defaults.
func resolveTarget(cmd durable.Command, rollout *durable.Rollout) (string, string) {
	version, queue := cmd.Version, cmd.TaskQueue

	var target *durable.RolloutTarget
	if rollout != nil && (version == "" || queue == "") {
		target = pickRollout(cmd.RunID, rollout.Targets)
	}

	if target != nil {
		if version == "" {
			version = target.Version
		}

		if queue == "" {
			queue = target.TaskQueue
		}
	}

	if version == "" {
		version = durable.DefaultVersion
	}

	if queue == "" {
		queue = cmd.Agent
	}

	return version, queue
}

func continueRun(r *run, cmd durable.Command) ([]durable.Record, error) {
	if r == nil {
		return nil, errUnknownRun
	}

	if !r.st.Closed() {
		return nil, errRunOpen
	}

	if cmd.Message == nil {
		return nil, fmt.Errorf("%w: continue_run requires message", errBadCommand)
	}

	return []durable.Record{{Type: durable.RecordRunContinued, Message: cmd.Message}}, nil
}

func sendInput(r *run, cmd durable.Command) ([]durable.Record, error) {
	if r == nil {
		return nil, errUnknownRun
	}

	if r.st.Closed() {
		return nil, errRunClosed
	}

	aw := r.st.Awaiting
	if aw != nil && aw.Kind == durable.AwaitInput && (aw.Name == "" || aw.Name == cmd.Name) {
		return []durable.Record{{
			Type: durable.RecordInputDelivered, ToolCallID: aw.ToolCallID, Name: cmd.Name, Payload: cmd.Payload,
		}}, nil
	}

	return []durable.Record{{Type: durable.RecordInputReceived, Name: cmd.Name, Payload: cmd.Payload}}, nil
}

func appendMessage(r *run, cmd durable.Command) ([]durable.Record, error) {
	if cmd.Message == nil {
		return nil, errBadCommand
	}

	switch {
	case cmd.MessageIndex == len(r.st.Messages):
		return []durable.Record{{Type: durable.RecordMessageAppended, MessageIndex: cmd.MessageIndex, Message: cmd.Message}}, nil
	case cmd.MessageIndex < len(r.st.Messages):
		return nil, nil // duplicate from a retried send; already journaled
	default:
		return nil, fmt.Errorf("%w: got %d, have %d", errMessageIndex, cmd.MessageIndex, len(r.st.Messages))
	}
}

func suspendRun(r *run, cmd durable.Command) ([]durable.Record, error) {
	aw := cmd.Awaiting
	if aw == nil || aw.ToolCallID == "" {
		return nil, errBadCommand
	}

	if aw.Kind == durable.AwaitInput {
		if p, ok := findPending(r.st.PendingInputs, aw.Name); ok {
			return []durable.Record{
				{Type: durable.RecordRunSuspended, Awaiting: aw},
				{Type: durable.RecordInputDelivered, ToolCallID: aw.ToolCallID, Name: p.Name, Payload: p.Payload},
			}, nil
		}
	}

	return []durable.Record{{Type: durable.RecordRunSuspended, Awaiting: aw}}, nil
}

func attemptFailed(r *run, cmd durable.Command, now time.Time) ([]durable.Record, error) {
	failures := r.st.Failures + 1
	retryable := cmd.Retryable == nil || *cmd.Retryable
	exhausted := !retryable || r.st.RetryPolicy.Exhausted(failures)

	rec := durable.Record{
		Type:      durable.RecordAttemptFailed,
		Error:     cmd.Error,
		Failures:  failures,
		Exhausted: exhausted,
	}

	if !exhausted {
		at := now.Add(r.st.RetryPolicy.Backoff(failures))
		rec.RetryAt = &at
	}

	return []durable.Record{rec}, nil
}

// pickRollout selects a target deterministically by hashing the run id.
func pickRollout(runID string, targets []durable.RolloutTarget) *durable.RolloutTarget {
	var total uint32
	for _, t := range targets {
		total += t.Weight
	}

	if total == 0 {
		return nil
	}

	h := fnv.New32a()
	_, _ = h.Write([]byte(runID))
	point := h.Sum32() % total

	for i := range targets {
		if point < targets[i].Weight {
			return &targets[i]
		}

		point -= targets[i].Weight
	}

	return nil
}

// timerPayload is the tool result delivered when a durable timer fires.
func timerPayload(at time.Time) json.RawMessage {
	b, _ := json.Marshal(map[string]string{"fired_at": at.UTC().Format(time.RFC3339Nano)}) //nolint:errchkjson // static shape

	return b
}

// result builds the Result published when a run closes.
func result(st durable.RunState) durable.Result {
	res := durable.Result{
		RunID:        st.RunID,
		Agent:        st.Agent,
		Version:      st.Version,
		Status:       st.Status,
		FinishReason: st.FinishReason,
		Error:        st.Error,
		Usage:        st.Usage,
		Metadata:     st.Metadata,
		StartedAt:    st.StartedAt,
	}

	if st.ClosedAt != nil {
		res.ClosedAt = *st.ClosedAt
	}

	for i := len(st.Messages) - 1; i >= 0; i-- {
		if st.Messages[i].Role == llm.RoleAssistant {
			m := st.Messages[i]
			res.FinalMessage = &m

			break
		}
	}

	return res
}
