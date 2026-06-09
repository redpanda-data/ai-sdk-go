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

package runner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"time"

	"github.com/google/uuid"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Result is the caller-supplied resume payload for a single pending
// tool call. Exactly one of Output / Error should be set; if both are
// empty the call is treated as a no-op cancellation with a generic
// error.
type Result struct {
	// CallID is the originating tool call ID that paused.
	CallID string

	// Output is the final tool result JSON. Set this when the external
	// job succeeded and you have the final payload.
	Output json.RawMessage

	// Error is a tool-error message. When non-empty the runtime emits
	// an IsError response to the model.
	Error string

	// Metadata is optional audit/transport context. Stored on the
	// ResumeReceipt but NOT included in the duplicate-detection hash.
	Metadata map[string]any
}

// ErrResumeConflict is returned when a duplicate Resume arrives for an
// already-resolved call ID with a different result payload. Callers
// that retry safely (same payload) get ResumeAcknowledgedEvent instead.
var ErrResumeConflict = errors.New("runner: resume conflict")

// ErrPendingCallNotFound is returned when the caller submits a Result
// for a call ID that has no pending entry (and no receipt, so it can't
// be a duplicate either).
var ErrPendingCallNotFound = errors.New("runner: pending tool call not found")

// Resume submits one or more tool results to a paused session. It is
// idempotent for at-least-once delivery systems: a duplicate submission
// with the same result hash emits ResumeAcknowledgedEvent and does not
// touch the session.
//
// Resume mutates EAGERLY: by the time it returns, every result has been
// validated, applied, and the session saved. The returned stream first
// replays the events produced while applying the results, then — once
// no pending calls remain — streams the agent's continuation. Dropping
// the stream skips only the model continuation, never the mutation.
// The continuation re-acquires the session lock when ranged; another
// Run/Resume may interleave between the mutation and the continuation,
// which is safe because the session is fully consistent after the save.
//
// A non-nil error means nothing was applied: session load/save failed,
// or every supplied result was rejected (errors joined). When at least
// one result applies, per-result rejections surface on the stream
// instead.
//
// userID is the actor performing this resume — not necessarily the
// original session owner. Applications use ResumeAuthorizer to enforce
// per-operation policy.
func (r *Runner) Resume(
	ctx context.Context,
	userID string,
	sessionID string,
	results ...Result,
) (iter.Seq2[agent.Event, error], error) {
	return r.resume(ctx, userID, sessionID, ResumeOperationResume, results)
}

func (r *Runner) resume(
	ctx context.Context,
	userID string,
	sessionID string,
	op ResumeOperation,
	results []Result,
) (iter.Seq2[agent.Event, error], error) {
	var (
		items         []streamItem
		needsContinue bool
	)

	err := r.config.sessionStore.WithSessionLock(ctx, sessionID, func(ctx context.Context) error {
		var innerErr error
		items, needsContinue, innerErr = r.applyResumeBatch(ctx, userID, sessionID, op, results)

		return innerErr
	})
	if err != nil {
		return nil, err
	}

	stream := func(yield func(agent.Event, error) bool) {
		for _, it := range items {
			if !yield(it.evt, it.err) {
				return
			}
		}

		if !needsContinue {
			return
		}

		// Continuation: re-acquire the lock and reload — the mutation
		// lock was released when Resume returned.
		lockErr := r.config.sessionStore.WithSessionLock(ctx, sessionID, func(ctx context.Context) error {
			sess, err := r.config.sessionStore.Load(ctx, sessionID)
			if err != nil {
				yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionLoad, err))
				return errYielded
			}

			if len(sess.PendingToolCalls) > 0 {
				// A concurrent Run/Resume paused the session again
				// between our mutation and this continuation; it owns
				// the next step.
				return nil
			}

			r.runAgentAfterResume(ctx, sess, yield)

			return nil
		})
		if lockErr != nil && !errors.Is(lockErr, errYielded) {
			yield(nil, lockErr)
		}
	}

	return stream, nil
}

// streamItem is a buffered (event, error) pair produced during the
// eager mutation phase and replayed on the returned stream.
type streamItem struct {
	evt agent.Event
	err error
}

// errYielded is an internal sentinel used to tell WithSessionLock that
// the inner function already surfaced its error via yield. It must not
// escape this package.
var errYielded = errors.New("runner: error already yielded")

// applyResumeBatch loads the session, applies every result, and saves.
// It runs under the session lock. The returned items replay on the
// stream; needsContinue reports whether the agent loop should re-enter.
func (r *Runner) applyResumeBatch(
	ctx context.Context,
	userID string,
	sessionID string,
	op ResumeOperation,
	results []Result,
) ([]streamItem, bool, error) {
	sess, err := r.config.sessionStore.Load(ctx, sessionID)
	if err != nil {
		return nil, false, fmt.Errorf("%w: %w", agent.ErrSessionLoad, err)
	}

	session.MigratePendingToolCalls(sess)
	r.sweepExpired(sess, time.Now().UTC())

	envelope := func() agent.EventEnvelope {
		return agent.EventEnvelope{
			InvocationID: "resume-" + uuid.NewString(),
			SessionID:    sessionID,
			At:           time.Now().UTC(),
		}
	}

	var (
		items      []streamItem
		rejections []error
		anyApplied bool
	)

	for _, res := range results {
		applied, evts, resErr := r.applyResume(ctx, userID, sess, res, op, envelope)

		for _, evt := range evts {
			items = append(items, streamItem{evt: evt})
		}

		if resErr != nil {
			items = append(items, streamItem{err: resErr})
			rejections = append(rejections, resErr)
		}

		anyApplied = anyApplied || applied
	}

	// All results rejected → fail eagerly so callers that don't range
	// the stream still see the problem.
	if !anyApplied && len(rejections) > 0 && len(rejections) == len(results) {
		return nil, false, errors.Join(rejections...)
	}

	if err := r.config.sessionStore.Save(ctx, sess); err != nil {
		return nil, false, fmt.Errorf("%w: %w", agent.ErrSessionSave, err)
	}

	// If there are still pending calls, emit a paused end event; the
	// agent loop is only re-entered when all pending calls are resolved.
	if len(sess.PendingToolCalls) > 0 {
		items = append(items, streamItem{evt: agent.InvocationEndEvent{
			Envelope:     envelope(),
			FinishReason: agent.FinishReasonPaused,
			PendingCalls: summariesFromSession(sess),
		}})

		return items, false, nil
	}

	return items, anyApplied, nil
}

// applyResume reconciles a single Result against the session. Returns
// applied=true when the session was mutated (or the call was already
// resolved with the same payload, since that still represents a
// successful caller intent).
//

func (r *Runner) applyResume(
	ctx context.Context,
	userID string,
	sess *session.State,
	res Result,
	op ResumeOperation,
	envelope func() agent.EventEnvelope,
) (bool, []agent.Event, error) {
	resultHash := computeResultHash(res)

	// Idempotency: a duplicate with the same hash gets acked without
	// session mutation; a different hash is a conflict.
	if receipt, ok := sess.ResumeReceipts[res.CallID]; ok {
		if receipt.ResultHash == resultHash {
			ack := agent.ResumeAcknowledgedEvent{
				Envelope: envelope(),
				CallID:   res.CallID,
				Status:   "already_resolved",
			}

			return false, []agent.Event{ack}, nil
		}

		return false, nil, fmt.Errorf("%w: call_id=%s", ErrResumeConflict, res.CallID)
	}

	pc, ok := sess.PendingToolCalls[res.CallID]
	if !ok {
		return false, nil, fmt.Errorf("%w: call_id=%s", ErrPendingCallNotFound, res.CallID)
	}

	if err := r.runAuthorizer(ctx, userID, sess.ID, pc, res, op); err != nil {
		return false, nil, fmt.Errorf("resume not authorized: %w", err)
	}

	switch tool.ResumeMode(pc.Resume) {
	case tool.ResumeWithMessage:
		// Callers must use Runner.Run with the user's next message
		// for these pauses. Reject here so they don't silently no-op.
		return false, nil, fmt.Errorf("call %s expects ResumeWithMessage; use Runner.Run with a user message", res.CallID)

	case tool.ResumeWithToolResponse:
		applyToolResponseResume(sess, &pc, res)
	case tool.ResumeWithReentry:
		if applyReentryResume(ctx, r, sess, &pc, res) {
			// Chained pause: the call is pending again under the same
			// ID. Storing a receipt here would make the new pause
			// unresumable (the receipt check short-circuits before the
			// pending lookup), so receipts are only written for
			// terminal resolutions.
			chained := agent.ToolPendingEvent{
				Envelope:    envelope(),
				PendingCall: summarizePendingCall(sess.PendingToolCalls[res.CallID]),
				Placeholder: *llm.NewToolResponsePart(pc.ID, pc.Name, sess.PendingToolCalls[res.CallID].LastOutput),
			}

			return false, []agent.Event{chained}, nil
		}
	default:
		return false, nil, fmt.Errorf("call %s has unknown resume mode %q", res.CallID, pc.Resume)
	}

	// Resolve aliases (coalesced calls) with the same payload.
	for _, alias := range pc.CoalescedIDs {
		if ac, aliasOK := sess.PendingToolCalls[alias]; aliasOK {
			applyToolResponseResume(sess, &ac, res)
		}
	}

	storeReceipt(sess, res.CallID, resultHash, res.Metadata)

	return true, nil, nil
}

// applyToolResponseResume replaces the placeholder ToolResponsePart for
// pc.ID in session history with a final response, removes the pending
// entry, and writes a ResumeReceipt.
func applyToolResponseResume(sess *session.State, pc *session.PendingToolCall, res Result) {
	resp := makeResumeResponse(pc, res)
	replacePlaceholder(sess, pc.ID, resp)

	delete(sess.PendingToolCalls, pc.ID)
}

// toolResumer is implemented by agents (llmagent in particular) that can
// re-enter a paused tool call through their tool interceptor chain. This
// matters for interceptor-created pauses — approval gates — where the
// interceptor, not the tool, must consume the resume decision. See
// agent.ToolCallInfo.Resume.
type toolResumer interface {
	ExecuteToolResume(ctx context.Context, sess *session.State, pc session.PendingToolCall, payload *tool.ResumePayload) (tool.Execution, error)
}

// applyReentryResume re-enters the paused tool call with Call.Resume
// populated — through the agent's interceptor chain when the agent
// supports it, otherwise directly against the registry. The returned
// bool reports whether the call paused again (chained pause): in that
// case the new PendingToolCall replaces the old one and NO receipt must
// be stored, or the chained pause could never be resumed.
func applyReentryResume(ctx context.Context, r *Runner, sess *session.State, pc *session.PendingToolCall, res Result) bool {
	payload := &tool.ResumePayload{
		PriorState: pc.State,
		Result:     res.Output,
		Error:      res.Error,
		Metadata:   res.Metadata,
		Progress:   progressEntriesFromSession(pc.Progress),
	}

	exec, execErr := executeReentry(ctx, r, sess, pc, payload)
	if execErr != nil {
		applyToolResponseResume(sess, pc, Result{CallID: res.CallID, Error: execErr.Error()})
		return false
	}

	if exec.Await != nil {
		// Chained pause: replace the pause-specific fields wholesale —
		// inheriting the previous pause's expiry, correlation ID, or
		// metadata would mislabel the new pause. Call ID and origin
		// time stay.
		now := time.Now().UTC()
		updated := *pc
		updated.Reason = string(exec.Await.Reason)
		updated.Resume = string(exec.Await.Resume)
		updated.Message = exec.Await.Message
		updated.Prompt = exec.Await.Prompt
		updated.State = exec.Await.State
		updated.LastOutput = exec.Output
		updated.CorrelationID = exec.Await.CorrelationID
		updated.Metadata = exec.Await.Metadata
		updated.ExpiresAt = nil

		if exec.Await.ExpiresAt != nil {
			t := *exec.Await.ExpiresAt
			updated.ExpiresAt = &t
		} else if exec.Await.Timeout > 0 {
			t := now.Add(exec.Await.Timeout)
			updated.ExpiresAt = &t
		}

		sess.PendingToolCalls[pc.ID] = updated
		// Update the placeholder in history with the new Output too.
		replacePlaceholder(sess, pc.ID, llm.NewToolResponsePart(pc.ID, pc.Name, exec.Output))

		return true
	}

	applyToolResponseResume(sess, pc, Result{CallID: res.CallID, Output: exec.Output})

	return false
}

// executeReentry routes the re-entry through the agent's interceptor
// chain when available, falling back to the raw registry tool, and as a
// last resort treating the supplied payload as the final tool response.
func executeReentry(ctx context.Context, r *Runner, sess *session.State, pc *session.PendingToolCall, payload *tool.ResumePayload) (tool.Execution, error) {
	if tr, ok := r.config.agent.(toolResumer); ok {
		return tr.ExecuteToolResume(ctx, sess, *pc, payload)
	}

	registry := registryFromAgent(r.config.agent)
	if registry == nil {
		// No registry wired: degrade to recording the payload as the
		// final output (or error) without re-entering anything.
		if payload.Error != "" {
			return tool.Execution{}, errors.New(payload.Error)
		}

		return tool.Execution{Output: payload.Result}, nil
	}

	if _, err := registry.Get(pc.Name); err != nil {
		// Synthesize a tool error so old pending calls complete with an
		// explicit model-visible failure rather than hanging.
		return tool.Execution{}, fmt.Errorf("pending tool %q is no longer registered", pc.Name)
	}

	req := &llm.ToolRequestPart{ID: pc.ID, Name: pc.Name, Arguments: pc.Arguments}
	out := registry.Resume(ctx, tool.InvocationInfo{SessionID: sess.ID}, req, payload)

	return out.Execution, out.Err
}

// makeResumeResponse builds the wire ToolResponsePart for a resolved
// pending call.
func makeResumeResponse(pc *session.PendingToolCall, res Result) *llm.ToolResponsePart {
	if res.Error != "" {
		return llm.NewToolErrorPart(pc.ID, pc.Name, res.Error)
	}

	out := res.Output
	if len(out) == 0 {
		// Empty success → preserve the last placeholder so the model
		// at least sees the original {"status":"queued"}-style payload.
		out = pc.LastOutput
	}

	if len(out) == 0 {
		out = json.RawMessage(`{}`)
	}

	return llm.NewToolResponsePart(pc.ID, pc.Name, out)
}

// replacePlaceholder swaps the matching ToolResponsePart in session
// history with the supplied resolved part. If no placeholder exists
// (e.g. the pause was created without one), it appends a new user
// message with the response so providers still see a valid sequence.
func replacePlaceholder(sess *session.State, callID string, resp *llm.ToolResponsePart) {
	for i := range sess.Messages {
		if sess.Messages[i].Role != llm.RoleUser {
			continue
		}

		content := sess.Messages[i].Content
		for j := range content {
			if tr, ok := content[j].(*llm.ToolResponsePart); ok && tr != nil && tr.ID == callID {
				content[j] = resp
				return
			}
		}
	}

	// Fallback: append.
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, resp))
}

// computeResultHash hashes the semantic resume payload (output or
// error) using JCS canonicalization. Metadata is intentionally
// excluded — the first successful resume's metadata wins, and
// at-least-once duplicates with different webhook delivery IDs should
// still be acked.
func computeResultHash(res Result) string {
	payload := struct {
		Output json.RawMessage `json:"output,omitempty"`
		Error  string          `json:"error,omitempty"`
	}{Output: res.Output, Error: res.Error}

	raw, err := json.Marshal(payload)
	if err != nil {
		// Should not happen; fall back to a hash of the call ID so we
		// still produce a stable value.
		sum := sha256.Sum256([]byte(res.CallID))
		return hex.EncodeToString(sum[:])
	}

	hash, err := tool.ArgumentsHash(raw)
	if err != nil {
		sum := sha256.Sum256(raw)
		return hex.EncodeToString(sum[:])
	}

	return hash
}

func storeReceipt(sess *session.State, callID, hash string, metadata map[string]any) {
	if sess.ResumeReceipts == nil {
		sess.ResumeReceipts = make(map[string]session.ResumeReceipt)
	}

	sess.ResumeReceipts[callID] = session.ResumeReceipt{
		CallID:     callID,
		ResultHash: hash,
		ResolvedAt: time.Now().UTC(),
		Metadata:   metadata,
	}
}

// sweepExpired removes pending calls whose ExpiresAt has passed and
// replaces their placeholders with synthetic tool-error responses. This
// is the passive-timeout mechanism: it runs on every session-mutating
// operation, so the SDK does not need a background timer.
func (r *Runner) sweepExpired(sess *session.State, now time.Time) {
	for id, pc := range sess.PendingToolCalls {
		if pc.ExpiresAt == nil || pc.ExpiresAt.After(now) {
			continue
		}

		errMsg := fmt.Sprintf("pending tool %q expired at %s", pc.Name, pc.ExpiresAt.Format(time.RFC3339))
		applyToolResponseResume(sess, &pc, Result{CallID: id, Error: errMsg})
		storeReceipt(sess, id, computeResultHash(Result{CallID: id, Error: errMsg}), nil)
	}
}

// summariesFromSession projects the pending map into a stable slice for
// InvocationEndEvent. Order is best-effort — callers should not rely
// on iteration order in tests.
func summariesFromSession(sess *session.State) []agent.PendingCallSummary {
	out := make([]agent.PendingCallSummary, 0, len(sess.PendingToolCalls))
	for _, pc := range sess.PendingToolCalls {
		out = append(out, summarizePendingCall(pc))
	}

	return out
}

// summarizePendingCall projects one pending call into the event-facing
// summary shape.
func summarizePendingCall(pc session.PendingToolCall) agent.PendingCallSummary {
	return agent.PendingCallSummary{
		CallID:        pc.ID,
		ToolName:      pc.Name,
		Reason:        tool.AwaitReason(pc.Reason),
		Resume:        tool.ResumeMode(pc.Resume),
		Message:       pc.Message,
		Prompt:        pc.Prompt,
		CorrelationID: pc.CorrelationID,
		ExpiresAt:     pc.ExpiresAt,
	}
}

func progressEntriesFromSession(src []session.ProgressEntry) []tool.ProgressEntry {
	out := make([]tool.ProgressEntry, len(src))
	for i, p := range src {
		out[i] = tool.ProgressEntry{At: p.At, Payload: p.Payload}
	}

	return out
}

func (r *Runner) runAuthorizer(
	ctx context.Context,
	userID string,
	sessionID string,
	pc session.PendingToolCall,
	res Result,
	op ResumeOperation,
) error {
	if r.config.authorize == nil {
		return nil
	}

	return r.config.authorize(ctx, ResumeInfo{
		UserID:      userID,
		SessionID:   sessionID,
		PendingCall: pc,
		Result:      res,
		Operation:   op,
	})
}

// runAgentAfterResume re-enters the agent loop using the resumed
// session. The new invocation produces a fresh InvocationID.
func (r *Runner) runAgentAfterResume(
	ctx context.Context,
	sess *session.State,
	yield func(agent.Event, error) bool,
) {
	inv := agent.NewInvocationMetadata(sess, r.config.agent.Info())

	for evt, err := range r.config.agent.Run(ctx, inv) {
		if err != nil {
			if !yield(nil, err) {
				return
			}

			continue
		}

		// Save after each MessageEvent (same incremental persistence
		// pattern as Run).
		if _, ok := evt.(agent.MessageEvent); ok {
			if err := r.config.sessionStore.Save(ctx, sess); err != nil {
				_ = yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionSave, err))
				return
			}
		}

		if !yield(evt, nil) {
			return
		}

		if _, ok := evt.(agent.InvocationEndEvent); ok {
			break
		}
	}

	// Final save after the loop, mirroring Runner.Run's defer behavior.
	if err := r.config.sessionStore.Save(ctx, sess); err != nil {
		yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionSave, err))
	}
}

// Progress records a non-terminal update on a pending tool call and
// returns the ToolProgressEvent for the caller to forward. It mutates
// eagerly and never calls the model.
func (r *Runner) Progress(
	ctx context.Context,
	userID string,
	sessionID string,
	callID string,
	payload json.RawMessage,
) (agent.ToolProgressEvent, error) {
	var out agent.ToolProgressEvent

	err := r.config.sessionStore.WithSessionLock(ctx, sessionID, func(ctx context.Context) error {
		sess, err := r.config.sessionStore.Load(ctx, sessionID)
		if err != nil {
			return fmt.Errorf("%w: %w", agent.ErrSessionLoad, err)
		}

		session.MigratePendingToolCalls(sess)

		pc, ok := sess.PendingToolCalls[callID]
		if !ok {
			return fmt.Errorf("%w: call_id=%s", ErrPendingCallNotFound, callID)
		}

		if err := r.runAuthorizer(ctx, userID, sessionID, pc, Result{CallID: callID, Output: payload}, ResumeOperationProgress); err != nil {
			return fmt.Errorf("progress not authorized: %w", err)
		}

		pc.Progress = append(pc.Progress, session.ProgressEntry{At: time.Now().UTC(), Payload: payload})
		sess.PendingToolCalls[callID] = pc

		if err := r.config.sessionStore.Save(ctx, sess); err != nil {
			return fmt.Errorf("%w: %w", agent.ErrSessionSave, err)
		}

		out = agent.ToolProgressEvent{
			Envelope: agent.EventEnvelope{
				InvocationID: "progress-" + uuid.NewString(),
				SessionID:    sessionID,
				At:           time.Now().UTC(),
			},
			CallID:  callID,
			Payload: payload,
		}

		return nil
	})

	return out, err
}

// Cancel resolves a pending tool call with a tool-error response. It
// mutates eagerly like Resume; the returned stream carries the agent
// continuation unless other pending calls remain. The ResumeAuthorizer
// sees ResumeOperationCancel.
func (r *Runner) Cancel(
	ctx context.Context,
	userID string,
	sessionID string,
	callID string,
	reason string,
) (iter.Seq2[agent.Event, error], error) {
	if reason == "" {
		reason = "canceled"
	}

	return r.resume(ctx, userID, sessionID, ResumeOperationCancel, []Result{{CallID: callID, Error: reason}})
}

// registryFromAgent returns the tool registry attached to a, if any.
// Looking it up via an exported interface avoids dragging the
// llmagent package into runner.
func registryFromAgent(a agent.Agent) tool.Registry {
	if r, ok := a.(interface{ Tools() tool.Registry }); ok {
		return r.Tools()
	}

	return nil
}
