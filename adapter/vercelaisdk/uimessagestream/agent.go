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

package uimessagestream

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// Handler serves a chat resource over the Vercel AI SDK UI Message Stream
// protocol, backed by a full agent.Agent — its system prompt, tool registry
// (MCP + subagents-as-tools), interceptor chain (OTel/transcripts), and agentic
// tool-calling loop — with sessions persisted server-side in the store.
//
// Routes, relative to the mount point (mount with http.StripPrefix, see the
// package README):
//
//	POST   /{$}   run a turn; responds with the UI Message Stream (SSE)
//	GET    /{$}   list chats (JSON; 501 when WithSessionKey is configured or
//	              the store does not support listing)
//	GET    /{id}  chat history as UI messages (JSON, for useChat({messages}))
//	DELETE /{id}  delete the chat (204)
//
// History is server-authoritative, following the Vercel AI SDK's canonical
// persistence pattern: POST loads the session by chat id (creating it on first
// use), appends the single posted user message, runs the agent, and saves —
// incrementally after each completed assistant message and finally when the run
// ends, with a context that survives client disconnects. Regenerate
// (trigger "regenerate-message") truncates to the last user message and re-runs
// without appending. Posted history beyond the last message is ignored; the
// store is the source of truth.
//
// The client-supplied chat id maps to the storage key via WithSessionKey (the
// tenant-isolation seam); concurrent POSTs to the same chat are serialized
// per-process by a keyed lock.
//
// This takes an agent.Agent, not a *runner.Runner, even though the runner has
// the load-append-run-save shape: regenerate needs truncate-without-append,
// which the runner cannot express, and the handler's saves must outlive the
// request context (see persistingAgent).
//
// Contract: tool calls are taken from MessageEvent (the parts an assistant
// message carries), which llmagent always emits — the same assumption
// adapter/a2a's Executor makes. A hypothetical agent that reports tool calls
// ONLY via ToolRequestEvent and never in a MessageEvent is not supported;
// ToolRequestEvent is treated as an informational breadcrumb, not a source of
// tool-input chunks, so its calls would not surface. Every agent.Agent in this
// SDK (llmagent) satisfies the contract.
func Handler(ag agent.Agent, store session.Store, opts ...Option) http.Handler {
	cfg := &config{
		logger:       slog.Default(),
		maxBodyBytes: 1 << 20, // 1MB
		onError:      defaultErrorMapper,
	}
	for _, o := range opts {
		o(cfg)
	}

	h := &chatHandler{agent: ag, store: store, cfg: cfg, locks: newKeyedMutex()}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /{$}", h.handlePost)
	mux.HandleFunc("GET /{$}", h.handleList)
	mux.HandleFunc("GET /{id}", h.handleGet)
	mux.HandleFunc("DELETE /{id}", h.handleDelete)
	h.mux = mux

	return h
}

type chatHandler struct {
	agent agent.Agent
	store session.Store
	cfg   *config
	locks *keyedMutex
	mux   *http.ServeMux
}

func (h *chatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// http.StripPrefix turns a request for the exact mount path into "", which
	// ServeMux would 301-redirect — and a redirect on POST drops the body.
	if r.URL.Path == "" {
		r = r.Clone(r.Context())
		r.URL.Path = "/"
	}

	h.mux.ServeHTTP(w, r)
}

// resolveKey maps the client-supplied chat id to the storage key.
func (h *chatHandler) resolveKey(r *http.Request, chatID string) (string, error) {
	if h.cfg.sessionKey == nil {
		return chatID, nil
	}

	return h.cfg.sessionKey(r, chatID)
}

func (h *chatHandler) handlePost(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, h.cfg.maxBodyBytes)

	var body chatRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	trigger := body.Trigger
	if trigger == "" {
		trigger = triggerSubmit
	}

	if trigger != triggerSubmit && trigger != triggerRegenerate {
		http.Error(w, "unknown trigger", http.StatusBadRequest)
		return
	}

	// useChat always sends an id. Minting one server-side would create a
	// session the client can never address again — an orphaned store entry
	// with per-request amnesia — so an id-less request is a client bug: 400.
	chatID := body.ID
	if chatID == "" {
		http.Error(w, "missing chat id", http.StatusBadRequest)
		return
	}

	key, err := h.resolveKey(r, chatID)
	if err != nil {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}

	// Serialize per chat: concurrent POSTs interleaving load-modify-save would
	// lose messages.
	unlock := h.locks.lock(key)
	defer unlock()

	sess, ok := h.prepareSession(w, r, trigger, key, &body)
	if !ok {
		return
	}

	// Pre-run save: a store failure is a clean 500 while headers are still
	// writable, regenerate's truncation persists even if the run fails, and a
	// concurrent GET already sees the user's message during the run.
	if err := h.store.Save(context.WithoutCancel(r.Context()), sess); err != nil {
		h.cfg.logger.Error("failed to save session", "sessionId", key, "error", err)
		http.Error(w, "failed to save session", http.StatusInternalServerError)

		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	inv := agent.NewInvocationMetadata(sess, h.agent.Info())

	setSSEHeaders(w)

	ew := &EventWriter{w: w, f: flusher}
	pag := &persistingAgent{inner: h.agent, store: h.store, sess: sess, logger: h.cfg.logger}
	StreamAgent(r.Context(), pag, inv, ew, h.cfg.logger, h.cfg.onError)
}

// prepareSession loads and mutates the session for the trigger: submit appends
// the posted user message (creating the session on first use), regenerate
// truncates to the last user message. On failure it writes the HTTP error and
// returns ok=false.
func (h *chatHandler) prepareSession(w http.ResponseWriter, r *http.Request, trigger, key string, body *chatRequest) (*session.State, bool) {
	switch trigger {
	case triggerSubmit:
		msg := body.Message
		if msg == nil && len(body.Messages) > 0 {
			// Default (untrimmed) transport: only the last message counts;
			// server-side history is authoritative.
			msg = &body.Messages[len(body.Messages)-1]
		}

		if msg == nil {
			http.Error(w, "missing message", http.StatusBadRequest)
			return nil, false
		}

		userMsg, ok := convertUserMessage(*msg)
		if !ok {
			http.Error(w, "message must be a user message with text", http.StatusBadRequest)
			return nil, false
		}

		sess, err := loadOrCreate(r.Context(), h.store, key)
		if err != nil {
			h.cfg.logger.Error("failed to load session", "sessionId", key, "error", err)
			http.Error(w, "failed to load session", http.StatusInternalServerError)

			return nil, false
		}

		sess.Messages = append(sess.Messages, userMsg)

		return sess, true

	default: // triggerRegenerate, validated by the caller
		sess, err := h.store.Load(r.Context(), key)
		if errors.Is(err, session.ErrNotFound) {
			http.Error(w, "chat not found", http.StatusNotFound)
			return nil, false
		}

		if err != nil {
			h.cfg.logger.Error("failed to load session", "sessionId", key, "error", err)
			http.Error(w, "failed to load session", http.StatusInternalServerError)

			return nil, false
		}

		// body.MessageID is accepted but unused: sessions persist model
		// messages without UI ids, so v1 always regenerates from the last user
		// message (the common retry flow).
		if !truncateForRegenerate(sess) {
			http.Error(w, "no user message to regenerate from", http.StatusConflict)
			return nil, false
		}

		return sess, true
	}
}

// agentStreamer maps an agent.Event stream onto UI Message Stream chunks. It
// tracks step state so the outbound stream stays a strictly balanced grammar
// (start/finish, start-step/finish-step, text-start/text-end) — the reference
// client rejects or hangs on a malformed stream. Steps open lazily on the first
// content chunk (never on a trusted status event) and every terminal path closes
// the open step before the terminator.
type agentStreamer struct {
	sw                *streamWriter
	ew                *EventWriter
	logger            *slog.Logger
	onError           ErrorMapper
	stepOpen          bool
	streamedText      bool
	streamedReasoning bool
	// pending holds toolCallIds emitted as tool-input-available but not yet
	// resolved by a ToolResponseEvent, so terminal paths can close them as
	// tool-output-error rather than strand the client's dynamic tool part.
	pending map[string]struct{}
	// errored records that an error chunk was already written, so an errored
	// finish does not emit a second, generic one (which would call the client's
	// onError twice and mask the first, mapped error).
	errored bool
}

func (as *agentStreamer) ensureStep() error {
	if as.stepOpen {
		return nil
	}

	if err := as.ew.WriteChunk(Chunk{"type": "start-step"}); err != nil {
		return err
	}

	as.stepOpen = true

	return nil
}

// endStep closes any open text/reasoning span and the step. Safe to call when no
// step is open (no-op), so terminal paths can call it unconditionally.
func (as *agentStreamer) endStep() error {
	if !as.stepOpen {
		return nil
	}

	as.sw.closeSpans()

	if err := as.ew.WriteChunk(Chunk{"type": "finish-step"}); err != nil {
		return err
	}

	as.stepOpen = false
	as.streamedText = false
	as.streamedReasoning = false

	return nil
}

// writeDynamicToolRequest emits a dynamic tool-input pair. Agent tools (MCP tools
// and subagents-as-tools) are discovered at runtime and are unknown to the
// browser's compile-time tool registry, so they MUST be tagged dynamic:true —
// otherwise useChat materializes a statically-typed tool-<name> part it cannot
// correlate against a client-side schema.
func (as *agentStreamer) writeDynamicToolRequest(tr *llm.ToolRequestPart) error {
	if tr == nil {
		return nil
	}

	if err := as.ew.WriteChunk(Chunk{
		"type": "tool-input-start", "toolCallId": tr.ID, "toolName": tr.Name, "dynamic": true,
	}); err != nil {
		return err
	}

	var input any
	if len(tr.Arguments) > 0 {
		if err := json.Unmarshal(tr.Arguments, &input); err != nil {
			as.logger.Warn("failed to unmarshal tool input", "toolCallId", tr.ID, "error", err)
		}
	}

	if err := as.ew.WriteChunk(Chunk{
		"type": "tool-input-available", "toolCallId": tr.ID, "toolName": tr.Name, "input": input, "dynamic": true,
	}); err != nil {
		return err
	}

	// Outstanding until a ToolResponseEvent resolves it (or a terminal path
	// closes it as tool-output-error).
	as.pending[tr.ID] = struct{}{}

	return nil
}

func (as *agentStreamer) writeDynamicToolResponse(tr *llm.ToolResponsePart) error {
	if tr == nil {
		return nil
	}

	// A tool-output-* chunk is only valid after a tool-input-* for the same
	// toolCallId. If no input was emitted (e.g. llmagent's incomplete-tool-call
	// recovery yields a ToolResponseEvent before any MessageEvent carries the
	// call), emitting the output would error the client stream — skip it.
	if _, ok := as.pending[tr.ID]; !ok {
		return nil
	}

	delete(as.pending, tr.ID)

	if tr.IsError {
		// Route the tool error through onError before it reaches the browser, so
		// failed MCP/subagent tools cannot leak server-side detail (default), while
		// a custom onError still sees the actual error string. The registry stores
		// the failure as a {"error": "..."} JSON payload; unwrap it so onError gets
		// the message, not the JSON envelope.
		return as.ew.WriteChunk(Chunk{
			"type": "tool-output-error", "toolCallId": tr.ID,
			"errorText": as.onError(errors.New(toolErrorText(tr.Result))), "dynamic": true,
		})
	}

	var output any
	if len(tr.Result) > 0 {
		if err := json.Unmarshal(tr.Result, &output); err != nil {
			as.logger.Warn("failed to unmarshal tool output", "toolCallId", tr.ID, "error", err)
		}
	}

	return as.ew.WriteChunk(Chunk{
		"type": "tool-output-available", "toolCallId": tr.ID, "output": output, "dynamic": true,
	})
}

// StreamAgent runs the agent and streams its events as UI Message Stream chunks.
// This is the core logic behind AgentHandler, exported for custom HTTP handlers.
// It never returns an error: all failures are surfaced as protocol chunks and the
// stream is always terminated with [DONE].
func StreamAgent(ctx context.Context, ag agent.Agent, inv *agent.InvocationMetadata, ew *EventWriter, logger *slog.Logger, onError ErrorMapper) {
	if logger == nil {
		logger = slog.Default()
	}

	if onError == nil {
		onError = defaultErrorMapper
	}

	messageID := generateMessageID()
	if err := ew.WriteChunk(Chunk{"type": "start", "messageId": messageID}); err != nil {
		return
	}

	as := &agentStreamer{
		sw:      newStreamWriter(ew, logger, onError),
		ew:      ew,
		logger:  logger,
		onError: onError,
		pending: make(map[string]struct{}),
	}

	for event, err := range ag.Run(ctx, inv) {
		if err != nil {
			// Cancellation: emit an abort (best-effort — the client may be gone).
			if ctx.Err() != nil {
				as.abort(ctx)
				return
			}

			logger.Error("agent run error", "error", err)

			as.terminate(onError(err))

			return
		}

		// A cancellation between turns surfaces as InvocationEndEvent{interrupted}
		// with no iterator error. Emit an abort (not a normal finish) so useChat
		// treats it as aborted rather than a completed assistant message.
		if end, ok := event.(agent.InvocationEndEvent); ok &&
			(end.FinishReason == agent.FinishReasonInterrupted || ctx.Err() != nil) {
			as.abort(ctx)
			return
		}

		if done := as.handleEvent(event, logger); done {
			return
		}
	}

	// The stream ended without an InvocationEndEvent. Terminate cleanly rather
	// than leaving the browser hanging on an open stream.
	logger.Warn("agent stream ended without InvocationEndEvent")
	as.terminate(onError(errors.New("incomplete agent run")))
}

// abort closes the open step and resolves any pending tool call before the
// abort terminator. Cancellation is a terminal path like any other: without the
// cleanup, a client that outlives the cancel is left with an unbalanced
// start-step or a dynamic tool part stuck in input-available.
func (as *agentStreamer) abort(ctx context.Context) {
	_ = as.endStep()
	as.closePendingTools()
	as.sw.writeAbort(ctx)
}

// terminate closes the open step, resolves any tool call still pending (so the
// client never strands a dynamic tool part in input-available), then emits the
// error + finish{error} + [DONE] terminator. The error chunk is skipped when a
// recoverable ErrorEvent already surfaced one — the terminal invariant is at
// most one error chunk, then finish{error}.
func (as *agentStreamer) terminate(errText string) {
	_ = as.endStep()
	as.closePendingTools()

	if !as.errored {
		_ = as.ew.WriteChunk(Chunk{"type": "error", "errorText": errText})
	}

	_ = as.ew.WriteChunk(Chunk{"type": "finish", "finishReason": finishReasonError})
	_ = as.ew.WriteDone()
}

// closePendingTools emits a tool-output-error for every tool call that was
// requested but never produced a ToolResponseEvent, transitioning the client's
// dynamic tool part out of input-available. Called on terminal paths.
func (as *agentStreamer) closePendingTools() {
	for id := range as.pending {
		_ = as.ew.WriteChunk(Chunk{
			"type": "tool-output-error", "toolCallId": id,
			"errorText": as.onError(errors.New("tool call did not complete")), "dynamic": true,
		})
	}

	as.pending = make(map[string]struct{})
}

// handleEvent maps one agent.Event onto UI chunks. It returns true when the
// stream has terminated ([DONE] written) and the caller must stop.
func (as *agentStreamer) handleEvent(event agent.Event, logger *slog.Logger) bool {
	switch e := event.(type) {
	case agent.AssistantDeltaEvent:
		as.handleDelta(e)

	case agent.MessageEvent:
		as.handleMessage(e)

	case agent.ToolResponseEvent:
		resp := e.Response
		_ = as.writeDynamicToolResponse(&resp)

	case agent.StatusEvent:
		// Keep the SSE connection warm across a (possibly slow) tool call. A
		// comment ping is not a protocol chunk, so it cannot desync the grammar.
		if e.Stage == agent.StatusStageToolExec {
			_ = as.ew.WritePing()
		}

	case agent.StreamResetEvent:
		// The retry interceptor is abandoning this attempt. Bytes already flushed
		// cannot be retracted, so close the abandoned step cleanly (text-end /
		// finish-step) and reset span state; the retry re-opens a fresh step.
		_ = as.endStep()

	case agent.ErrorEvent:
		// Observable (recoverable) error — surface it but keep streaming.
		logger.Warn("recoverable agent error", "message", e.Message)
		_ = as.ew.WriteChunk(Chunk{"type": "error", "errorText": as.onError(agentErrorEventErr(e))})
		as.errored = true

	case agent.InvocationEndEvent:
		as.handleInvocationEnd(e)
		return true

	default:
		// ToolRequestEvent is intentionally ignored: tool calls are sourced from
		// MessageEvent (which llmagent always emits and which carries them as
		// parts), exactly as adapter/a2a's Executor does. Handling ToolRequestEvent
		// separately would duplicate llmagent's emission and leave the tool-call
		// step ambiguous relative to the post-tool answer. Also covers future
		// event kinds.
	}

	return false
}

func (as *agentStreamer) handleDelta(e agent.AssistantDeltaEvent) {
	switch p := e.Delta.Part.(type) {
	case *llm.TextPart:
		if err := as.ensureStep(); err != nil {
			return
		}

		if err := as.sw.writeTextDelta(p.Text); err == nil {
			as.streamedText = true
		}
	case *llm.ReasoningPart:
		if err := as.ensureStep(); err != nil {
			return
		}

		if err := as.sw.writeReasoningDelta(p); err == nil {
			as.streamedReasoning = true
		}
	}
	// Tool-request deltas are coalesced into the tool-input pair at MessageEvent.
}

// handleMessage emits the complete assistant message for a turn: any tool-request
// parts as dynamic tool chunks, plus text/reasoning that did not already stream
// as deltas (non-streaming models), then closes the step.
func (as *agentStreamer) handleMessage(e agent.MessageEvent) {
	if err := as.ensureStep(); err != nil {
		return
	}

	for _, part := range e.Response.Message.Content {
		switch p := part.(type) {
		case *llm.TextPart:
			if !as.streamedText {
				_ = as.sw.writeTextDelta(p.Text)
			}
		case *llm.ReasoningPart:
			if !as.streamedReasoning {
				_ = as.sw.writeReasoningDelta(p)
			}
		case *llm.ToolRequestPart:
			as.emitToolRequest(p)
		}
	}

	_ = as.endStep()
}

// emitToolRequest closes any open text/reasoning span (a tool part cannot sit
// inside one) and writes the dynamic tool-input pair. Shared by the MessageEvent
// and ToolRequestEvent paths; writeDynamicToolRequest dedups by toolCallId.
func (as *agentStreamer) emitToolRequest(p *llm.ToolRequestPart) {
	_ = as.sw.endReasoning()
	_ = as.sw.endTextAndAdvance()
	_ = as.writeDynamicToolRequest(p)
}

// toolErrorText unwraps the tool failure payload. The registry stores failures
// as {"error":"..."}; return that message so a custom onError sees the real
// string, else fall back to the raw payload.
func toolErrorText(result json.RawMessage) string {
	var wrapped struct {
		Error string `json:"error"`
	}

	if err := json.Unmarshal(result, &wrapped); err == nil && wrapped.Error != "" {
		return wrapped.Error
	}

	return string(result)
}

func (as *agentStreamer) handleInvocationEnd(e agent.InvocationEndEvent) {
	_ = as.endStep()

	// Resolve any tool call the model requested but that never produced a result
	// (e.g. input-required, or a run that ended mid-tool-turn), so the client's
	// dynamic tool part transitions to output-error instead of hanging.
	as.closePendingTools()

	reason, controlText := mapAgentFinishReason(e.FinishReason)

	switch {
	case controlText != "" && !as.errored:
		// A fixed, non-sensitive control message (max-turns / input-required).
		// Emit it verbatim rather than through onError, which sanitizes to a
		// generic string — this preserves parity with the A2A path. Skipped when
		// a recoverable ErrorEvent already surfaced an error chunk: at most one
		// error chunk per stream, or the client's onError fires twice.
		_ = as.ew.WriteChunk(Chunk{"type": "error", "errorText": controlText})
	case reason == finishReasonError && !as.errored:
		// An error finish with no fixed control message and no prior error chunk
		// (a bare FinishReasonError). Emit one sanitized error so the client gets
		// error text, matching the iterator-error and incomplete-run paths. If a
		// recoverable ErrorEvent already surfaced the mapped error, don't emit a
		// second generic one — that would call the client's onError twice.
		_ = as.ew.WriteChunk(Chunk{"type": "error", "errorText": as.onError(errors.New("agent run failed"))})
	}

	finish := Chunk{"type": "finish", "finishReason": reason}
	if meta := usageMetadata(e.Usage); meta != nil {
		finish["messageMetadata"] = meta
	}

	_ = as.ew.WriteChunk(finish)
	_ = as.ew.WriteDone()
}

// mapAgentFinishReason maps an agent.FinishReason to the UI finishReason enum
// (stop|length|content-filter|tool-calls|error|other) and, for error-ish
// reasons, a fixed control message to emit as a terminal error chunk. In the UI
// Message Stream an error chunk is effectively terminal, so we emit error then
// finish{error} (in that order), never a normal finish after an error. Returns
// (uiFinishReason, controlMessage); controlMessage is "" when no error chunk.
func mapAgentFinishReason(fr agent.FinishReason) (string, string) {
	switch fr {
	case agent.FinishReasonStop, agent.FinishReasonTransfer:
		return finishReasonStop, ""
	case agent.FinishReasonMaxTurns:
		return finishReasonError, "maximum iterations reached"
	case agent.FinishReasonLength:
		// The UI stream has a dedicated "length" reason. Surface it as a normal
		// partial completion (no error chunk), matching the model-level handler,
		// rather than marking the whole message errored.
		return "length", ""
	case agent.FinishReasonInputRequired:
		// HITL is not supported here; resolve the turn visibly rather than
		// leaving a tool part spinning forever.
		return finishReasonError, "agent requested input, which is not supported over this endpoint"
	case agent.FinishReasonError:
		return finishReasonError, ""
	case agent.FinishReasonInterrupted:
		// Normally reached via the ctx-cancel abort path; if it arrives as an
		// event, treat it as a non-error terminal.
		return finishReasonOther, ""
	default:
		return finishReasonOther, ""
	}
}

// usageMetadata builds the messageMetadata payload carrying token usage, mirroring
// the usage the A2A executor attaches to its messages/status. Returns nil when no
// usage is available so the finish chunk stays minimal.
func usageMetadata(u *llm.TokenUsage) map[string]any {
	if u == nil {
		return nil
	}

	// camelCase to match the AI SDK's own LanguageModelUsage shape, so a client
	// that wires a messageMetadata schema expecting the SDK convention validates.
	return map[string]any{
		"usage": map[string]any{
			"inputTokens":       u.InputTokens,
			"outputTokens":      u.OutputTokens,
			"totalTokens":       u.TotalBilledTokens(),
			"cachedInputTokens": u.CachedInputTokens,
			"reasoningTokens":   u.ReasoningTokens,
		},
	}
}

// agentErrorEventErr converts a recoverable agent.ErrorEvent into an error so it
// can pass through the configured ErrorMapper, mirroring errorEventErr for the
// llm-level path.
func agentErrorEventErr(e agent.ErrorEvent) error {
	if e.Err != nil {
		return e.Err
	}

	return errors.New(e.Message)
}

// WritePing writes an SSE comment line and flushes. Comments (lines starting
// with ':') are ignored by the EventSource / useChat parser, so they keep the
// connection warm during a slow tool call without emitting a protocol chunk that
// could desync the stream grammar.
func (ew *EventWriter) WritePing() error {
	if _, err := io.WriteString(ew.w, ":\n\n"); err != nil {
		return err
	}

	if ew.f != nil {
		ew.f.Flush()
	}

	return nil
}
