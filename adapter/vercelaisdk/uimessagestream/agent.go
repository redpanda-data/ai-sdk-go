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

// AgentHandler serves the Vercel AI SDK UI Message Stream protocol backed by a
// full agent.Agent — its system prompt, tool registry (MCP + subagents-as-tools),
// interceptor chain (OTel/transcripts), and agentic tool-calling loop.
//
// It is a pure protocol translator: it runs no loop of its own, converting the
// inbound UI messages into an agent invocation and mapping the agent's event
// stream onto UI Message Stream chunks. The agent package owns all business
// logic. This is the mirror of adapter/a2a's Executor, one protocol lower.
//
// History is client-authoritative: useChat re-sends the full message list every
// turn, so each request rebuilds the whole conversation from the posted messages
// and runs it against a fresh, non-persisted session. There is deliberately no
// server-side session keyed on the client-supplied chat id (that would be a
// tenant-isolation hazard and would fight useChat's regenerate/edit). Transcript
// and span grouping come from the agent's interceptor chain plus the conversation
// id the caller forwards out-of-band, not from a shared session store — which is
// why this takes an agent.Agent and not a *runner.Runner: the runner's
// load-session-by-id-and-append-one-message shape cannot express full-history
// per request, and its persistence is exactly what client-authoritative avoids.
func AgentHandler(ag agent.Agent, opts ...Option) http.Handler {
	cfg := &config{
		logger:       slog.Default(),
		maxBodyBytes: 1 << 20, // 1MB
		onError:      defaultErrorMapper,
	}
	for _, o := range opts {
		o(cfg)
	}

	return &agentHTTPHandler{agent: ag, cfg: cfg}
}

type agentHTTPHandler struct {
	agent agent.Agent
	cfg   *config
}

func (h *agentHTTPHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.cfg.maxBodyBytes)

	var body chatRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	// Client-authoritative: rebuild the full conversation from the posted
	// messages. No system prompt is injected here — the agent supplies its own.
	messages := convertMessages(body.Messages)
	if len(messages) == 0 {
		http.Error(w, "empty messages", http.StatusBadRequest)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// Fresh, non-persisted session seeded with the whole conversation. The id is
	// cosmetic (telemetry only) since nothing is saved; prefer the client's chat
	// id when present for trace correlation.
	sessionID := body.ID
	if sessionID == "" {
		sessionID = generateMessageID()
	}

	sess := &session.State{
		ID:       sessionID,
		Messages: messages,
		Metadata: make(map[string]any),
	}
	inv := agent.NewInvocationMetadata(sess, h.agent.Info())

	setSSEHeaders(w)

	ew := &EventWriter{w: w, f: flusher}
	StreamAgent(r.Context(), h.agent, inv, ew, h.cfg.logger, h.cfg.onError)
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

	return as.ew.WriteChunk(Chunk{
		"type": "tool-input-available", "toolCallId": tr.ID, "toolName": tr.Name, "input": input, "dynamic": true,
	})
}

func (as *agentStreamer) writeDynamicToolResponse(tr *llm.ToolResponsePart) error {
	if tr == nil {
		return nil
	}

	if tr.IsError {
		// The agent/tool registry stores the raw err.Error() in Result. Route it
		// through onError before it reaches the browser, so failed MCP/subagent
		// tools cannot leak server-side detail — matching the sanitization the
		// model-level executor path applies to client-facing tool errors.
		return as.ew.WriteChunk(Chunk{
			"type": "tool-output-error", "toolCallId": tr.ID,
			"errorText": as.onError(errors.New(string(tr.Result))), "dynamic": true,
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
	}

	for event, err := range ag.Run(ctx, inv) {
		if err != nil {
			// Cancellation: emit an abort (best-effort — the client may be gone).
			if ctx.Err() != nil {
				as.sw.writeAbort(ctx)
				return
			}

			logger.Error("agent run error", "error", err)

			_ = as.endStep()
			_ = ew.WriteChunk(Chunk{"type": "error", "errorText": onError(err)})
			_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": finishReasonError})
			_ = ew.WriteDone()

			return
		}

		if done := as.handleEvent(event, logger); done {
			return
		}
	}

	// The stream ended without an InvocationEndEvent. Terminate cleanly rather
	// than leaving the browser hanging on an open stream.
	logger.Warn("agent stream ended without InvocationEndEvent")

	_ = as.endStep()
	_ = ew.WriteChunk(Chunk{"type": "error", "errorText": onError(errors.New("incomplete agent run"))})
	_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": finishReasonError})
	_ = ew.WriteDone()
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

	case agent.InvocationEndEvent:
		as.handleInvocationEnd(e)
		return true

	default:
		// ToolRequestEvent (parts ride MessageEvent) and any future events.
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
			_ = as.sw.endReasoning()
			_ = as.sw.endTextAndAdvance()
			_ = as.writeDynamicToolRequest(p)
		}
	}

	_ = as.endStep()
}

func (as *agentStreamer) handleInvocationEnd(e agent.InvocationEndEvent) {
	_ = as.endStep()

	reason, controlText := mapAgentFinishReason(e.FinishReason)

	switch {
	case controlText != "":
		// A fixed, non-sensitive control message (max-turns / input-required).
		// Emit it verbatim rather than through onError, which sanitizes to a
		// generic string — this preserves parity with the A2A path.
		_ = as.ew.WriteChunk(Chunk{"type": "error", "errorText": controlText})
	case reason == finishReasonError:
		// An error finish with no fixed control message (FinishReasonError). Emit
		// a sanitized error chunk so the client always gets error text on an
		// errored finish, matching the iterator-error and incomplete-run paths.
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
