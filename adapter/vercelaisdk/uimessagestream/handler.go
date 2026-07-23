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

// Package uimessagestream implements the server half of the Vercel AI SDK UI
// Message Stream protocol (v1) — the wire format the useChat hook from
// "@ai-sdk/react" speaks. It exposes an ai-sdk-go agent over that protocol via
// Handler (see agent.go), with server-side sessions persisted in a
// session.Store.
//
// History is server-authoritative, following the Vercel AI SDK's canonical
// persistence pattern (https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-message-persistence):
// the server loads the chat by id, appends the one posted user message, runs
// the agent, and saves — the client sends only {id, message} via
// prepareSendMessagesRequest (the default full-body transport also works; only
// the last message is used and the posted history is ignored). Sessions store
// llm.Message — the SDK's canonical conversation shape, shared with the A2A
// adapter and the runner — and are projected to UI messages on read; this
// deliberately diverges from the AI SDK's persist-UIMessages advice, at the
// cost that UI message ids and custom data parts do not round-trip.
//
// Wire format: Server-Sent Events (SSE) with JSON chunks. Each event is written
// as "data: <json>\n\n"; the stream terminates with "data: [DONE]\n\n". The
// chunk sequence for a simple text response is:
//
//	start → start-step → text-start → text-delta* → text-end → finish-step → finish → [DONE]
//
// Per-part chunk shapes mirror the Vercel AI SDK's
// packages/ai/src/ui-message-stream/to-ui-message-chunk.ts, and the SSE framing
// mirrors json-to-sse-transform-stream.ts. Verified against ai@7.0.6.
//
// This file holds the protocol primitives (request decoding, the SSE
// EventWriter, and the streamWriter span bookkeeping) shared by Handler.
//
// Migration: the former model-level Handler, StreamModel,
// StreamModelWithTools, ToolExecutor, WithSystem, WithTools, and WithMaxTurns
// APIs were removed. Callers must construct an agent.Agent and use Handler with
// a session.Store instead. This keeps the system prompt, tools, interceptor
// chain, and agentic loop at the agent layer rather than duplicating them in a
// wire-protocol adapter.
//
// Known limitations:
//   - Inbound file/image parts are not forwarded: the llm.Part type has no file
//     kind yet. Inbound reasoning parts are likewise dropped.
//   - Editing a historical message is not supported (sessions do not persist
//     UI message ids); regenerate covers the retry-last-answer flow.
//
// Reference: https://github.com/vercel/ai
package uimessagestream

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Option configures a handler.
type Option func(*config)

// ErrorMapper maps an error to the client-facing text emitted in "error" and
// "tool-output-error" chunks. It mirrors the Vercel AI SDK's onError option
// (toUIMessageStream / toUIMessageChunk). The default returns a sanitized,
// generic message to avoid leaking server-side error details to the client.
type ErrorMapper func(error) string

// defaultErrorText matches the Vercel AI SDK default onError return value
// (packages/ai/src/ui-message-stream/to-ui-message-chunk.ts: () => 'An error occurred.').
const defaultErrorText = "An error occurred."

func defaultErrorMapper(error) string { return defaultErrorText }

// SessionKeyFunc derives the storage key for a client-supplied chat id. It is
// the tenant-isolation seam: a multi-tenant deployment derives the key from the
// authenticated request (e.g. "userID/chatID") so one user cannot address
// another's chat. Returning an error rejects the request with 403.
type SessionKeyFunc func(r *http.Request, chatID string) (string, error)

type config struct {
	logger       *slog.Logger
	maxBodyBytes int64
	onError      ErrorMapper
	sessionKey   SessionKeyFunc
}

// WithLogger sets the logger for the handler.
func WithLogger(l *slog.Logger) Option {
	return func(c *config) { c.logger = l }
}

// WithMaxBodyBytes sets the maximum request body size in bytes. Default is 1MB.
func WithMaxBodyBytes(n int64) Option {
	return func(c *config) { c.maxBodyBytes = n }
}

// WithOnError sets the mapper that produces the client-facing error text for
// "error" and "tool-output-error" chunks. This mirrors the Vercel AI SDK's
// onError option. By default a sanitized, generic message ("An error occurred.")
// is sent to avoid leaking server-side error details. Provide a custom mapper
// to surface specific error information to the client. A nil mapper is ignored.
func WithOnError(fn ErrorMapper) Option {
	return func(c *config) {
		if fn != nil {
			c.onError = fn
		}
	}
}

// WithSessionKey sets the function deriving the session storage key from the
// request and the client-supplied chat id. Without it the chat id is used
// verbatim, which is only safe when every caller is trusted with every chat
// (single-tenant / dev). Configuring a custom key disables the list endpoint:
// session.Store.List enumerates all storage keys and cannot be tenant-scoped
// here — expose your own list API instead. A nil fn is ignored.
func WithSessionKey(fn SessionKeyFunc) Option {
	return func(c *config) {
		if fn != nil {
			c.sessionKey = fn
		}
	}
}

// Chat lifecycle triggers sent by useChat's DefaultChatTransport.
const (
	triggerSubmit        = "submit-message"
	triggerRegenerate    = "regenerate-message"
	toolStateOutputError = "output-error"
)

// chatRequest matches the JSON body sent by useChat's DefaultChatTransport.
// The canonical server-session client trims the body to {id, trigger,
// messageId, message} via prepareSendMessagesRequest; the default transport
// sends the full Messages list instead, of which only the last entry is used —
// server-side history is authoritative.
type chatRequest struct {
	ID        string        `json:"id"`
	Trigger   string        `json:"trigger"`
	MessageID string        `json:"messageId"` //nolint:tagliatelle // Vercel AI SDK wire format is camelCase
	Message   *chatMessage  `json:"message"`
	Messages  []chatMessage `json:"messages"`
}

type chatMessage struct {
	Role    string        `json:"role"`
	Content string        `json:"content"`
	Parts   []messagePart `json:"parts"`
}

// messagePart is the wire shape of a UI message part, both inbound (decoding
// useChat requests) and outbound (the GET-history projection). omitempty keeps
// projected parts minimal — a text part must not carry "toolCallId":"".
type messagePart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`

	// Tool-call part fields. Tool parts carry a Type of "tool-<toolName>"
	// (static tools) or "dynamic-tool" (with ToolName set), mirroring the v7 UI
	// message ToolUIPart / DynamicToolUIPart shapes. The JSON tags are the
	// camelCase names the Vercel AI SDK useChat client sends on the wire.
	ToolCallID string          `json:"toolCallId,omitempty"` //nolint:tagliatelle // Vercel AI SDK wire format is camelCase
	ToolName   string          `json:"toolName,omitempty"`   //nolint:tagliatelle // Vercel AI SDK wire format is camelCase
	State      string          `json:"state,omitempty"`
	Input      json.RawMessage `json:"input,omitempty"`
	Output     json.RawMessage `json:"output,omitempty"`
	ErrorText  string          `json:"errorText,omitempty"` //nolint:tagliatelle // Vercel AI SDK wire format is camelCase
}

// textContent extracts the text content from a message, supporting both
// the v7 parts-based format and the legacy content field.
// All text parts are concatenated to preserve multi-step conversation history.
func (m chatMessage) textContent() string {
	var parts []string

	for _, p := range m.Parts {
		if p.Type == "text" {
			parts = append(parts, p.Text)
		}
	}

	if len(parts) > 0 {
		return strings.Join(parts, "")
	}

	return m.Content
}

// convertUserMessage converts the posted UI message into the user llm.Message
// appended to the session. Only genuine user text is accepted: the client must
// not be able to append assistant or system turns, and tool results only ever
// originate server-side.
func convertUserMessage(m chatMessage) (llm.Message, bool) {
	if m.Role == "assistant" || m.Role == "system" {
		return llm.Message{}, false
	}

	text := m.textContent()
	if text == "" {
		return llm.Message{}, false
	}

	return llm.NewMessage(llm.RoleUser, llm.NewTextPart(text)), true
}

// generateMessageID creates a random 16-character hex ID for use as a messageId.
func generateMessageID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "0000000000000000"
	}

	return hex.EncodeToString(b)
}

const (
	finishReasonStop  = "stop"
	finishReasonError = "error"
	finishReasonOther = "other"
)

// streamWriter manages SSE event state (open text/reasoning spans) for a single
// streaming response. Extracting this keeps the per-event handlers small.
type streamWriter struct {
	ew               *EventWriter
	logger           *slog.Logger
	onError          ErrorMapper
	textID           string
	reasoningID      string
	textStarted      bool
	reasoningStarted bool
	textCounter      int
	reasoningCounter int
}

func newStreamWriter(ew *EventWriter, logger *slog.Logger, onError ErrorMapper) *streamWriter {
	if onError == nil {
		onError = defaultErrorMapper
	}

	return &streamWriter{
		ew:          ew,
		logger:      logger,
		onError:     onError,
		textID:      "text-0",
		reasoningID: "reasoning-0",
	}
}

func (sw *streamWriter) endReasoning() error {
	if !sw.reasoningStarted {
		return nil
	}

	if err := sw.ew.WriteChunk(Chunk{"type": "reasoning-end", "id": sw.reasoningID}); err != nil {
		return err
	}

	sw.reasoningStarted = false
	sw.reasoningCounter++
	sw.reasoningID = fmt.Sprintf("reasoning-%d", sw.reasoningCounter)

	return nil
}

func (sw *streamWriter) endTextAndAdvance() error {
	if !sw.textStarted {
		return nil
	}

	if err := sw.ew.WriteChunk(Chunk{"type": "text-end", "id": sw.textID}); err != nil {
		return err
	}

	sw.textStarted = false
	sw.textCounter++
	sw.textID = fmt.Sprintf("text-%d", sw.textCounter)

	return nil
}

func (sw *streamWriter) writeTextDelta(text string) error {
	if err := sw.endReasoning(); err != nil {
		return err
	}

	if !sw.textStarted {
		if err := sw.ew.WriteChunk(Chunk{"type": "text-start", "id": sw.textID}); err != nil {
			return err
		}

		sw.textStarted = true
	}

	return sw.ew.WriteChunk(Chunk{"type": "text-delta", "id": sw.textID, "delta": text})
}

func (sw *streamWriter) writeReasoningDelta(reasoning *llm.ReasoningPart) error {
	if err := sw.endTextAndAdvance(); err != nil {
		return err
	}

	if !sw.reasoningStarted {
		if err := sw.ew.WriteChunk(Chunk{"type": "reasoning-start", "id": sw.reasoningID}); err != nil {
			return err
		}

		sw.reasoningStarted = true
	}

	if reasoning == nil {
		return nil
	}

	return sw.ew.WriteChunk(Chunk{"type": "reasoning-delta", "id": sw.reasoningID, "delta": reasoning.Text})
}

// closeSpans closes any open reasoning/text spans, ignoring write errors.
func (sw *streamWriter) closeSpans() {
	if sw.reasoningStarted {
		_ = sw.ew.WriteChunk(Chunk{"type": "reasoning-end", "id": sw.reasoningID})
		sw.reasoningStarted = false
		sw.reasoningCounter++
		sw.reasoningID = fmt.Sprintf("reasoning-%d", sw.reasoningCounter)
	}

	if sw.textStarted {
		_ = sw.ew.WriteChunk(Chunk{"type": "text-end", "id": sw.textID})
		sw.textStarted = false
		sw.textCounter++
		sw.textID = fmt.Sprintf("text-%d", sw.textCounter)
	}
}

// writeAbort emits an "abort" chunk (with an optional reason) followed by the
// terminal [DONE], mirroring stream-text.ts abort handling. Used when the
// request context is cancelled. Write errors are ignored: a cancelled context
// usually means the client has already disconnected.
func (sw *streamWriter) writeAbort(ctx context.Context) {
	chunk := Chunk{"type": "abort"}
	if cause := context.Cause(ctx); cause != nil && !errors.Is(cause, context.Canceled) {
		chunk["reason"] = cause.Error()
	}

	_ = sw.ew.WriteChunk(chunk)
	_ = sw.ew.WriteDone()
}

// setSSEHeaders sets the required HTTP headers for the AI SDK UI Message
// Stream protocol. These match the headers from the Vercel AI SDK:
// packages/ai/src/ui-message-stream/ui-message-stream-headers.ts.
func setSSEHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Vercel-Ai-Ui-Message-Stream", "v1")
	w.Header().Set("X-Accel-Buffering", "no")
}

// Chunk is a JSON-serializable SSE event payload.
type Chunk map[string]any

// EventWriter writes AI SDK SSE events to an http.ResponseWriter.
type EventWriter struct {
	w http.ResponseWriter
	f http.Flusher
}

// NewEventWriter creates an EventWriter from an http.ResponseWriter.
func NewEventWriter(w http.ResponseWriter) *EventWriter {
	flusher, _ := w.(http.Flusher)
	return &EventWriter{w: w, f: flusher}
}

// WriteChunk writes a single SSE data event and flushes.
func (ew *EventWriter) WriteChunk(c Chunk) error {
	// Match the reference SSE framing byte-for-byte: JSON.stringify does not
	// HTML-escape <, >, or &, but Go's json.Marshal does. Use an Encoder with
	// HTML escaping disabled so deltas containing markup/code are sent verbatim.
	var buf bytes.Buffer

	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)

	if err := enc.Encode(c); err != nil {
		return err
	}

	// Encoder.Encode appends a trailing newline; the SSE framing adds its own.
	data := bytes.TrimRight(buf.Bytes(), "\n")

	if _, err := io.WriteString(ew.w, "data: "); err != nil {
		return err
	}

	if _, err := ew.w.Write(data); err != nil {
		return err
	}

	if _, err := io.WriteString(ew.w, "\n\n"); err != nil {
		return err
	}

	if ew.f != nil {
		ew.f.Flush()
	}

	return nil
}

// WriteDone writes the terminal [DONE] event and flushes.
func (ew *EventWriter) WriteDone() error {
	if _, err := io.WriteString(ew.w, "data: [DONE]\n\n"); err != nil {
		return err
	}

	if ew.f != nil {
		ew.f.Flush()
	}

	return nil
}
