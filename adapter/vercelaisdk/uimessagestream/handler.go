// Package uimessagestream provides an HTTP handler that implements the Vercel
// AI SDK UI Message Stream protocol (v1). This allows any ai-sdk-go llm.Model
// to serve responses compatible with the useChat hook from "@ai-sdk/react".
//
// Wire format: Server-Sent Events (SSE) with JSON chunks.
// Each event is written as: "data: <json>\n\n"
// Stream terminates with: "data: [DONE]\n\n"
//
// The chunk sequence for a simple text response is:
//
//	start → start-step → text-start → text-delta* → text-end → finish-step → finish → [DONE]
//
// This is a 1:1 port of the Vercel AI SDK's toUIMessageStream() from
// packages/ai/src/generate-text/stream-text.ts and the SSE framing from
// packages/ai/src/ui-message-stream/json-to-sse-transform-stream.ts.
//
// Reference: https://github.com/vercel/ai
package uimessagestream

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Handler returns an http.Handler that serves the AI SDK UI Message Stream
// protocol. It accepts POST requests with a JSON body containing messages
// and streams back SSE events compatible with useChat.
func Handler(model llm.Model, opts ...Option) http.Handler {
	cfg := &config{logger: slog.Default()}
	for _, o := range opts {
		o(cfg)
	}
	return &handler{model: model, cfg: cfg}
}

// Option configures the handler.
type Option func(*config)

type config struct {
	system string
	logger *slog.Logger
}

// WithSystem sets the system prompt prepended to every request.
func WithSystem(prompt string) Option {
	return func(c *config) { c.system = prompt }
}

// WithLogger sets the logger for the handler.
func WithLogger(l *slog.Logger) Option {
	return func(c *config) { c.logger = l }
}

type handler struct {
	model llm.Model
	cfg   *config
}

// chatRequest matches the JSON body sent by useChat.
type chatRequest struct {
	ID       string        `json:"id"`
	Messages []chatMessage `json:"messages"`
	Trigger  string        `json:"trigger"`
}

type chatMessage struct {
	Role    string        `json:"role"`
	Content string        `json:"content"`
	Parts   []messagePart `json:"parts"`
}

type messagePart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// textContent extracts the text content from a message, supporting both
// the v6 parts-based format and the legacy content field.
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

func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Limit request body to 1MB to prevent abuse.
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var body chatRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	messages := convertMessages(body.Messages, h.cfg.system)

	// Reject requests with no meaningful messages.
	if len(messages) == 0 {
		http.Error(w, "empty messages", http.StatusBadRequest)
		return
	}

	req := &llm.Request{Messages: messages}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	setSSEHeaders(w)

	ew := &EventWriter{w: w, f: flusher}
	StreamModel(r.Context(), h.model, req, ew, h.cfg.logger)
}

// generateMessageID creates a random 16-character hex ID for use as a messageId.
func generateMessageID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		// Fallback — should never happen in practice.
		return "msg-0000000000000000"
	}
	return hex.EncodeToString(b)
}

// StreamModel streams responses from an llm.Model as AI SDK UI Message Stream
// events. This is the core streaming logic, usable from custom HTTP handlers.
func StreamModel(ctx context.Context, model llm.Model, req *llm.Request, ew *EventWriter, logger *slog.Logger) {
	if logger == nil {
		logger = slog.Default()
	}

	messageID := generateMessageID()

	// start
	if err := ew.WriteChunk(Chunk{"type": "start", "messageId": messageID}); err != nil {
		return
	}

	// start-step
	if err := ew.WriteChunk(Chunk{"type": "start-step"}); err != nil {
		return
	}

	textID := "text-0"
	textStarted := false

	reasoningID := "reasoning-0"
	reasoningStarted := false

	for event, err := range model.GenerateEvents(ctx, req) {
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			logger.Error("stream error", "error", err)
			_ = ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"})
			_ = ew.WriteChunk(Chunk{"type": "finish-step"})
			_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": "error"})
			break
		}

		switch e := event.(type) {
		case llm.ContentPartEvent:
			switch e.Part.Kind {
			case llm.PartText:
				// Close reasoning span before starting text.
				if reasoningStarted {
					if err := ew.WriteChunk(Chunk{"type": "reasoning-end", "id": reasoningID}); err != nil {
						return
					}
					reasoningStarted = false
				}
				if !textStarted {
					if err := ew.WriteChunk(Chunk{"type": "text-start", "id": textID}); err != nil {
						return
					}
					textStarted = true
				}
				if err := ew.WriteChunk(Chunk{"type": "text-delta", "id": textID, "delta": e.Part.Text}); err != nil {
					return
				}

			case llm.PartReasoning:
				// Forward reasoning traces with stateful tracking.
				if !reasoningStarted {
					if err := ew.WriteChunk(Chunk{"type": "reasoning-start", "id": reasoningID}); err != nil {
						return
					}
					reasoningStarted = true
				}
				if e.Part.ReasoningTrace != nil {
					if err := ew.WriteChunk(Chunk{"type": "reasoning-delta", "id": reasoningID, "delta": e.Part.ReasoningTrace.Text}); err != nil {
						return
					}
				}
			}

		case llm.ErrorEvent:
			// ErrorEvent is non-terminal (stream continues). Log and sanitize.
			logger.Warn("recoverable LLM error", "message", e.Message)
			if err := ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"}); err != nil {
				return
			}

		case llm.StreamResetEvent:
			// Reset accumulated state. Close open text/reasoning spans first.
			if textStarted {
				if err := ew.WriteChunk(Chunk{"type": "text-end", "id": textID}); err != nil {
					return
				}
				textStarted = false
			}
			if reasoningStarted {
				if err := ew.WriteChunk(Chunk{"type": "reasoning-end", "id": reasoningID}); err != nil {
					return
				}
				reasoningStarted = false
			}

		case llm.StreamEndEvent:
			hasError := e.Error != nil
			if hasError {
				logger.Error("LLM error", "error", e.Error)
				_ = ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"})
			}

			if reasoningStarted {
				_ = ew.WriteChunk(Chunk{"type": "reasoning-end", "id": reasoningID})
			}

			if textStarted {
				_ = ew.WriteChunk(Chunk{"type": "text-end", "id": textID})
			}

			_ = ew.WriteChunk(Chunk{"type": "finish-step"})

			reason := "stop"
			if hasError {
				reason = "error"
			} else if e.Response != nil {
				reason = mapFinishReason(e.Response.FinishReason)
			}
			_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": reason})
		}
	}

	ew.WriteDone()
}

func convertMessages(msgs []chatMessage, system string) []llm.Message {
	out := make([]llm.Message, 0, len(msgs)+1)
	if system != "" {
		out = append(out, llm.NewMessage(llm.RoleSystem, llm.NewTextPart(system)))
	}
	for _, m := range msgs {
		text := m.textContent()
		// Skip messages with no text content. This handles v6 step-start parts
		// and other non-text-only messages that don't carry user/assistant text.
		if text == "" {
			continue
		}
		role := llm.RoleUser
		switch m.Role {
		case "assistant":
			role = llm.RoleAssistant
		case "system":
			role = llm.RoleSystem
		}
		out = append(out, llm.NewMessage(role, llm.NewTextPart(text)))
	}
	return out
}

// setSSEHeaders sets the required HTTP headers for the AI SDK UI Message
// Stream protocol. These match the headers from the Vercel AI SDK:
// packages/ai/src/ui-message-stream/ui-message-stream-headers.ts
func setSSEHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Vercel-AI-UI-Message-Stream", "v1")
	w.Header().Set("X-Accel-Buffering", "no")
}

func mapFinishReason(fr llm.FinishReason) string {
	switch fr {
	case llm.FinishReasonStop:
		return "stop"
	case llm.FinishReasonLength:
		return "length"
	case llm.FinishReasonContentFilter:
		return "content-filter"
	case llm.FinishReasonToolCalls:
		return "tool-calls"
	default:
		return "other"
	}
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
// Returns an error if marshaling or writing fails.
func (ew *EventWriter) WriteChunk(c Chunk) error {
	data, err := json.Marshal(c)
	if err != nil {
		return err
	}
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
