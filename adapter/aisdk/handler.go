// Package aisdk provides an HTTP handler that implements the Vercel AI SDK
// UI Message Stream protocol (v1). This allows any ai-sdk-go llm.Model to
// serve responses compatible with the useChat hook from the "@ai-sdk/react"
// npm package.
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
package aisdk

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"

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
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var body chatRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	messages := convertMessages(body.Messages, h.cfg.system)
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

// StreamModel streams responses from an llm.Model as AI SDK UI Message Stream
// events. This is the core streaming logic, usable from custom HTTP handlers.
func StreamModel(ctx context.Context, model llm.Model, req *llm.Request, ew *EventWriter, logger *slog.Logger) {
	if logger == nil {
		logger = slog.Default()
	}

	// start
	ew.WriteChunk(Chunk{"type": "start"})

	// start-step
	ew.WriteChunk(Chunk{"type": "start-step"})

	textID := "text-0"
	textStarted := false

	for event, err := range model.GenerateEvents(ctx, req) {
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			logger.Error("stream error", "error", err)
			ew.WriteChunk(Chunk{"type": "error", "errorText": err.Error()})
			break
		}

		switch e := event.(type) {
		case llm.ContentPartEvent:
			switch e.Part.Kind {
			case llm.PartText:
				if !textStarted {
					ew.WriteChunk(Chunk{"type": "text-start", "id": textID})
					textStarted = true
				}
				ew.WriteChunk(Chunk{"type": "text-delta", "id": textID, "delta": e.Part.Text})

			case llm.PartReasoning:
				// Forward reasoning traces
				reasoningID := "reasoning-0"
				ew.WriteChunk(Chunk{"type": "reasoning-start", "id": reasoningID})
				if e.Part.ReasoningTrace != nil {
					ew.WriteChunk(Chunk{"type": "reasoning-delta", "id": reasoningID, "delta": e.Part.ReasoningTrace.Text})
				}
				ew.WriteChunk(Chunk{"type": "reasoning-end", "id": reasoningID})
			}

		case llm.ErrorEvent:
			ew.WriteChunk(Chunk{"type": "error", "errorText": e.Message})

		case llm.StreamEndEvent:
			if e.Error != nil {
				logger.Error("LLM error", "error", e.Error)
				ew.WriteChunk(Chunk{"type": "error", "errorText": e.Error.Error()})
			}

			if textStarted {
				ew.WriteChunk(Chunk{"type": "text-end", "id": textID})
			}

			// finish-step
			ew.WriteChunk(Chunk{"type": "finish-step"})

			// finish
			reason := "stop"
			if e.Response != nil {
				reason = mapFinishReason(e.Response.FinishReason)
			}
			ew.WriteChunk(Chunk{"type": "finish", "finishReason": reason})
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
		role := llm.RoleUser
		switch m.Role {
		case "assistant":
			role = llm.RoleAssistant
		case "system":
			role = llm.RoleSystem
		}
		out = append(out, llm.NewMessage(role, llm.NewTextPart(m.Content)))
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
func (ew *EventWriter) WriteChunk(c Chunk) {
	data, err := json.Marshal(c)
	if err != nil {
		return
	}
	fmt.Fprintf(ew.w, "data: %s\n\n", data)
	if ew.f != nil {
		ew.f.Flush()
	}
}

// WriteDone writes the terminal [DONE] event and flushes.
func (ew *EventWriter) WriteDone() {
	fmt.Fprint(ew.w, "data: [DONE]\n\n")
	if ew.f != nil {
		ew.f.Flush()
	}
}
