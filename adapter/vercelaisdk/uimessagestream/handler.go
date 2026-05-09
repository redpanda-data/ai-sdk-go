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
	"fmt"
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

// ToolExecutor is called when the model requests a tool call. It receives the
// tool name and parsed JSON arguments, and returns the result as JSON bytes.
type ToolExecutor func(ctx context.Context, name string, args json.RawMessage) (json.RawMessage, error)

type config struct {
	system   string
	logger   *slog.Logger
	tools    []llm.ToolDefinition
	executor ToolExecutor
}

// WithSystem sets the system prompt prepended to every request.
func WithSystem(prompt string) Option {
	return func(c *config) { c.system = prompt }
}

// WithLogger sets the logger for the handler.
func WithLogger(l *slog.Logger) Option {
	return func(c *config) { c.logger = l }
}

// WithTools registers tool definitions and an executor for agentic tool calling.
// When the model requests a tool call, the handler will execute it via the
// executor, stream the result to the client, and feed it back to the model
// for the next turn.
func WithTools(tools []llm.ToolDefinition, executor ToolExecutor) Option {
	return func(c *config) {
		c.tools = tools
		c.executor = executor
	}
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

	req := &llm.Request{
		Messages: messages,
		Tools:    h.cfg.tools,
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	setSSEHeaders(w)

	ew := &EventWriter{w: w, f: flusher}
	StreamModelWithTools(r.Context(), h.model, req, ew, h.cfg.logger, h.cfg.executor)
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

			case llm.PartToolRequest:
					// Close text span before tool call (model often outputs text then calls tool).
					if textStarted {
						if err := ew.WriteChunk(Chunk{"type": "text-end", "id": textID}); err != nil {
							return
						}
						textStarted = false
					}
					tr := e.Part.ToolRequest
					if tr != nil {
						// tool-input-start
						if err := ew.WriteChunk(Chunk{
							"type":       "tool-input-start",
							"toolCallId": tr.ID,
							"toolName":   tr.Name,
						}); err != nil {
							return
						}
						// tool-input-available (Go SDK gives complete args, not streaming deltas)
						var input any
						if len(tr.Arguments) > 0 {
							_ = json.Unmarshal(tr.Arguments, &input)
						}
						if err := ew.WriteChunk(Chunk{
							"type":       "tool-input-available",
							"toolCallId": tr.ID,
							"toolName":   tr.Name,
							"input":      input,
						}); err != nil {
							return
						}
					}

				case llm.PartToolResponse:
					tr := e.Part.ToolResponse
					if tr != nil {
						var output any
						if len(tr.Result) > 0 {
							_ = json.Unmarshal(tr.Result, &output)
						}
						if tr.Error != "" {
							_ = ew.WriteChunk(Chunk{
								"type":       "tool-output-error",
								"toolCallId": tr.ID,
								"errorText":  tr.Error,
							})
						} else {
							if err := ew.WriteChunk(Chunk{
								"type":       "tool-output-available",
								"toolCallId": tr.ID,
								"output":     output,
							}); err != nil {
								return
							}
						}
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

// StreamModelWithTools is like StreamModel but supports agentic tool calling.
// When the model returns tool calls, the executor is invoked for each, results
// are streamed to the client, and the model is called again with the results
// appended to the conversation. This loops until the model stops calling tools.
func StreamModelWithTools(ctx context.Context, model llm.Model, req *llm.Request, ew *EventWriter, logger *slog.Logger, executor ToolExecutor) {
	if executor == nil {
		StreamModel(ctx, model, req, ew, logger)
		return
	}
	if logger == nil {
		logger = slog.Default()
	}

	messageID := generateMessageID()
	if err := ew.WriteChunk(Chunk{"type": "start", "messageId": messageID}); err != nil {
		return
	}

	messages := make([]llm.Message, len(req.Messages))
	copy(messages, req.Messages)

	textCounter := 0
	const maxTurns = 10

	for turn := 0; turn < maxTurns; turn++ {
		if err := ew.WriteChunk(Chunk{"type": "start-step"}); err != nil {
			return
		}

		textID := fmt.Sprintf("text-%d", textCounter)
		textStarted := false
		var toolRequests []*llm.ToolRequest

		iterReq := &llm.Request{
			Messages:   messages,
			Tools:      req.Tools,
			ToolChoice: req.ToolChoice,
		}

		var finishReason string

		for event, err := range model.GenerateEvents(ctx, iterReq) {
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				logger.Error("stream error", "error", err)
				_ = ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"})
				_ = ew.WriteChunk(Chunk{"type": "finish-step"})
				_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": "error"})
				ew.WriteDone()
				return
			}

			switch e := event.(type) {
			case llm.ContentPartEvent:
				switch e.Part.Kind {
				case llm.PartText:
					if !textStarted {
						if err := ew.WriteChunk(Chunk{"type": "text-start", "id": textID}); err != nil {
							return
						}
						textStarted = true
					}
					if err := ew.WriteChunk(Chunk{"type": "text-delta", "id": textID, "delta": e.Part.Text}); err != nil {
						return
					}

				case llm.PartToolRequest:
					if textStarted {
						if err := ew.WriteChunk(Chunk{"type": "text-end", "id": textID}); err != nil {
							return
						}
						textStarted = false
						textCounter++
						textID = fmt.Sprintf("text-%d", textCounter)
					}
					tr := e.Part.ToolRequest
					if tr != nil {
						toolRequests = append(toolRequests, tr)
						if err := ew.WriteChunk(Chunk{
							"type": "tool-input-start", "toolCallId": tr.ID, "toolName": tr.Name,
						}); err != nil {
							return
						}
						var input any
						if len(tr.Arguments) > 0 {
							_ = json.Unmarshal(tr.Arguments, &input)
						}
						if err := ew.WriteChunk(Chunk{
							"type": "tool-input-available", "toolCallId": tr.ID, "toolName": tr.Name, "input": input,
						}); err != nil {
							return
						}
					}
				}

			case llm.StreamEndEvent:
				if e.Error != nil {
					logger.Error("LLM error", "error", e.Error)
					_ = ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"})
				}
				if textStarted {
					_ = ew.WriteChunk(Chunk{"type": "text-end", "id": textID})
					textStarted = false
					textCounter++
				}
				_ = ew.WriteChunk(Chunk{"type": "finish-step"})

				finishReason = "stop"
				if e.Error != nil {
					finishReason = "error"
				} else if e.Response != nil {
					finishReason = mapFinishReason(e.Response.FinishReason)
				}
			}
		}

		if len(toolRequests) == 0 || finishReason != "tool-calls" {
			_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": finishReason})
			ew.WriteDone()
			return
		}

		// Execute tools and append results to messages.
		assistantParts := make([]*llm.Part, 0, len(toolRequests))
		for _, tr := range toolRequests {
			assistantParts = append(assistantParts, llm.NewToolRequestPart(tr))
		}
		messages = append(messages, llm.Message{Role: llm.RoleAssistant, Content: assistantParts})

		toolResponseParts := make([]*llm.Part, 0, len(toolRequests))
		for _, tr := range toolRequests {
			result, err := executor(ctx, tr.Name, tr.Arguments)
			if err != nil {
				_ = ew.WriteChunk(Chunk{
					"type": "tool-output-error", "toolCallId": tr.ID, "errorText": err.Error(),
				})
				toolResponseParts = append(toolResponseParts, llm.NewToolResponsePart(&llm.ToolResponse{
					ID: tr.ID, Name: tr.Name, Error: err.Error(),
				}))
			} else {
				var output any
				if len(result) > 0 {
					_ = json.Unmarshal(result, &output)
				}
				if err := ew.WriteChunk(Chunk{
					"type": "tool-output-available", "toolCallId": tr.ID, "output": output,
				}); err != nil {
					return
				}
				toolResponseParts = append(toolResponseParts, llm.NewToolResponsePart(&llm.ToolResponse{
					ID: tr.ID, Name: tr.Name, Result: result,
				}))
			}
		}
		messages = append(messages, llm.Message{Role: llm.RoleUser, Content: toolResponseParts})
	}

	_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": "other"})
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
