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
	"slices"
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
		return "msg-0000000000000000"
	}

	return hex.EncodeToString(b)
}

const (
	finishReasonStop  = "stop"
	finishReasonError = "error"
	finishReasonOther = "other"
)

// streamWriter manages SSE event state (open text/reasoning spans) for a
// single streaming response. Extracting this reduces the cyclomatic
// complexity of StreamModel and StreamModelWithTools.
type streamWriter struct {
	ew               *EventWriter
	textID           string
	reasoningID      string
	textStarted      bool
	reasoningStarted bool
	textCounter      int
}

func newStreamWriter(ew *EventWriter) *streamWriter {
	return &streamWriter{
		ew:          ew,
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

	return nil
}

func (sw *streamWriter) endText() error {
	if !sw.textStarted {
		return nil
	}

	if err := sw.ew.WriteChunk(Chunk{"type": "text-end", "id": sw.textID}); err != nil {
		return err
	}

	sw.textStarted = false

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

func (sw *streamWriter) writeReasoningDelta(trace *llm.ReasoningTrace) error {
	if !sw.reasoningStarted {
		if err := sw.ew.WriteChunk(Chunk{"type": "reasoning-start", "id": sw.reasoningID}); err != nil {
			return err
		}

		sw.reasoningStarted = true
	}

	if trace == nil {
		return nil
	}

	return sw.ew.WriteChunk(Chunk{"type": "reasoning-delta", "id": sw.reasoningID, "delta": trace.Text})
}

func (sw *streamWriter) writeToolRequest(tr *llm.ToolRequest) error {
	if tr == nil {
		return nil
	}

	if err := sw.ew.WriteChunk(Chunk{
		"type": "tool-input-start", "toolCallId": tr.ID, "toolName": tr.Name,
	}); err != nil {
		return err
	}

	var input any
	if len(tr.Arguments) > 0 {
		_ = json.Unmarshal(tr.Arguments, &input)
	}

	return sw.ew.WriteChunk(Chunk{
		"type": "tool-input-available", "toolCallId": tr.ID, "toolName": tr.Name, "input": input,
	})
}

func (sw *streamWriter) writeToolResponse(tr *llm.ToolResponse) error {
	if tr == nil {
		return nil
	}

	if tr.Error != "" {
		return sw.ew.WriteChunk(Chunk{
			"type": "tool-output-error", "toolCallId": tr.ID, "errorText": tr.Error,
		})
	}

	var output any
	if len(tr.Result) > 0 {
		_ = json.Unmarshal(tr.Result, &output)
	}

	return sw.ew.WriteChunk(Chunk{
		"type": "tool-output-available", "toolCallId": tr.ID, "output": output,
	})
}

// handleContentPart dispatches a content part event to the appropriate writer method.
func (sw *streamWriter) handleContentPart(part *llm.Part) error {
	switch part.Kind {
	case llm.PartText:
		return sw.writeTextDelta(part.Text)
	case llm.PartToolRequest:
		if err := sw.endText(); err != nil {
			return err
		}

		return sw.writeToolRequest(part.ToolRequest)
	case llm.PartToolResponse:
		return sw.writeToolResponse(part.ToolResponse)
	case llm.PartReasoning:
		return sw.writeReasoningDelta(part.ReasoningTrace)
	}

	return nil
}

// closeSpans closes any open reasoning/text spans, ignoring write errors.
func (sw *streamWriter) closeSpans() {
	if sw.reasoningStarted {
		_ = sw.ew.WriteChunk(Chunk{"type": "reasoning-end", "id": sw.reasoningID})
		sw.reasoningStarted = false
	}

	if sw.textStarted {
		_ = sw.ew.WriteChunk(Chunk{"type": "text-end", "id": sw.textID})
		sw.textStarted = false
		sw.textCounter++
	}
}

// writeStreamEnd handles a StreamEndEvent: emits error/close/finish chunks.
func (sw *streamWriter) writeStreamEnd(e llm.StreamEndEvent, logger *slog.Logger) {
	if e.Error != nil {
		logger.Error("LLM error", "error", e.Error)

		_ = sw.ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"})
	}

	sw.closeSpans()

	_ = sw.ew.WriteChunk(Chunk{"type": "finish-step"})

	reason := finishReasonStop
	if e.Error != nil {
		reason = finishReasonError
	} else if e.Response != nil {
		reason = mapFinishReason(e.Response.FinishReason)
	}

	_ = sw.ew.WriteChunk(Chunk{"type": "finish", "finishReason": reason})
}

// StreamModel streams responses from an llm.Model as AI SDK UI Message Stream
// events. This is the core streaming logic, usable from custom HTTP handlers.
func StreamModel(ctx context.Context, model llm.Model, req *llm.Request, ew *EventWriter, logger *slog.Logger) {
	if logger == nil {
		logger = slog.Default()
	}

	messageID := generateMessageID()

	if err := ew.WriteChunk(Chunk{"type": "start", "messageId": messageID}); err != nil {
		return
	}

	if err := ew.WriteChunk(Chunk{"type": "start-step"}); err != nil {
		return
	}

	sw := newStreamWriter(ew)

	for event, err := range model.GenerateEvents(ctx, req) {
		if err != nil {
			if ctx.Err() != nil {
				return
			}

			logger.Error("stream error", "error", err)

			_ = ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"})
			_ = ew.WriteChunk(Chunk{"type": "finish-step"})
			_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": finishReasonError})

			break
		}

		switch e := event.(type) {
		case llm.ContentPartEvent:
			if err := sw.handleContentPart(e.Part); err != nil {
				return
			}

		case llm.ErrorEvent:
			logger.Warn("recoverable LLM error", "message", e.Message)

			if err := ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"}); err != nil {
				return
			}

		case llm.StreamResetEvent:
			if err := sw.endText(); err != nil {
				return
			}

			if err := sw.endReasoning(); err != nil {
				return
			}

		case llm.StreamEndEvent:
			sw.writeStreamEnd(e, logger)
		}
	}

	_ = ew.WriteDone()
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

	messages := slices.Clone(req.Messages)
	sw := newStreamWriter(ew)

	const maxTurns = 10

	for range maxTurns {
		finishReason, toolRequests := streamToolTurn(ctx, model, req, messages, sw, ew, logger)

		if len(toolRequests) == 0 || finishReason != "tool-calls" {
			_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": finishReason})
			_ = ew.WriteDone()

			return
		}

		assistantParts := make([]*llm.Part, 0, len(toolRequests))
		for _, tr := range toolRequests {
			assistantParts = append(assistantParts, llm.NewToolRequestPart(tr))
		}

		messages = append(messages, llm.Message{Role: llm.RoleAssistant, Content: assistantParts})

		if err := executeTools(ctx, toolRequests, &messages, ew, executor); err != nil {
			return
		}
	}

	_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": finishReasonOther})
	_ = ew.WriteDone()
}

// streamToolTurn runs a single model turn and returns the finish reason and any tool requests.
func streamToolTurn(
	ctx context.Context,
	model llm.Model,
	req *llm.Request,
	messages []llm.Message,
	sw *streamWriter,
	ew *EventWriter,
	logger *slog.Logger,
) (string, []*llm.ToolRequest) {
	if err := ew.WriteChunk(Chunk{"type": "start-step"}); err != nil {
		return "", nil
	}

	sw.textID = fmt.Sprintf("text-%d", sw.textCounter)
	sw.textStarted = false
	sw.reasoningStarted = false

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
				return "", nil
			}

			logger.Error("stream error", "error", err)

			_ = ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"})
			_ = ew.WriteChunk(Chunk{"type": "finish-step"})
			_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": finishReasonError})
			_ = ew.WriteDone()

			return finishReasonError, nil
		}

		switch e := event.(type) {
		case llm.ContentPartEvent:
			if abort := handleToolTurnPart(e, sw, &toolRequests); abort {
				return "", nil
			}

		case llm.StreamEndEvent:
			finishReason = writeToolTurnEnd(e, sw, ew, logger)
		}
	}

	return finishReason, toolRequests
}

func handleToolTurnPart(e llm.ContentPartEvent, sw *streamWriter, toolRequests *[]*llm.ToolRequest) bool {
	switch e.Part.Kind {
	case llm.PartText:
		if err := sw.writeTextDelta(e.Part.Text); err != nil {
			return true
		}

	case llm.PartReasoning:
		if err := sw.writeReasoningDelta(e.Part.ReasoningTrace); err != nil {
			return true
		}

	case llm.PartToolRequest:
		if err := sw.endReasoning(); err != nil {
			return true
		}

		if err := sw.endTextAndAdvance(); err != nil {
			return true
		}

		tr := e.Part.ToolRequest
		if tr != nil {
			*toolRequests = append(*toolRequests, tr)

			if err := sw.writeToolRequest(tr); err != nil {
				return true
			}
		}

	case llm.PartToolResponse:
		// Tool responses in tool-calling mode are handled by executeTools.
	}

	return false
}

func writeToolTurnEnd(e llm.StreamEndEvent, sw *streamWriter, ew *EventWriter, logger *slog.Logger) string {
	if e.Error != nil {
		logger.Error("LLM error", "error", e.Error)

		_ = ew.WriteChunk(Chunk{"type": "error", "errorText": "An error occurred"})
	}

	sw.closeSpans()

	_ = ew.WriteChunk(Chunk{"type": "finish-step"})

	reason := finishReasonStop
	if e.Error != nil {
		reason = finishReasonError
	} else if e.Response != nil {
		reason = mapFinishReason(e.Response.FinishReason)
	}

	return reason
}

func executeTools(ctx context.Context, toolRequests []*llm.ToolRequest, messages *[]llm.Message, ew *EventWriter, executor ToolExecutor) error {
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

			continue
		}

		var output any
		if len(result) > 0 {
			_ = json.Unmarshal(result, &output)
		}

		if err := ew.WriteChunk(Chunk{
			"type": "tool-output-available", "toolCallId": tr.ID, "output": output,
		}); err != nil {
			return err
		}

		toolResponseParts = append(toolResponseParts, llm.NewToolResponsePart(&llm.ToolResponse{
			ID: tr.ID, Name: tr.Name, Result: result,
		}))
	}

	*messages = append(*messages, llm.Message{Role: llm.RoleUser, Content: toolResponseParts})

	return nil
}

func convertMessages(msgs []chatMessage, system string) []llm.Message {
	out := make([]llm.Message, 0, len(msgs)+1)
	if system != "" {
		out = append(out, llm.NewMessage(llm.RoleSystem, llm.NewTextPart(system)))
	}

	for _, m := range msgs {
		text := m.textContent()
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
// packages/ai/src/ui-message-stream/ui-message-stream-headers.ts.
func setSSEHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Vercel-Ai-Ui-Message-Stream", "v1")
	w.Header().Set("X-Accel-Buffering", "no")
}

func mapFinishReason(fr llm.FinishReason) string {
	switch fr {
	case llm.FinishReasonStop:
		return finishReasonStop
	case llm.FinishReasonLength:
		return "length"
	case llm.FinishReasonContentFilter:
		return "content-filter"
	case llm.FinishReasonToolCalls:
		return "tool-calls"
	case llm.FinishReasonInterrupted:
		return finishReasonOther
	case llm.FinishReasonUnknown:
		return finishReasonOther
	default:
		return finishReasonOther
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
