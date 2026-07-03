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
// This is a port of the Vercel AI SDK's UI message stream. Per-part chunk
// shapes mirror packages/ai/src/ui-message-stream/to-ui-message-chunk.ts, the
// stream assembly mirrors to-ui-message-stream.ts, and the SSE framing mirrors
// json-to-sse-transform-stream.ts. Verified against ai@7.0.6.
//
// Inbound multi-turn history (including assistant tool calls and their results)
// is reconstructed from the UI message parts, mirroring convert-to-model-messages.ts.
//
// Known limitations:
//   - Inbound file/image parts are not forwarded to the model: the llm.Part
//     type has no file kind yet. Inbound reasoning parts are likewise dropped.
//   - The handler calls model.GenerateEvents directly; interceptor plugins
//     (retry, OTel) must be wired at the model level.
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
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Handler returns an http.Handler that serves the AI SDK UI Message Stream
// protocol. It accepts POST requests with a JSON body containing messages
// and streams back SSE events compatible with useChat.
func Handler(model llm.Model, opts ...Option) http.Handler {
	cfg := &config{
		logger:       slog.Default(),
		maxBodyBytes: 1 << 20, // 1MB
		maxTurns:     10,
		onError:      defaultErrorMapper,
	}
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

// ErrorMapper maps an error to the client-facing text emitted in "error" and
// "tool-output-error" chunks. It mirrors the Vercel AI SDK's onError option
// (toUIMessageStream / toUIMessageChunk). The default returns a sanitized,
// generic message to avoid leaking server-side error details to the client.
type ErrorMapper func(error) string

// defaultErrorText matches the Vercel AI SDK default onError return value
// (packages/ai/src/ui-message-stream/to-ui-message-chunk.ts: () => 'An error occurred.').
const defaultErrorText = "An error occurred."

func defaultErrorMapper(error) string { return defaultErrorText }

type config struct {
	system       string
	logger       *slog.Logger
	tools        []llm.ToolDefinition
	executor     ToolExecutor
	maxBodyBytes int64
	maxTurns     int
	onError      ErrorMapper
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

// WithMaxBodyBytes sets the maximum request body size in bytes. Default is 1MB.
func WithMaxBodyBytes(n int64) Option {
	return func(c *config) { c.maxBodyBytes = n }
}

// WithMaxTurns sets the maximum number of agentic tool-calling turns. Default is 10.
func WithMaxTurns(n int) Option {
	return func(c *config) { c.maxTurns = n }
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

type handler struct {
	model llm.Model
	cfg   *config
}

// chatRequest matches the JSON body sent by useChat.
type chatRequest struct {
	ID       string        `json:"id"`
	Messages []chatMessage `json:"messages"`
}

type chatMessage struct {
	Role    string        `json:"role"`
	Content string        `json:"content"`
	Parts   []messagePart `json:"parts"`
}

type messagePart struct {
	Type string `json:"type"`
	Text string `json:"text"`

	// Tool-call part fields. Tool parts carry a Type of "tool-<toolName>"
	// (static tools) or "dynamic-tool" (with ToolName set), mirroring the v7 UI
	// message ToolUIPart / DynamicToolUIPart shapes. The JSON tags are the
	// camelCase names the Vercel AI SDK useChat client sends on the wire.
	ToolCallID string          `json:"toolCallId"` //nolint:tagliatelle // Vercel AI SDK wire format is camelCase
	ToolName   string          `json:"toolName"`   //nolint:tagliatelle // Vercel AI SDK wire format is camelCase
	State      string          `json:"state"`
	Input      json.RawMessage `json:"input"`
	Output     json.RawMessage `json:"output"`
	ErrorText  string          `json:"errorText"` //nolint:tagliatelle // Vercel AI SDK wire format is camelCase
}

// isTool reports whether the part is a tool-call part.
func (p messagePart) isTool() bool {
	return p.Type == "dynamic-tool" || strings.HasPrefix(p.Type, "tool-")
}

// toolName returns the tool name for a tool-call part. Static tool parts encode
// the name in the type suffix ("tool-getWeather"); dynamic tool parts carry it
// in the toolName field.
func (p messagePart) toolName() string {
	if p.Type == "dynamic-tool" {
		return p.ToolName
	}

	return strings.TrimPrefix(p.Type, "tool-")
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

func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Limit request body size to prevent abuse.
	r.Body = http.MaxBytesReader(w, r.Body, h.cfg.maxBodyBytes)

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
	StreamModelWithTools(r.Context(), h.model, req, ew, h.cfg.logger, h.cfg.executor, h.cfg.maxTurns, h.cfg.onError)
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

// streamWriter manages SSE event state (open text/reasoning spans) for a
// single streaming response. Extracting this reduces the cyclomatic
// complexity of StreamModel and StreamModelWithTools.
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

func (sw *streamWriter) writeToolRequest(tr *llm.ToolRequestPart) error {
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
		if err := json.Unmarshal(tr.Arguments, &input); err != nil {
			sw.logger.Warn("failed to unmarshal tool input", "toolCallId", tr.ID, "error", err)
		}
	}

	return sw.ew.WriteChunk(Chunk{
		"type": "tool-input-available", "toolCallId": tr.ID, "toolName": tr.Name, "input": input,
	})
}

func (sw *streamWriter) writeToolResponse(tr *llm.ToolResponsePart) error {
	if tr == nil {
		return nil
	}

	if tr.IsError {
		// A provider-executed tool reported an error; its payload travels in
		// Result. The reference surfaces provider-executed tool errors verbatim.
		return sw.ew.WriteChunk(Chunk{
			"type": "tool-output-error", "toolCallId": tr.ID, "errorText": string(tr.Result),
		})
	}

	var output any
	if len(tr.Result) > 0 {
		if err := json.Unmarshal(tr.Result, &output); err != nil {
			sw.logger.Warn("failed to unmarshal tool output", "toolCallId", tr.ID, "error", err)
		}
	}

	return sw.ew.WriteChunk(Chunk{
		"type": "tool-output-available", "toolCallId": tr.ID, "output": output,
	})
}

// handleContentPart dispatches a content part event to the appropriate writer method.
func (sw *streamWriter) handleContentPart(part llm.Part) error {
	switch p := part.(type) {
	case *llm.TextPart:
		return sw.writeTextDelta(p.Text)
	case *llm.ToolRequestPart:
		if err := sw.endReasoning(); err != nil {
			return err
		}

		if err := sw.endTextAndAdvance(); err != nil {
			return err
		}

		return sw.writeToolRequest(p)
	case *llm.ToolResponsePart:
		if err := sw.endReasoning(); err != nil {
			return err
		}

		if err := sw.endTextAndAdvance(); err != nil {
			return err
		}

		return sw.writeToolResponse(p)
	case *llm.ReasoningPart:
		return sw.writeReasoningDelta(p)
	}

	return nil
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

// writeStreamEnd handles a StreamEndEvent: emits error/close/finish chunks.
func (sw *streamWriter) writeStreamEnd(e llm.StreamEndEvent, logger *slog.Logger) {
	sw.closeSpans()

	if e.Error != nil {
		logger.Error("LLM error", "error", e.Error)

		_ = sw.ew.WriteChunk(Chunk{"type": "error", "errorText": sw.onError(e.Error)})
	}

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
func StreamModel(ctx context.Context, model llm.Model, req *llm.Request, ew *EventWriter, logger *slog.Logger, onError ErrorMapper) {
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

	sw := newStreamWriter(ew, logger, onError)

	for event, err := range model.GenerateEvents(ctx, req) {
		if err != nil {
			if ctx.Err() != nil {
				sw.writeAbort(ctx)
				return
			}

			logger.Error("stream error", "error", err)

			sw.closeSpans()

			_ = ew.WriteChunk(Chunk{"type": "error", "errorText": sw.onError(err)})
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

			if err := ew.WriteChunk(Chunk{"type": "error", "errorText": sw.onError(errorEventErr(e))}); err != nil {
				return
			}

		case llm.StreamResetEvent:
			if err := sw.endTextAndAdvance(); err != nil {
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

// errorEventErr converts a recoverable ErrorEvent into an error so it can be
// passed through the configured ErrorMapper, mirroring how the reference routes
// every error chunk through onError.
func errorEventErr(e llm.ErrorEvent) error {
	if e.Code != "" {
		return fmt.Errorf("%s: %s", e.Code, e.Message)
	}

	return errors.New(e.Message)
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

// StreamModelWithTools is like StreamModel but supports agentic tool calling.
// When the model returns tool calls, the executor is invoked for each, results
// are streamed to the client, and the model is called again with the results
// appended to the conversation. This loops until the model stops calling tools
// or maxTurns is reached. If maxTurns is 0, it defaults to 10.
func StreamModelWithTools(ctx context.Context, model llm.Model, req *llm.Request, ew *EventWriter, logger *slog.Logger, executor ToolExecutor, maxTurns int, onError ErrorMapper) {
	if executor == nil {
		StreamModel(ctx, model, req, ew, logger, onError)
		return
	}

	if logger == nil {
		logger = slog.Default()
	}

	if maxTurns <= 0 {
		maxTurns = 10
	}

	messageID := generateMessageID()
	if err := ew.WriteChunk(Chunk{"type": "start", "messageId": messageID}); err != nil {
		return
	}

	messages := slices.Clone(req.Messages)
	sw := newStreamWriter(ew, logger, onError)

	lastFinishReason := finishReasonOther

	for range maxTurns {
		finishReason, toolRequests := streamToolTurn(ctx, model, req, messages, sw, ew, logger)

		// Empty finishReason means the stream was aborted (ctx cancel or write failure).
		if finishReason == "" {
			return
		}

		lastFinishReason = finishReason

		if len(toolRequests) == 0 || finishReason != "tool-calls" {
			_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": finishReason})
			_ = ew.WriteDone()

			return
		}

		assistantParts := make([]llm.Part, 0, len(toolRequests))
		for _, tr := range toolRequests {
			assistantParts = append(assistantParts, tr)
		}

		messages = append(messages, llm.Message{Role: llm.RoleAssistant, Content: assistantParts})

		if err := executeTools(ctx, toolRequests, &messages, ew, logger, executor, sw.onError); err != nil {
			return
		}
	}

	// maxTurns exhausted: surface the last turn's real finish reason (which is
	// "tool-calls", since the loop only continues while the model keeps calling
	// tools), matching the reference which always emits the model's last-step
	// finish reason rather than a synthetic value.
	_ = ew.WriteChunk(Chunk{"type": "finish", "finishReason": lastFinishReason})
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
) (string, []*llm.ToolRequestPart) {
	if err := ew.WriteChunk(Chunk{"type": "start-step"}); err != nil {
		return "", nil
	}

	var toolRequests []*llm.ToolRequestPart

	iterReq := *req
	iterReq.Messages = messages

	var finishReason string

	for event, err := range model.GenerateEvents(ctx, &iterReq) {
		if err != nil {
			if ctx.Err() != nil {
				sw.writeAbort(ctx)
				return "", nil
			}

			logger.Error("stream error", "error", err)

			sw.closeSpans()

			for _, tr := range toolRequests {
				_ = ew.WriteChunk(Chunk{
					"type": "tool-output-error", "toolCallId": tr.ID,
					"errorText": "stream error; tool call discarded",
				})
			}

			_ = ew.WriteChunk(Chunk{"type": "error", "errorText": sw.onError(err)})
			_ = ew.WriteChunk(Chunk{"type": "finish-step"})

			return finishReasonError, nil
		}

		switch e := event.(type) {
		case llm.ContentPartEvent:
			if abort := handleToolTurnPart(e, sw, &toolRequests); abort {
				return "", nil
			}

		case llm.ErrorEvent:
			logger.Warn("recoverable LLM error", "message", e.Message)

			if err := ew.WriteChunk(Chunk{"type": "error", "errorText": sw.onError(errorEventErr(e))}); err != nil {
				return "", nil
			}

		case llm.StreamResetEvent:
			if err := sw.endTextAndAdvance(); err != nil {
				return "", nil
			}

			if err := sw.endReasoning(); err != nil {
				return "", nil
			}

			for _, tr := range toolRequests {
				_ = ew.WriteChunk(Chunk{
					"type": "tool-output-error", "toolCallId": tr.ID,
					"errorText": "stream reset; tool call discarded",
				})
			}

			toolRequests = nil

		case llm.StreamEndEvent:
			finishReason = writeToolTurnEnd(e, sw, ew, logger)

			if e.Error != nil {
				for _, tr := range toolRequests {
					_ = ew.WriteChunk(Chunk{
						"type": "tool-output-error", "toolCallId": tr.ID,
						"errorText": "stream error; tool call discarded",
					})
				}

				toolRequests = nil
			}
		}
	}

	return finishReason, toolRequests
}

func handleToolTurnPart(e llm.ContentPartEvent, sw *streamWriter, toolRequests *[]*llm.ToolRequestPart) bool {
	switch p := e.Part.(type) {
	case *llm.TextPart:
		if err := sw.writeTextDelta(p.Text); err != nil {
			return true
		}

	case *llm.ReasoningPart:
		if err := sw.writeReasoningDelta(p); err != nil {
			return true
		}

	case *llm.ToolRequestPart:
		if err := sw.endReasoning(); err != nil {
			return true
		}

		if err := sw.endTextAndAdvance(); err != nil {
			return true
		}

		if p != nil {
			*toolRequests = append(*toolRequests, p)

			if err := sw.writeToolRequest(p); err != nil {
				return true
			}
		}

	case *llm.ToolResponsePart:
		// Tool responses in tool-calling mode are handled by executeTools.
	}

	return false
}

func writeToolTurnEnd(e llm.StreamEndEvent, sw *streamWriter, ew *EventWriter, logger *slog.Logger) string {
	sw.closeSpans()

	if e.Error != nil {
		logger.Error("LLM error", "error", e.Error)

		_ = ew.WriteChunk(Chunk{"type": "error", "errorText": sw.onError(e.Error)})
	}

	_ = ew.WriteChunk(Chunk{"type": "finish-step"})

	reason := finishReasonStop
	if e.Error != nil {
		reason = finishReasonError
	} else if e.Response != nil {
		reason = mapFinishReason(e.Response.FinishReason)
	}

	return reason
}

func executeTools(ctx context.Context, toolRequests []*llm.ToolRequestPart, messages *[]llm.Message, ew *EventWriter, logger *slog.Logger, executor ToolExecutor, onError ErrorMapper) error {
	toolResponseParts := make([]llm.Part, 0, len(toolRequests))

	for _, tr := range toolRequests {
		result, err := executor(ctx, tr.Name, tr.Arguments)
		if err != nil {
			// The client-facing text is sanitized through onError; the model
			// still receives the real error so it can recover on the next turn.
			_ = ew.WriteChunk(Chunk{
				"type": "tool-output-error", "toolCallId": tr.ID, "errorText": onError(err),
			})

			errPayload, mErr := json.Marshal(map[string]string{"error": err.Error()})
			if mErr != nil {
				errPayload = []byte(`{"error":"tool error"}`)
			}

			toolResponseParts = append(toolResponseParts, llm.NewToolResponsePart(tr.ID, tr.Name, errPayload, true))

			continue
		}

		var output any
		if len(result) > 0 {
			if err := json.Unmarshal(result, &output); err != nil {
				logger.Warn("failed to unmarshal tool result", "toolCallId", tr.ID, "error", err)
			}
		}

		if err := ew.WriteChunk(Chunk{
			"type": "tool-output-available", "toolCallId": tr.ID, "output": output,
		}); err != nil {
			return err
		}

		toolResponseParts = append(toolResponseParts, llm.NewToolResponsePart(tr.ID, tr.Name, result, false))
	}

	*messages = append(*messages, llm.Message{Role: llm.RoleUser, Content: toolResponseParts})

	return nil
}

func convertMessages(msgs []chatMessage, system string) []llm.Message {
	out := make([]llm.Message, 0, len(msgs)+1)

	// appendMessage coalesces consecutive same-role messages into one, mirroring
	// how the reference providers merge adjacent same-role model messages (e.g.
	// Anthropic's groupIntoBlocks). This keeps roles strictly alternating, which
	// providers such as Anthropic require. Without it, two text-only assistant
	// steps — or a dropped/incomplete assistant turn between two user turns —
	// would emit consecutive same-role messages and the provider would reject them.
	appendMessage := func(m llm.Message) {
		if n := len(out); n > 0 && out[n-1].Role == m.Role {
			out[n-1].Content = append(out[n-1].Content, m.Content...)
			return
		}

		out = append(out, m)
	}

	if system != "" {
		appendMessage(llm.NewMessage(llm.RoleSystem, llm.NewTextPart(system)))
	}

	for _, m := range msgs {
		role := messageRole(m.Role)

		if role == llm.RoleAssistant {
			for _, am := range reconstructAssistant(m) {
				appendMessage(am)
			}

			continue
		}

		// User and system messages forward their concatenated text. Inbound
		// file parts are dropped: llm.Part has no file kind.
		text := m.textContent()
		if text == "" {
			continue
		}

		appendMessage(llm.NewMessage(role, llm.NewTextPart(text)))
	}

	return out
}

// messageRole maps a UI message role string to an llm.MessageRole.
func messageRole(role string) llm.MessageRole {
	switch role {
	case "assistant":
		return llm.RoleAssistant
	case "system":
		return llm.RoleSystem
	default:
		return llm.RoleUser
	}
}

// reconstructAssistant rebuilds the model messages for an assistant turn from
// its UI message parts, mirroring convert-to-model-messages.ts. Each step
// (delimited by "step-start" parts) becomes an assistant message containing its
// text and tool-call parts, optionally followed by a user message carrying that
// step's tool results (output-available / output-error). Splitting on steps
// preserves the call -> result -> answer ordering that providers such as
// Anthropic require.
//
// Inbound reasoning and file parts are not reconstructed: providers vary in
// accepting reasoning history, and llm.Part has no file kind.
func reconstructAssistant(m chatMessage) []llm.Message {
	// Legacy content field (no parts): a single assistant text message.
	if len(m.Parts) == 0 {
		if m.Content == "" {
			return nil
		}

		return []llm.Message{llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(m.Content))}
	}

	var (
		msgs        []llm.Message
		assistant   []llm.Part
		toolResults []llm.Part
	)

	flush := func() {
		if len(assistant) > 0 {
			msgs = append(msgs, llm.Message{Role: llm.RoleAssistant, Content: assistant})
		}

		if len(toolResults) > 0 {
			msgs = append(msgs, llm.Message{Role: llm.RoleUser, Content: toolResults})
		}

		assistant = nil
		toolResults = nil
	}

	for _, p := range m.Parts {
		switch {
		case p.Type == "step-start":
			// Step boundary: close the current block so its tool results are
			// ordered before the next step's text/answer.
			flush()

		case p.Type == "text":
			if p.Text != "" {
				assistant = append(assistant, llm.NewTextPart(p.Text))
			}

		case p.isTool():
			if p.ToolCallID == "" {
				continue
			}

			name := p.toolName()

			// Only reconstruct COMPLETED tool calls — those with a result. A call
			// still streaming its input, or one that reached input-available but
			// never produced an output (an aborted/interrupted prior turn the
			// client re-sends), is dropped entirely: the tool-request part is
			// emitted ONLY inside the result-bearing states. Emitting a bare
			// tool-request with no paired tool-response would (a) make providers
			// such as Anthropic reject the request for an unmatched tool_use, and
			// (b) let a browser client forge an incomplete assistant tool call
			// that an agent's crash-recovery path would execute before consulting
			// the model. Dropping the unresolved call keeps history well-formed
			// and denies that vector.
			switch p.State {
			case "output-available":
				assistant = append(assistant, llm.NewToolRequestPart(p.ToolCallID, name, p.Input))
				toolResults = append(toolResults, llm.NewToolResponsePart(p.ToolCallID, name, p.Output, false))
			case "output-error":
				// The new ToolResponsePart carries the error payload in Result
				// with IsError set, rather than a dedicated error string field.
				errPayload, mErr := json.Marshal(map[string]string{"error": p.ErrorText})
				if mErr != nil {
					errPayload = []byte(`{"error":"tool error"}`)
				}

				assistant = append(assistant, llm.NewToolRequestPart(p.ToolCallID, name, p.Input))
				toolResults = append(toolResults, llm.NewToolResponsePart(p.ToolCallID, name, errPayload, true))
			}
		}
	}

	flush()

	return msgs
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
	case llm.FinishReasonContextOverflow:
		// The Vercel protocol has no dedicated overflow reason. Report it as
		// "length" — the token-limit bucket — which is also the value overflow
		// wired to before it was split out from FinishReasonLength.
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
