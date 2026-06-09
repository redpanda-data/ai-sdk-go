package uimessagestream

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

const (
	typeStart          = "start"
	typeStartStep      = "start-step"
	typeTextStart      = "text-start"
	typeTextDelta      = "text-delta"
	typeTextEnd        = "text-end"
	typeFinishStep     = "finish-step"
	typeFinish         = "finish"
	typeError          = "error"
	typeReasoningStart = "reasoning-start"
	typeReasoningDelta = "reasoning-delta"
	typeReasoningEnd   = "reasoning-end"

	sanitizedError = "An error occurred"
)

// parseSSE reads SSE events from a response body and returns the parsed JSON
// chunks and whether the stream was properly terminated with [DONE].
func parseSSE(t *testing.T, body io.Reader) ([]Chunk, bool) {
	t.Helper()

	var chunks []Chunk

	done := false

	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			t.Fatalf("unexpected line format: %q", line)
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			done = true

			continue
		}

		var chunk Chunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("failed to parse chunk JSON: %v\ndata: %s", err, data)
		}

		chunks = append(chunks, chunk)
	}

	return chunks, done
}

func chunkTypes(chunks []Chunk) []string {
	types := make([]string, len(chunks))
	for i, c := range chunks {
		types[i], _ = c["type"].(string)
	}

	return types
}

func chunkStr(t *testing.T, c Chunk, key string) string {
	t.Helper()

	v, ok := c[key].(string)
	if !ok {
		t.Fatalf("chunk[%q] is not a string: %v", key, c[key])
	}

	return v
}

func newPostRequest(t *testing.T, body string) *http.Request {
	t.Helper()

	req := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	return req
}

func TestHandler_SimpleTextResponse(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("Hello, world!", fakellm.StreamConfig{ChunkSize: 100})

	h := Handler(model)

	body := `{"id":"chat-1","messages":[{"role":"user","content":"hi"}]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	if v := rec.Header().Get("X-Vercel-Ai-Ui-Message-Stream"); v != "v1" {
		t.Errorf("X-Vercel-Ai-Ui-Message-Stream = %q, want v1", v)
	}

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expectedPrefix := []string{typeStart, typeStartStep, typeTextStart}
	for i, exp := range expectedPrefix {
		if i >= len(types) || types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall types: %v", i, types[i], exp, types)
		}
	}

	for i := 3; i < len(types)-3; i++ {
		if types[i] != typeTextDelta {
			t.Errorf("chunk[%d] type = %q, want text-delta", i, types[i])
		}
	}

	expectedSuffix := []string{typeTextEnd, typeFinishStep, typeFinish}
	for i, exp := range expectedSuffix {
		idx := len(types) - 3 + i
		if idx < 0 || idx >= len(types) || types[idx] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall types: %v", idx, types[idx], exp, types)
		}
	}

	var text strings.Builder

	for _, c := range chunks {
		if c["type"] == typeTextDelta {
			text.WriteString(chunkStr(t, c, "delta"))
		}
	}

	if got := text.String(); got != "Hello, world!" {
		t.Errorf("assembled text = %q, want %q", got, "Hello, world!")
	}

	var startID, endID string

	for _, c := range chunks {
		if c["type"] == typeTextStart {
			startID = chunkStr(t, c, "id")
		}

		if c["type"] == typeTextEnd {
			endID = chunkStr(t, c, "id")
		}
	}

	if startID != endID {
		t.Errorf("text-start id = %q, text-end id = %q, want match", startID, endID)
	}

	for _, c := range chunks {
		if c["type"] == typeFinish {
			if reason := chunkStr(t, c, "finishReason"); reason != finishReasonStop {
				t.Errorf("finish.finishReason = %v, want 'stop'", c["finishReason"])
			}
		}
	}
}

func TestHandler_StreamingTextResponse(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("Streaming works!", fakellm.StreamConfig{ChunkSize: 4})

	h := Handler(model)

	body := `{"id":"chat-2","messages":[{"role":"user","content":"test"}]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	var text strings.Builder

	deltaCount := 0

	for _, c := range chunks {
		if c["type"] == typeTextDelta {
			text.WriteString(chunkStr(t, c, "delta"))

			deltaCount++
		}
	}

	if got := text.String(); got != "Streaming works!" {
		t.Errorf("assembled text = %q, want %q", got, "Streaming works!")
	}

	if deltaCount < 2 {
		t.Errorf("expected multiple text-delta chunks for streaming, got %d", deltaCount)
	}
}

func TestHandler_ErrorResponse(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenError(llm.ErrRateLimitExceeded)

	h := Handler(model)

	body := `{"id":"chat-3","messages":[{"role":"user","content":"fail"}]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{typeStart, typeStartStep, typeError, typeFinishStep, typeFinish}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}

	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall types: %v", i, types[i], exp, types)
		}
	}

	if et := chunkStr(t, chunks[2], "errorText"); et != sanitizedError {
		t.Errorf("error chunk errorText = %q, want sanitized %q", et, sanitizedError)
	}

	if reason := chunkStr(t, chunks[4], "finishReason"); reason != finishReasonError {
		t.Errorf("finish.finishReason = %v, want %q", chunks[4]["finishReason"], finishReasonError)
	}
}

func TestHandler_SystemPrompt(t *testing.T) {
	t.Parallel()

	var capturedMessages []llm.Message

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			capturedMessages = req.Messages

			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("ok")),
				FinishReason: llm.FinishReasonStop,
			}, nil
		})

	h := Handler(model, WithSystem("Be concise."))

	body := `{"id":"chat-4","messages":[{"role":"user","content":"hi"}]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if len(capturedMessages) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(capturedMessages))
	}

	if capturedMessages[0].Role != llm.RoleSystem {
		t.Errorf("first message role = %q, want system", capturedMessages[0].Role)
	}

	if capturedMessages[0].TextContent() != "Be concise." {
		t.Errorf("system prompt = %q, want 'Be concise.'", capturedMessages[0].TextContent())
	}
}

func TestHandler_MultiTurnConversation(t *testing.T) {
	t.Parallel()

	var capturedMessages []llm.Message

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			capturedMessages = req.Messages

			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("response")),
				FinishReason: llm.FinishReasonStop,
			}, nil
		})

	h := Handler(model)

	body := `{"id":"chat-5","messages":[
		{"role":"user","content":"hello"},
		{"role":"assistant","content":"hi there"},
		{"role":"user","content":"how are you?"}
	]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if len(capturedMessages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(capturedMessages))
	}

	if capturedMessages[0].Role != llm.RoleUser {
		t.Errorf("msg[0] role = %q, want user", capturedMessages[0].Role)
	}

	if capturedMessages[1].Role != llm.RoleAssistant {
		t.Errorf("msg[1] role = %q, want assistant", capturedMessages[1].Role)
	}

	if capturedMessages[2].Role != llm.RoleUser {
		t.Errorf("msg[2] role = %q, want user", capturedMessages[2].Role)
	}
}

func TestHandler_MethodNotAllowed(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	h := Handler(model)

	req := httptest.NewRequestWithContext(context.Background(), http.MethodGet, "/api/chat", nil)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", rec.Code)
	}
}

func TestHandler_InvalidBody(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	h := Handler(model)

	req := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/api/chat", strings.NewReader("not json"))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rec.Code)
	}
}

func TestHandler_V6PartsFormat(t *testing.T) {
	t.Parallel()

	var capturedMessages []llm.Message

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			capturedMessages = req.Messages

			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("ok")),
				FinishReason: llm.FinishReasonStop,
			}, nil
		})

	h := Handler(model)

	body := `{
		"id": "chat-v6",
		"trigger": "submit-message",
		"messages": [
			{
				"role": "user",
				"parts": [{"type": "text", "text": "hello from v6"}],
				"id": "msg-1"
			},
			{
				"role": "assistant",
				"parts": [{"type": "step-start"}, {"type": "text", "text": "hi there", "state": "done"}],
				"id": "msg-2"
			},
			{
				"role": "user",
				"parts": [{"type": "text", "text": "follow up"}],
				"id": "msg-3"
			}
		]
	}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if len(capturedMessages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(capturedMessages))
	}

	if capturedMessages[0].TextContent() != "hello from v6" {
		t.Errorf("msg[0] text = %q, want 'hello from v6'", capturedMessages[0].TextContent())
	}

	if capturedMessages[1].TextContent() != "hi there" {
		t.Errorf("msg[1] text = %q, want 'hi there'", capturedMessages[1].TextContent())
	}

	if capturedMessages[2].TextContent() != "follow up" {
		t.Errorf("msg[2] text = %q, want 'follow up'", capturedMessages[2].TextContent())
	}
}

// errorStreamModel is a minimal llm.Model that yields specific event sequences
// for testing error handling in StreamModel.
type errorStreamModel struct {
	events []llm.Event
}

func (m *errorStreamModel) Name() string     { return "error-test-model" }
func (m *errorStreamModel) Provider() string { return "test" }

func (m *errorStreamModel) Capabilities() llm.ModelCapabilities {
	return llm.ModelCapabilities{Streaming: true}
}

func (m *errorStreamModel) Constraints() llm.ModelConstraints { return llm.ModelConstraints{} }

func (m *errorStreamModel) Generate(_ context.Context, _ *llm.Request) (*llm.Response, error) {
	return nil, errors.New("not implemented")
}

func (m *errorStreamModel) GenerateEvents(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, e := range m.events {
			if !yield(e, nil) {
				return
			}
		}
	}
}

func TestStreamModel_StreamEndEventWithError(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("partial")},
			llm.StreamEndEvent{Error: errors.New("provider exploded")},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{typeStart, typeStartStep, typeTextStart, typeTextDelta, typeError, typeTextEnd, typeFinishStep, typeFinish}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}

	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall types: %v", i, types[i], exp, types)
		}
	}

	for _, c := range chunks {
		if c["type"] == typeError {
			if et := chunkStr(t, c, "errorText"); et != sanitizedError {
				t.Errorf("error.errorText = %v, want %q", c["errorText"], sanitizedError)
			}
		}
	}

	finishChunk := chunks[len(chunks)-1]
	if reason := chunkStr(t, finishChunk, "finishReason"); reason != finishReasonError {
		t.Errorf("finish.finishReason = %v, want %q", finishChunk["finishReason"], finishReasonError)
	}
}

func TestHandler_FinishReasonMapping(t *testing.T) {
	t.Parallel()

	tests := []struct {
		reason llm.FinishReason
		want   string
	}{
		{llm.FinishReasonStop, "stop"},
		{llm.FinishReasonLength, "length"},
		{llm.FinishReasonContentFilter, "content-filter"},
		{llm.FinishReasonToolCalls, "tool-calls"},
		{llm.FinishReasonInterrupted, "other"},
		{llm.FinishReasonUnknown, "other"},
		{llm.FinishReason("unknown_reason"), "other"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			t.Parallel()

			if got := mapFinishReason(tt.reason); got != tt.want {
				t.Errorf("mapFinishReason(%q) = %q, want %q", tt.reason, got, tt.want)
			}
		})
	}
}

func TestHandler_WithLogger(t *testing.T) {
	t.Parallel()

	logger := slog.New(slog.DiscardHandler)
	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("ok", fakellm.StreamConfig{ChunkSize: 100})

	h := Handler(model, WithLogger(logger))

	body := `{"id":"chat-log","messages":[{"role":"user","content":"hi"}]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

func TestHandler_SystemRoleMessage(t *testing.T) {
	t.Parallel()

	var capturedMessages []llm.Message

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			capturedMessages = req.Messages

			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("ok")),
				FinishReason: llm.FinishReasonStop,
			}, nil
		})

	h := Handler(model)

	body := `{"id":"chat-sys","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if len(capturedMessages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(capturedMessages))
	}

	if capturedMessages[0].Role != llm.RoleSystem {
		t.Errorf("msg[0] role = %q, want system", capturedMessages[0].Role)
	}
}

func TestHandler_EmptyMessagesSkipped(t *testing.T) {
	t.Parallel()

	var capturedMessages []llm.Message

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			capturedMessages = req.Messages

			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("ok")),
				FinishReason: llm.FinishReasonStop,
			}, nil
		})

	h := Handler(model)

	body := `{"id":"chat-empty","messages":[
		{"role":"user","parts":[{"type":"step-start"}]},
		{"role":"user","content":"real message"}
	]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if len(capturedMessages) != 1 {
		t.Fatalf("expected 1 message (empty skipped), got %d", len(capturedMessages))
	}

	if capturedMessages[0].TextContent() != "real message" {
		t.Errorf("msg text = %q, want 'real message'", capturedMessages[0].TextContent())
	}
}

func TestHandler_ContextCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cancellingModel := &cancellingErrorModel{cancel: cancel}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(ctx, cancellingModel, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil)

	chunks, _ := parseSSE(t, rec.Body)

	types := chunkTypes(chunks)
	if len(types) < 2 || types[0] != typeStart || types[1] != typeStartStep {
		t.Fatalf("expected at least start + start-step, got %v", types)
	}

	for _, tp := range types {
		if tp == typeFinish {
			t.Error("should not have finish chunk when context is cancelled")
		}
	}
}

func TestHandler_ErrorEventNonTerminal(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("before")},
			llm.ErrorEvent{Message: "recoverable warning"},
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart(" after")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{typeStart, typeStartStep, typeTextStart, typeTextDelta, typeError, typeTextDelta, typeTextEnd, typeFinishStep, typeFinish}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}

	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall: %v", i, types[i], exp, types)
		}
	}

	for _, c := range chunks {
		if c["type"] == typeError {
			if et := chunkStr(t, c, "errorText"); et != sanitizedError {
				t.Errorf("errorText = %q, want %q (sanitized)", et, sanitizedError)
			}
		}
	}

	var text strings.Builder

	for _, c := range chunks {
		if c["type"] == typeTextDelta {
			text.WriteString(chunkStr(t, c, "delta"))
		}
	}

	if got := text.String(); got != "before after" {
		t.Errorf("assembled text = %q, want 'before after'", got)
	}
}

func TestHandler_TextContentPrefersPartsOverContent(t *testing.T) {
	t.Parallel()

	msg := chatMessage{
		Role:    "user",
		Content: "legacy content",
		Parts:   []messagePart{{Type: "text", Text: "parts content"}},
	}

	if got := msg.textContent(); got != "parts content" {
		t.Errorf("textContent() = %q, want 'parts content'", got)
	}
}

func TestHandler_TextContentFallsBackToContent(t *testing.T) {
	t.Parallel()

	msg := chatMessage{
		Role:    "user",
		Content: "legacy content",
	}

	if got := msg.textContent(); got != "legacy content" {
		t.Errorf("textContent() = %q, want 'legacy content'", got)
	}
}

func TestStreamModel_ReasoningTrace(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart("thinking...")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewTextPart("result")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("think"))},
	}, ew, nil)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{typeStart, typeStartStep, typeReasoningStart, typeReasoningDelta, typeReasoningEnd, typeTextStart, typeTextDelta, typeTextEnd, typeFinishStep, typeFinish}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}

	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall: %v", i, types[i], exp, types)
		}
	}

	for _, c := range chunks {
		if c["type"] == typeReasoningDelta {
			if d := chunkStr(t, c, "delta"); d != "thinking..." {
				t.Errorf("reasoning-delta = %q, want 'thinking...'", d)
			}
		}
	}
}

func TestStreamModel_ReasoningStatefulTracking(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart("step 1")},
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart(" step 2")},
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart(" step 3")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewTextPart("result")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("think hard"))},
	}, ew, nil)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeReasoningStart, typeReasoningDelta, typeReasoningDelta, typeReasoningDelta, typeReasoningEnd,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}

	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall: %v", i, types[i], exp, types)
		}
	}

	var rStartID, rEndID string

	var deltaIDs []string

	for _, c := range chunks {
		switch c["type"] {
		case typeReasoningStart:
			rStartID = chunkStr(t, c, "id")
		case typeReasoningEnd:
			rEndID = chunkStr(t, c, "id")
		case typeReasoningDelta:
			deltaIDs = append(deltaIDs, chunkStr(t, c, "id"))
		}
	}

	if rStartID != rEndID {
		t.Errorf("reasoning-start id = %q, reasoning-end id = %q, want match", rStartID, rEndID)
	}

	for i, id := range deltaIDs {
		if id != rStartID {
			t.Errorf("reasoning-delta[%d] id = %q, want %q", i, id, rStartID)
		}
	}

	var reasoning strings.Builder

	for _, c := range chunks {
		if c["type"] == typeReasoningDelta {
			reasoning.WriteString(chunkStr(t, c, "delta"))
		}
	}

	if got := reasoning.String(); got != "step 1 step 2 step 3" {
		t.Errorf("assembled reasoning = %q, want 'step 1 step 2 step 3'", got)
	}
}

func TestStreamModel_ReasoningNilTrace(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart("")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("think"))},
	}, ew, nil)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{typeStart, typeStartStep, typeReasoningStart, typeReasoningEnd, typeFinishStep, typeFinish}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}
}

type noFlushResponseWriter struct {
	http.ResponseWriter
}

func TestHandler_NoFlusherSupport(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	h := Handler(model)

	body := `{"id":"nf","messages":[{"role":"user","content":"hi"}]}`
	req := newPostRequest(t, body)

	inner := httptest.NewRecorder()
	rec := &noFlushResponseWriter{inner}

	h.ServeHTTP(rec, req)

	if inner.Code != http.StatusInternalServerError {
		t.Errorf("expected 500, got %d", inner.Code)
	}
}

func TestStreamModel_IteratorErrorWithCancelledContext(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cancellingModel := &cancellingErrorModel{cancel: cancel}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(ctx, cancellingModel, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil)

	chunks, _ := parseSSE(t, rec.Body)

	types := chunkTypes(chunks)
	if len(types) < 2 || types[0] != typeStart || types[1] != typeStartStep {
		t.Fatalf("expected at least start + start-step, got %v", types)
	}

	for _, tp := range types {
		if tp == typeFinish {
			t.Error("should not have finish chunk when context is cancelled")
		}
	}
}

// cancellingErrorModel cancels the context then returns an error from the iterator.
type cancellingErrorModel struct {
	errorStreamModel

	cancel context.CancelFunc
}

func (m *cancellingErrorModel) GenerateEvents(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		m.cancel()
		yield(nil, errors.New("after cancel"))
	}
}

func TestHandler_AllRequiredHeaders(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("x", fakellm.StreamConfig{ChunkSize: 100})

	h := Handler(model)

	body := `{"id":"hdr","messages":[{"role":"user","content":"hi"}]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	headers := map[string]string{
		"Content-Type":                  "text/event-stream",
		"Cache-Control":                 "no-cache",
		"Connection":                    "keep-alive",
		"X-Vercel-Ai-Ui-Message-Stream": "v1",
		"X-Accel-Buffering":             "no",
	}

	for k, want := range headers {
		if got := rec.Header().Get(k); got != want {
			t.Errorf("header %q = %q, want %q", k, got, want)
		}
	}
}

func TestHandler_StartChunkHasMessageID(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("ok", fakellm.StreamConfig{ChunkSize: 100})

	h := Handler(model)

	body := `{"id":"chat-mid","messages":[{"role":"user","content":"hi"}]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}

	startChunk := chunks[0]
	if startChunk["type"] != typeStart {
		t.Fatalf("first chunk type = %q, want 'start'", startChunk["type"])
	}

	mid := chunkStr(t, startChunk, "messageId")
	if mid == "" {
		t.Fatalf("start chunk has no messageId")
	}

	if len(mid) != 16 {
		t.Errorf("messageId length = %d, want 16", len(mid))
	}
}

func TestHandler_RequestBodySizeLimit(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	h := Handler(model)

	bigBody := strings.Repeat("x", 1<<20+1)
	req := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/api/chat", strings.NewReader(bigBody))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for oversized body, got %d", rec.Code)
	}
}

func TestHandler_EmptyMessagesArray(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	h := Handler(model)

	body := `{"id":"chat-none","messages":[]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty messages, got %d", rec.Code)
	}
}

func TestStreamModel_StreamResetEvent(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("attempt1")},
			llm.StreamResetEvent{Attempt: 1, Reason: "retrying"},
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("attempt2")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}

	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall: %v", i, types[i], exp, types)
		}
	}
}

func TestStreamModel_StreamResetEventWithReasoning(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart("think1")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewTextPart("text1")},
			llm.StreamResetEvent{Attempt: 1, Reason: "retrying"},
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("text2")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeReasoningStart, typeReasoningDelta,
		typeReasoningEnd,
		typeTextStart, typeTextDelta,
		typeTextEnd,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}

	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall: %v", i, types[i], exp, types)
		}
	}
}

func TestHandler_TextContentConcatenatesAllParts(t *testing.T) {
	t.Parallel()

	msg := chatMessage{
		Role: "assistant",
		Parts: []messagePart{
			{Type: "text", Text: "Hello "},
			{Type: "step-start"},
			{Type: "text", Text: "World"},
		},
	}

	if got := msg.textContent(); got != "Hello World" {
		t.Errorf("textContent() = %q, want 'Hello World'", got)
	}
}

func TestHandler_ErrorResponseSanitized(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.StreamEndEvent{Error: errors.New("secret internal error: db password is foo")},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil)

	chunks, _ := parseSSE(t, rec.Body)

	for _, c := range chunks {
		if c["type"] == typeError {
			et := chunkStr(t, c, "errorText")
			if strings.Contains(et, "secret") || strings.Contains(et, "password") {
				t.Errorf("error text leaks internals: %q", et)
			}

			if et != sanitizedError {
				t.Errorf("error text = %q, want %q", et, sanitizedError)
			}
		}
	}
}

func TestStreamModel_ToolCall(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("Let me check the weather.")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewToolRequestPart("call-1", "getWeather", json.RawMessage(`{"location":"San Francisco"}`))},
			llm.ContentPartEvent{Index: 2, Part: llm.NewToolResponsePart("call-1", "getWeather", json.RawMessage(`{"temp":72,"condition":"sunny"}`))},
			llm.ContentPartEvent{Index: 3, Part: llm.NewTextPart("It's 72F and sunny!")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("weather?"))},
	}, ew, nil)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeTextStart, typeTextDelta, typeTextEnd,
		"tool-input-start", "tool-input-available",
		"tool-output-available",
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v\nwant       = %v", types, expected)
	}

	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall: %v", i, types[i], exp, types)
		}
	}

	for _, c := range chunks {
		if c["type"] == "tool-input-start" {
			if c["toolCallId"] != "call-1" {
				t.Errorf("tool-input-start toolCallId = %v, want call-1", c["toolCallId"])
			}

			if c["toolName"] != "getWeather" {
				t.Errorf("tool-input-start toolName = %v, want getWeather", c["toolName"])
			}
		}
	}

	for _, c := range chunks {
		if c["type"] == "tool-input-available" {
			input, ok := c["input"].(map[string]any)
			if !ok {
				t.Fatalf("tool-input-available input is not map: %T", c["input"])
			}

			if input["location"] != "San Francisco" {
				t.Errorf("input.location = %v, want San Francisco", input["location"])
			}
		}
	}

	for _, c := range chunks {
		if c["type"] == "tool-output-available" {
			output, ok := c["output"].(map[string]any)
			if !ok {
				t.Fatalf("tool-output-available output is not map: %T", c["output"])
			}

			if output["condition"] != "sunny" {
				t.Errorf("output.condition = %v, want sunny", output["condition"])
			}
		}
	}
}

func TestStreamModel_ToolCallError(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewToolRequestPart("call-2", "failTool", json.RawMessage(`{}`))},
			llm.ContentPartEvent{Index: 1, Part: llm.NewToolErrorPart("call-2", "failTool", "tool execution failed")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("fail"))},
	}, ew, nil)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		"tool-input-start", "tool-input-available",
		"tool-output-error",
		typeFinishStep, typeFinish,
	}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v\nwant       = %v", types, expected)
	}

	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] = %q, want %q", i, types[i], exp)
		}
	}

	for _, c := range chunks {
		if c["type"] == "tool-output-error" {
			if c["errorText"] != "tool execution failed" {
				t.Errorf("errorText = %v, want 'tool execution failed'", c["errorText"])
			}

			if c["toolCallId"] != "call-2" {
				t.Errorf("toolCallId = %v, want call-2", c["toolCallId"])
			}
		}
	}
}
