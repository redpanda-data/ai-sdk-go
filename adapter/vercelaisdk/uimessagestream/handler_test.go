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

	if tp, ok := capturedMessages[0].Content[0].(*llm.TextPart); !ok || tp.Text != "Be concise." {
		t.Errorf("text = %q, want 'Be concise.'", tp.Text)
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

	if tp, ok := capturedMessages[0].Content[0].(*llm.TextPart); !ok || tp.Text != "hello from v6" {
		t.Errorf("text = %q, want 'hello from v6'", tp.Text)
	}

	if tp, ok := capturedMessages[1].Content[0].(*llm.TextPart); !ok || tp.Text != "hi there" {
		t.Errorf("text = %q, want 'hi there'", tp.Text)
	}

	if tp, ok := capturedMessages[2].Content[0].(*llm.TextPart); !ok || tp.Text != "follow up" {
		t.Errorf("text = %q, want 'follow up'", tp.Text)
	}
}

// errorStreamModel is a minimal llm.Model that yields specific event sequences
// for testing error handling in StreamModel. After all events are yielded, if
// terminalErr is non-nil it is yielded as (nil, err) to terminate the stream.
type errorStreamModel struct {
	events      []llm.Event
	terminalErr error
}

func (m *errorStreamModel) Name() llm.ModelID        { return "error-test-model" }
func (m *errorStreamModel) Provider() llm.ProviderID { return "test" }

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

		if m.terminalErr != nil {
			yield(nil, m.terminalErr)
		}
	}
}

func TestStreamModel_StreamEndEventWithError(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("partial")},
		},
		terminalErr: errors.New("provider exploded"),
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

	if tp, ok := capturedMessages[0].Content[0].(*llm.TextPart); !ok || tp.Text != "real message" {
		t.Errorf("text = %q, want 'real message'", tp.Text)
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
			llm.ContentPartEvent{Index: 0, Part: &llm.ReasoningPart{Text: "thinking..."}},
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
			llm.ContentPartEvent{Index: 0, Part: &llm.ReasoningPart{Text: "step 1"}},
			llm.ContentPartEvent{Index: 0, Part: &llm.ReasoningPart{Text: " step 2"}},
			llm.ContentPartEvent{Index: 0, Part: &llm.ReasoningPart{Text: " step 3"}},
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
			llm.ContentPartEvent{Index: 0, Part: &llm.ReasoningPart{}},
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
			llm.ContentPartEvent{Index: 0, Part: &llm.ReasoningPart{Text: "think1"}},
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
		terminalErr: errors.New("secret internal error: db password is foo"),
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
			llm.ContentPartEvent{Index: 1, Part: &llm.ToolRequestPart{
				ID:        "call-1",
				Name:      "getWeather",
				Arguments: json.RawMessage(`{"location":"San Francisco"}`),
			}},
			llm.ContentPartEvent{Index: 2, Part: &llm.ToolResponsePart{
				ID:     "call-1",
				Name:   "getWeather",
				Result: json.RawMessage(`{"temp":72,"condition":"sunny"}`),
			}},
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

// requestCapturingModel records each *llm.Request StreamModel passes
// to GenerateEvents so tests can assert the iteration loop carries
// per-request fields (Sampling, ResponseFormat, Metadata) across turns.
type requestCapturingModel struct {
	captured []*llm.Request
	turns    [][]llm.Event
}

func (*requestCapturingModel) Name() llm.ModelID                  { return "capture" }
func (*requestCapturingModel) Provider() llm.ProviderID           { return "test" }
func (*requestCapturingModel) Capabilities() llm.ModelCapabilities { return llm.ModelCapabilities{Streaming: true} }
func (*requestCapturingModel) Constraints() llm.ModelConstraints  { return llm.ModelConstraints{} }
func (*requestCapturingModel) Generate(_ context.Context, _ *llm.Request) (*llm.Response, error) {
	return nil, errors.New("not implemented")
}

func (m *requestCapturingModel) GenerateEvents(_ context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	turn := len(m.captured)

	// Snapshot the request as the handler saw it.
	snap := *req
	m.captured = append(m.captured, &snap)

	return func(yield func(llm.Event, error) bool) {
		if turn >= len(m.turns) {
			return
		}

		for _, e := range m.turns[turn] {
			if !yield(e, nil) {
				return
			}
		}
	}
}

// TestStreamModel_PreservesRequestFieldsAcrossTurns ensures that the
// per-turn iteration request keeps Sampling, ResponseFormat and
// Metadata from the original request rather than rebuilding a stripped
// llm.Request. Without this, tool-loop iterations would drop the
// caller's generation knobs after the first turn.
func TestStreamModel_PreservesRequestFieldsAcrossTurns(t *testing.T) {
	t.Parallel()

	model := &requestCapturingModel{
		turns: [][]llm.Event{
			{
				llm.ContentPartEvent{Index: 0, Part: &llm.ToolRequestPart{
					ID:        "call-1",
					Name:      "noop",
					Arguments: json.RawMessage(`{}`),
				}},
				llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonToolCalls}},
			},
			{
				llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("done")},
				llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
			},
		},
	}

	temp := 0.7

	req := &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
		Sampling: &llm.SamplingParams{Temperature: &temp},
		ResponseFormat: &llm.ResponseFormat{
			Type: llm.ResponseFormatJSONObject,
		},
		Metadata: map[string]string{"trace_id": "abc"},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"ok":true}`), nil
	}

	StreamModelWithTools(context.Background(), model, req, ew, nil, executor)

	if len(model.captured) < 2 {
		t.Fatalf("expected at least 2 turns captured, got %d", len(model.captured))
	}

	for i, cap := range model.captured {
		if cap.Sampling == nil || cap.Sampling.Temperature == nil || *cap.Sampling.Temperature != 0.7 {
			t.Errorf("turn %d: Sampling.Temperature missing or wrong: %+v", i, cap.Sampling)
		}

		if cap.ResponseFormat == nil || cap.ResponseFormat.Type != llm.ResponseFormatJSONObject {
			t.Errorf("turn %d: ResponseFormat missing: %+v", i, cap.ResponseFormat)
		}

		if cap.Metadata["trace_id"] != "abc" {
			t.Errorf("turn %d: Metadata.trace_id missing: %+v", i, cap.Metadata)
		}
	}
}

func TestStreamModel_ToolCallError(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: &llm.ToolRequestPart{
				ID:        "call-2",
				Name:      "failTool",
				Arguments: json.RawMessage(`{}`),
			}},
			llm.ContentPartEvent{Index: 1, Part: &llm.ToolResponsePart{
				ID:    "call-2",
				Name:  "failTool",
				Error: "tool execution failed",
			}},
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
