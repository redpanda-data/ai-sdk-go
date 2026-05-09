package uimessagestream

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

func TestHandler_SimpleTextResponse(t *testing.T) {
	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("Hello, world!", fakellm.StreamConfig{ChunkSize: 100})

	h := Handler(model)

	body := `{"id":"chat-1","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	// Verify headers
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	if v := rec.Header().Get("X-Vercel-AI-UI-Message-Stream"); v != "v1" {
		t.Errorf("X-Vercel-AI-UI-Message-Stream = %q, want v1", v)
	}

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	types := chunkTypes(chunks)

	// Verify the exact sequence: start → start-step → text-start → text-delta(s) → text-end → finish-step → finish
	expectedPrefix := []string{"start", "start-step", "text-start"}
	for i, exp := range expectedPrefix {
		if i >= len(types) || types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall types: %v", i, types[i], exp, types)
		}
	}

	// All middle chunks should be text-delta
	for i := 3; i < len(types)-3; i++ {
		if types[i] != "text-delta" {
			t.Errorf("chunk[%d] type = %q, want text-delta", i, types[i])
		}
	}

	expectedSuffix := []string{"text-end", "finish-step", "finish"}
	for i, exp := range expectedSuffix {
		idx := len(types) - 3 + i
		if idx < 0 || idx >= len(types) || types[idx] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall types: %v", idx, types[idx], exp, types)
		}
	}

	// Verify text content by concatenating deltas
	var text strings.Builder
	for _, c := range chunks {
		if c["type"] == "text-delta" {
			text.WriteString(c["delta"].(string))
		}
	}
	if got := text.String(); got != "Hello, world!" {
		t.Errorf("assembled text = %q, want %q", got, "Hello, world!")
	}

	// Verify text-start and text-end have matching IDs
	var startID, endID string
	for _, c := range chunks {
		if c["type"] == "text-start" {
			startID = c["id"].(string)
		}
		if c["type"] == "text-end" {
			endID = c["id"].(string)
		}
	}
	if startID != endID {
		t.Errorf("text-start id = %q, text-end id = %q, want match", startID, endID)
	}

	// Verify finish has finishReason
	for _, c := range chunks {
		if c["type"] == "finish" {
			if reason, ok := c["finishReason"].(string); !ok || reason != "stop" {
				t.Errorf("finish.finishReason = %v, want 'stop'", c["finishReason"])
			}
		}
	}
}

func TestHandler_StreamingTextResponse(t *testing.T) {
	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("Streaming works!", fakellm.StreamConfig{ChunkSize: 4})

	h := Handler(model)

	body := `{"id":"chat-2","messages":[{"role":"user","content":"test"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	var text strings.Builder
	deltaCount := 0
	for _, c := range chunks {
		if c["type"] == "text-delta" {
			text.WriteString(c["delta"].(string))
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
	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenError(llm.ErrRateLimitExceeded)

	h := Handler(model)

	body := `{"id":"chat-3","messages":[{"role":"user","content":"fail"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	chunks, done := parseSSE(t, rec.Body)
	if !done {
		t.Error("stream not terminated with [DONE]")
	}

	// Verify the exact chunk sequence:
	// start → start-step → error → finish-step → finish(finishReason:"error") → [DONE]
	types := chunkTypes(chunks)
	expected := []string{"start", "start-step", "error", "finish-step", "finish"}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}
	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall types: %v", i, types[i], exp, types)
		}
	}

	// Verify error chunk has sanitized errorText (not raw error)
	if et, ok := chunks[2]["errorText"].(string); !ok || et == "" {
		t.Error("error chunk has empty errorText")
	} else if et != "An error occurred" {
		t.Errorf("error chunk errorText = %q, want sanitized 'An error occurred'", et)
	}

	// Verify finish has finishReason "error"
	if reason, ok := chunks[4]["finishReason"].(string); !ok || reason != "error" {
		t.Errorf("finish.finishReason = %v, want \"error\"", chunks[4]["finishReason"])
	}
}

func TestHandler_SystemPrompt(t *testing.T) {
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
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if len(capturedMessages) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(capturedMessages))
	}
	if capturedMessages[0].Role != llm.RoleSystem {
		t.Errorf("first message role = %q, want system", capturedMessages[0].Role)
	}
	if capturedMessages[0].Content[0].Text != "Be concise." {
		t.Errorf("system prompt = %q, want 'Be concise.'", capturedMessages[0].Content[0].Text)
	}
}

func TestHandler_MultiTurnConversation(t *testing.T) {
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
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
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
	model := fakellm.NewFakeModel()
	h := Handler(model)

	req := httptest.NewRequest(http.MethodGet, "/api/chat", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", rec.Code)
	}
}

func TestHandler_InvalidBody(t *testing.T) {
	model := fakellm.NewFakeModel()
	h := Handler(model)

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader("not json"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rec.Code)
	}
}

func TestHandler_V6PartsFormat(t *testing.T) {
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

	// v6 format: messages use parts array instead of content string
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
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if len(capturedMessages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(capturedMessages))
	}
	if capturedMessages[0].Content[0].Text != "hello from v6" {
		t.Errorf("msg[0] text = %q, want 'hello from v6'", capturedMessages[0].Content[0].Text)
	}
	if capturedMessages[1].Content[0].Text != "hi there" {
		t.Errorf("msg[1] text = %q, want 'hi there'", capturedMessages[1].Content[0].Text)
	}
	if capturedMessages[2].Content[0].Text != "follow up" {
		t.Errorf("msg[2] text = %q, want 'follow up'", capturedMessages[2].Content[0].Text)
	}
}

// errorStreamModel is a minimal llm.Model that yields specific event sequences
// for testing error handling in StreamModel.
type errorStreamModel struct {
	events []llm.Event
}

func (m *errorStreamModel) Name() string                        { return "error-test-model" }
func (m *errorStreamModel) Provider() string                    { return "test" }
func (m *errorStreamModel) Capabilities() llm.ModelCapabilities { return llm.ModelCapabilities{Streaming: true} }
func (m *errorStreamModel) Constraints() llm.ModelConstraints   { return llm.ModelConstraints{} }
func (m *errorStreamModel) Generate(_ context.Context, _ *llm.Request) (*llm.Response, error) {
	return nil, fmt.Errorf("not implemented")
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
	// Test that when StreamEndEvent carries an error, the adapter emits
	// error + finish-step + finish(finishReason:"error")
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
	// start → start-step → text-start → text-delta → error → text-end → finish-step → finish
	expected := []string{"start", "start-step", "text-start", "text-delta", "error", "text-end", "finish-step", "finish"}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}
	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall types: %v", i, types[i], exp, types)
		}
	}

	// Verify error text is sanitized
	for _, c := range chunks {
		if c["type"] == "error" {
			if et, ok := c["errorText"].(string); !ok || et != "An error occurred" {
				t.Errorf("error.errorText = %v, want \"An error occurred\"", c["errorText"])
			}
		}
	}

	// Verify finish has finishReason "error" (NOT "stop")
	finishChunk := chunks[len(chunks)-1]
	if reason, ok := finishChunk["finishReason"].(string); !ok || reason != "error" {
		t.Errorf("finish.finishReason = %v, want \"error\"", finishChunk["finishReason"])
	}
}

func TestHandler_FinishReasonMapping(t *testing.T) {
	tests := []struct {
		reason llm.FinishReason
		want   string
	}{
		{llm.FinishReasonStop, "stop"},
		{llm.FinishReasonLength, "length"},
		{llm.FinishReasonContentFilter, "content-filter"},
		{llm.FinishReasonToolCalls, "tool-calls"},
		{llm.FinishReason("unknown_reason"), "other"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := mapFinishReason(tt.reason); got != tt.want {
				t.Errorf("mapFinishReason(%q) = %q, want %q", tt.reason, got, tt.want)
			}
		})
	}
}

func TestHandler_WithLogger(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("ok", fakellm.StreamConfig{ChunkSize: 100})

	h := Handler(model, WithLogger(logger))

	body := `{"id":"chat-log","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

func TestHandler_SystemRoleMessage(t *testing.T) {
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
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
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

	// Message with empty parts and no content should be skipped
	body := `{"id":"chat-empty","messages":[
		{"role":"user","parts":[{"type":"step-start"}]},
		{"role":"user","content":"real message"}
	]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if len(capturedMessages) != 1 {
		t.Fatalf("expected 1 message (empty skipped), got %d", len(capturedMessages))
	}
	if capturedMessages[0].Content[0].Text != "real message" {
		t.Errorf("msg text = %q, want 'real message'", capturedMessages[0].Content[0].Text)
	}
}

func TestHandler_ContextCancellation(t *testing.T) {
	// Model that cancels context mid-stream then yields an error,
	// simulating a real cancellation scenario.
	ctx, cancel := context.WithCancel(context.Background())

	cancellingModel := &cancellingErrorModel{cancel: cancel}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)
	StreamModel(ctx, cancellingModel, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil)

	// Should have start + start-step but no finish (context cancelled)
	chunks, _ := parseSSE(t, rec.Body)
	types := chunkTypes(chunks)
	if len(types) < 2 || types[0] != "start" || types[1] != "start-step" {
		t.Fatalf("expected at least start + start-step, got %v", types)
	}
	// Should NOT have finish (early return on ctx cancel)
	for _, tp := range types {
		if tp == "finish" {
			t.Error("should not have finish chunk when context is cancelled")
		}
	}
}

func TestHandler_ErrorEventNonTerminal(t *testing.T) {
	// ErrorEvent is non-terminal — stream continues after it
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
	expected := []string{"start", "start-step", "text-start", "text-delta", "error", "text-delta", "text-end", "finish-step", "finish"}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}
	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall: %v", i, types[i], exp, types)
		}
	}

	// Verify the error chunk text
	for _, c := range chunks {
		if c["type"] == "error" {
			if et := c["errorText"].(string); et != "An error occurred" {
				t.Errorf("errorText = %q, want 'An error occurred' (sanitized)", et)
			}
		}
	}

	// Verify assembled text includes both deltas
	var text strings.Builder
	for _, c := range chunks {
		if c["type"] == "text-delta" {
			text.WriteString(c["delta"].(string))
		}
	}
	if got := text.String(); got != "before after" {
		t.Errorf("assembled text = %q, want 'before after'", got)
	}
}

func TestHandler_TextContentPrefersPartsOverContent(t *testing.T) {
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
	msg := chatMessage{
		Role:    "user",
		Content: "legacy content",
	}
	if got := msg.textContent(); got != "legacy content" {
		t.Errorf("textContent() = %q, want 'legacy content'", got)
	}
}

func TestStreamModel_ReasoningTrace(t *testing.T) {
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart(&llm.ReasoningTrace{Text: "thinking..."})},
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
	expected := []string{"start", "start-step", "reasoning-start", "reasoning-delta", "reasoning-end", "text-start", "text-delta", "text-end", "finish-step", "finish"}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}
	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall: %v", i, types[i], exp, types)
		}
	}

	// Verify reasoning delta content
	for _, c := range chunks {
		if c["type"] == "reasoning-delta" {
			if d := c["delta"].(string); d != "thinking..." {
				t.Errorf("reasoning-delta = %q, want 'thinking...'", d)
			}
		}
	}
}

func TestStreamModel_ReasoningStatefulTracking(t *testing.T) {
	// Multiple reasoning events should produce one reasoning-start,
	// multiple reasoning-deltas, and one reasoning-end.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart(&llm.ReasoningTrace{Text: "step 1"})},
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart(&llm.ReasoningTrace{Text: " step 2"})},
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart(&llm.ReasoningTrace{Text: " step 3"})},
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
		"start", "start-step",
		"reasoning-start", "reasoning-delta", "reasoning-delta", "reasoning-delta", "reasoning-end",
		"text-start", "text-delta", "text-end",
		"finish-step", "finish",
	}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}
	for i, exp := range expected {
		if types[i] != exp {
			t.Fatalf("chunk[%d] type = %q, want %q\nall: %v", i, types[i], exp, types)
		}
	}

	// Verify all reasoning deltas share the same ID as the start/end
	var rStartID, rEndID string
	var deltaIDs []string
	for _, c := range chunks {
		switch c["type"] {
		case "reasoning-start":
			rStartID = c["id"].(string)
		case "reasoning-end":
			rEndID = c["id"].(string)
		case "reasoning-delta":
			deltaIDs = append(deltaIDs, c["id"].(string))
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

	// Verify concatenated reasoning text
	var reasoning strings.Builder
	for _, c := range chunks {
		if c["type"] == "reasoning-delta" {
			reasoning.WriteString(c["delta"].(string))
		}
	}
	if got := reasoning.String(); got != "step 1 step 2 step 3" {
		t.Errorf("assembled reasoning = %q, want 'step 1 step 2 step 3'", got)
	}
}

func TestStreamModel_ReasoningNilTrace(t *testing.T) {
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart(nil)},
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
	// reasoning-start (no delta since trace is nil) + reasoning-end at StreamEnd
	expected := []string{"start", "start-step", "reasoning-start", "reasoning-end", "finish-step", "finish"}
	if len(types) != len(expected) {
		t.Fatalf("chunk types = %v, want %v", types, expected)
	}
}

type noFlushResponseWriter struct {
	http.ResponseWriter
}

func TestHandler_NoFlusherSupport(t *testing.T) {
	model := fakellm.NewFakeModel()
	h := Handler(model)

	body := `{"id":"nf","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	inner := httptest.NewRecorder()
	rec := &noFlushResponseWriter{inner}

	h.ServeHTTP(rec, req)

	if inner.Code != http.StatusInternalServerError {
		t.Errorf("expected 500, got %d", inner.Code)
	}
}

func TestStreamModel_IteratorErrorWithCancelledContext(t *testing.T) {
	// When iterator yields error AND context is cancelled, handler should return silently
	ctx, cancel := context.WithCancel(context.Background())

	cancellingModel := &cancellingErrorModel{cancel: cancel}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)
	StreamModel(ctx, cancellingModel, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil)

	// Should have start + start-step but no finish (context cancelled)
	chunks, _ := parseSSE(t, rec.Body)
	types := chunkTypes(chunks)
	if len(types) < 2 || types[0] != "start" || types[1] != "start-step" {
		t.Fatalf("expected at least start + start-step, got %v", types)
	}
	// Should NOT have finish (early return on ctx cancel)
	for _, tp := range types {
		if tp == "finish" {
			t.Error("should not have finish chunk when context is cancelled")
		}
	}
}

// cancellingErrorModel cancels the context then returns an error from the iterator
type cancellingErrorModel struct {
	errorStreamModel
	cancel context.CancelFunc
}

func (m *cancellingErrorModel) GenerateEvents(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		m.cancel()
		yield(nil, fmt.Errorf("after cancel"))
	}
}

func TestHandler_AllRequiredHeaders(t *testing.T) {
	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("x", fakellm.StreamConfig{ChunkSize: 100})

	h := Handler(model)

	body := `{"id":"hdr","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	headers := map[string]string{
		"Content-Type":                  "text/event-stream",
		"Cache-Control":                 "no-cache",
		"Connection":                    "keep-alive",
		"X-Vercel-AI-UI-Message-Stream": "v1",
		"X-Accel-Buffering":             "no",
	}
	for k, want := range headers {
		if got := rec.Header().Get(k); got != want {
			t.Errorf("header %q = %q, want %q", k, got, want)
		}
	}
}

func TestHandler_StartChunkHasMessageID(t *testing.T) {
	model := fakellm.NewFakeModel().
		When(fakellm.Any()).
		ThenStreamText("ok", fakellm.StreamConfig{ChunkSize: 100})

	h := Handler(model)

	body := `{"id":"chat-mid","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
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
	if startChunk["type"] != "start" {
		t.Fatalf("first chunk type = %q, want 'start'", startChunk["type"])
	}
	mid, ok := startChunk["messageId"].(string)
	if !ok || mid == "" {
		t.Fatalf("start chunk has no messageId")
	}
	if len(mid) != 16 {
		t.Errorf("messageId length = %d, want 16", len(mid))
	}
}

func TestHandler_RequestBodySizeLimit(t *testing.T) {
	model := fakellm.NewFakeModel()
	h := Handler(model)

	// Create a body larger than 1MB
	bigBody := strings.Repeat("x", 1<<20+1)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(bigBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for oversized body, got %d", rec.Code)
	}
}

func TestHandler_EmptyMessagesArray(t *testing.T) {
	model := fakellm.NewFakeModel()
	h := Handler(model)

	body := `{"id":"chat-none","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty messages, got %d", rec.Code)
	}
}

func TestStreamModel_StreamResetEvent(t *testing.T) {
	// StreamResetEvent should close open text span and reset state,
	// allowing a new text sequence to begin.
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
		"start", "start-step",
		"text-start", "text-delta", "text-end", // first attempt
		"text-start", "text-delta", "text-end",  // second attempt after reset
		"finish-step", "finish",
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
	// StreamResetEvent should also close open reasoning span.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart(&llm.ReasoningTrace{Text: "think1"})},
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
		"start", "start-step",
		"reasoning-start", "reasoning-delta",   // reasoning in first attempt
		"reasoning-end",                        // closed when text part arrives
		"text-start", "text-delta",             // text in first attempt
		"text-end",                             // reset closes text
		"text-start", "text-delta", "text-end", // second attempt
		"finish-step", "finish",
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
	// Verify that StreamEndEvent errors are sanitized (no raw error text leaked)
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
		if c["type"] == "error" {
			et := c["errorText"].(string)
			if strings.Contains(et, "secret") || strings.Contains(et, "password") {
				t.Errorf("error text leaks internals: %q", et)
			}
			if et != "An error occurred" {
				t.Errorf("error text = %q, want 'An error occurred'", et)
			}
		}
	}
}
