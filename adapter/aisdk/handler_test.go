package aisdk

import (
	"bufio"
	"encoding/json"
	"io"
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

	hasError := false
	for _, c := range chunks {
		if c["type"] == "error" {
			hasError = true
			if et, ok := c["errorText"].(string); !ok || et == "" {
				t.Error("error chunk has empty errorText")
			}
		}
	}
	if !hasError {
		t.Error("expected error chunk in stream")
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

func TestHandler_FinishReasonMapping(t *testing.T) {
	tests := []struct {
		reason llm.FinishReason
		want   string
	}{
		{llm.FinishReasonStop, "stop"},
		{llm.FinishReasonLength, "length"},
		{llm.FinishReasonContentFilter, "content-filter"},
		{llm.FinishReasonToolCalls, "tool-calls"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := mapFinishReason(tt.reason); got != tt.want {
				t.Errorf("mapFinishReason(%q) = %q, want %q", tt.reason, got, tt.want)
			}
		})
	}
}
