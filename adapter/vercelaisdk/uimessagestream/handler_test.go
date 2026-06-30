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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

const (
	typeStart              = "start"
	typeStartStep          = "start-step"
	typeTextStart          = "text-start"
	typeTextDelta          = "text-delta"
	typeTextEnd            = "text-end"
	typeFinishStep         = "finish-step"
	typeFinish             = "finish"
	typeError              = "error"
	typeReasoningStart     = "reasoning-start"
	typeReasoningDelta     = "reasoning-delta"
	typeReasoningEnd       = "reasoning-end"
	typeToolInputStart     = "tool-input-start"
	typeToolInputAvailable = "tool-input-available"
	typeToolOutputAvail    = "tool-output-available"
	typeToolOutputError    = "tool-output-error"
	typeAbort              = "abort"

	// sanitizedError is the default client-facing error text. It matches the
	// Vercel AI SDK default onError return value ("An error occurred." — with
	// the trailing period).
	sanitizedError = "An error occurred."
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

// partText asserts that p is a *llm.TextPart and returns its text.
func partText(t *testing.T, p llm.Part) string {
	t.Helper()

	tp, ok := p.(*llm.TextPart)
	require.True(t, ok, "expected *llm.TextPart, got %T", p)

	return tp.Text
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

	require.Equal(t, http.StatusOK, rec.Code, "response body: %s", rec.Body.String())
	assert.Equal(t, "text/event-stream", rec.Header().Get("Content-Type"))
	assert.Equal(t, "v1", rec.Header().Get("X-Vercel-Ai-Ui-Message-Stream"))

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expectedPrefix := []string{typeStart, typeStartStep, typeTextStart}
	for i, exp := range expectedPrefix {
		require.Greater(t, len(types), i, "not enough chunks, all types: %v", types)
		require.Equal(t, exp, types[i], "chunk[%d] type mismatch, all types: %v", i, types)
	}

	for i := 3; i < len(types)-3; i++ {
		assert.Equal(t, typeTextDelta, types[i], "chunk[%d] type mismatch", i)
	}

	expectedSuffix := []string{typeTextEnd, typeFinishStep, typeFinish}
	for i, exp := range expectedSuffix {
		idx := len(types) - 3 + i
		require.True(t, idx >= 0 && idx < len(types), "suffix index %d out of range, all types: %v", idx, types)
		require.Equal(t, exp, types[idx], "chunk[%d] type mismatch, all types: %v", idx, types)
	}

	var text strings.Builder

	for _, c := range chunks {
		if c["type"] == typeTextDelta {
			text.WriteString(chunkStr(t, c, "delta"))
		}
	}

	assert.Equal(t, "Hello, world!", text.String(), "assembled text mismatch")

	var startID, endID string

	for _, c := range chunks {
		if c["type"] == typeTextStart {
			startID = chunkStr(t, c, "id")
		}

		if c["type"] == typeTextEnd {
			endID = chunkStr(t, c, "id")
		}
	}

	assert.Equal(t, startID, endID, "text-start id and text-end id should match")

	for _, c := range chunks {
		if c["type"] == typeFinish {
			assert.Equal(t, finishReasonStop, chunkStr(t, c, "finishReason"))
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
	assert.True(t, done, "stream not terminated with [DONE]")

	var text strings.Builder

	deltaCount := 0

	for _, c := range chunks {
		if c["type"] == typeTextDelta {
			text.WriteString(chunkStr(t, c, "delta"))

			deltaCount++
		}
	}

	assert.Equal(t, "Streaming works!", text.String(), "assembled text mismatch")
	assert.GreaterOrEqual(t, deltaCount, 2, "expected multiple text-delta chunks for streaming")
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
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{typeStart, typeStartStep, typeError, typeFinishStep, typeFinish}
	require.Equal(t, expected, types, "chunk types mismatch")

	assert.Equal(t, sanitizedError, chunkStr(t, chunks[2], "errorText"))
	assert.Equal(t, finishReasonError, chunkStr(t, chunks[4], "finishReason"))
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

	require.Len(t, capturedMessages, 2, "expected system + user messages")
	assert.Equal(t, llm.RoleSystem, capturedMessages[0].Role)
	assert.Equal(t, "Be concise.", partText(t, capturedMessages[0].Content[0]))
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

	require.Len(t, capturedMessages, 3)
	assert.Equal(t, llm.RoleUser, capturedMessages[0].Role)
	assert.Equal(t, llm.RoleAssistant, capturedMessages[1].Role)
	assert.Equal(t, llm.RoleUser, capturedMessages[2].Role)
}

func TestHandler_MethodNotAllowed(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	h := Handler(model)

	req := httptest.NewRequestWithContext(context.Background(), http.MethodGet, "/api/chat", nil)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rec.Code)
}

func TestHandler_InvalidBody(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	h := Handler(model)

	req := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/api/chat", strings.NewReader("not json"))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
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

	require.Len(t, capturedMessages, 3)
	assert.Equal(t, "hello from v6", partText(t, capturedMessages[0].Content[0]))
	assert.Equal(t, "hi there", partText(t, capturedMessages[1].Content[0]))
	assert.Equal(t, "follow up", partText(t, capturedMessages[2].Content[0]))
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
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	// Open spans are closed (text-end) BEFORE the error chunk, matching the
	// iterator-error path. The error chunk is never wedged inside an open run.
	expected := []string{typeStart, typeStartStep, typeTextStart, typeTextDelta, typeTextEnd, typeError, typeFinishStep, typeFinish}
	require.Equal(t, expected, types, "chunk types mismatch")

	for _, c := range chunks {
		if c["type"] == typeError {
			assert.Equal(t, sanitizedError, chunkStr(t, c, "errorText"))
		}
	}

	finishChunk := chunks[len(chunks)-1]
	assert.Equal(t, finishReasonError, chunkStr(t, finishChunk, "finishReason"))
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

			assert.Equal(t, tt.want, mapFinishReason(tt.reason))
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

	require.Equal(t, http.StatusOK, rec.Code)
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

	require.Len(t, capturedMessages, 2)
	assert.Equal(t, llm.RoleSystem, capturedMessages[0].Role)
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

	require.Len(t, capturedMessages, 1, "empty message should be skipped")
	assert.Equal(t, "real message", partText(t, capturedMessages[0].Content[0]))
}

func TestHandler_ContextCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Drive cancellation through the full Handler/ServeHTTP path.
	model := &cancellingErrorModel{cancel: cancel}
	h := Handler(model)

	req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/api/chat",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	chunks, done := parseSSE(t, rec.Body)
	assertAbortedSequence(t, chunks, done)
}

// assertAbortedSequence verifies a cancelled stream: start, start-step, an
// abort chunk, [DONE], and no finish chunk.
func assertAbortedSequence(t *testing.T, chunks []Chunk, done bool) {
	t.Helper()

	types := chunkTypes(chunks)
	require.GreaterOrEqual(t, len(types), 2, "expected at least start + start-step, got %v", types)
	require.Equal(t, typeStart, types[0])
	require.Equal(t, typeStartStep, types[1])

	// On context cancellation the adapter emits an abort chunk then terminates
	// with [DONE], mirroring stream-text.ts. No finish chunk is emitted.
	assert.True(t, done, "stream should terminate with [DONE] after abort")
	assert.Contains(t, types, typeAbort, "expected an abort chunk on cancellation")

	for _, tp := range types {
		assert.NotEqual(t, typeFinish, tp, "should not have finish chunk when context is cancelled")
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
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{typeStart, typeStartStep, typeTextStart, typeTextDelta, typeError, typeTextDelta, typeTextEnd, typeFinishStep, typeFinish}
	require.Equal(t, expected, types, "chunk types mismatch")

	for _, c := range chunks {
		if c["type"] == typeError {
			assert.Equal(t, sanitizedError, chunkStr(t, c, "errorText"), "error text should be sanitized")
		}
	}

	var text strings.Builder

	for _, c := range chunks {
		if c["type"] == typeTextDelta {
			text.WriteString(chunkStr(t, c, "delta"))
		}
	}

	assert.Equal(t, "before after", text.String(), "assembled text mismatch")
}

func TestHandler_TextContentPrefersPartsOverContent(t *testing.T) {
	t.Parallel()

	msg := chatMessage{
		Role:    "user",
		Content: "legacy content",
		Parts:   []messagePart{{Type: "text", Text: "parts content"}},
	}

	assert.Equal(t, "parts content", msg.textContent())
}

func TestHandler_TextContentFallsBackToContent(t *testing.T) {
	t.Parallel()

	msg := chatMessage{
		Role:    "user",
		Content: "legacy content",
	}

	assert.Equal(t, "legacy content", msg.textContent())
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
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{typeStart, typeStartStep, typeReasoningStart, typeReasoningDelta, typeReasoningEnd, typeTextStart, typeTextDelta, typeTextEnd, typeFinishStep, typeFinish}
	require.Equal(t, expected, types, "chunk types mismatch")

	for _, c := range chunks {
		if c["type"] == typeReasoningDelta {
			assert.Equal(t, "thinking...", chunkStr(t, c, "delta"))
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
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeReasoningStart, typeReasoningDelta, typeReasoningDelta, typeReasoningDelta, typeReasoningEnd,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

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

	assert.Equal(t, rStartID, rEndID, "reasoning-start id and reasoning-end id should match")

	for i, id := range deltaIDs {
		assert.Equal(t, rStartID, id, "reasoning-delta[%d] id mismatch", i)
	}

	var reasoning strings.Builder

	for _, c := range chunks {
		if c["type"] == typeReasoningDelta {
			reasoning.WriteString(chunkStr(t, c, "delta"))
		}
	}

	assert.Equal(t, "step 1 step 2 step 3", reasoning.String(), "assembled reasoning mismatch")
}

func TestStreamModel_ReasoningNilTrace(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: (*llm.ReasoningPart)(nil)},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("think"))},
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	// A nil reasoning part still opens and closes the span, with no delta.
	expected := []string{typeStart, typeStartStep, typeReasoningStart, typeReasoningEnd, typeFinishStep, typeFinish}
	require.Equal(t, expected, types, "chunk types mismatch")
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

	assert.Equal(t, http.StatusInternalServerError, inner.Code)
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
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assertAbortedSequence(t, chunks, done)
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
		assert.Equal(t, want, rec.Header().Get(k), "header %q mismatch", k)
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
	assert.True(t, done, "stream not terminated with [DONE]")

	require.NotEmpty(t, chunks, "no chunks")

	startChunk := chunks[0]
	require.Equal(t, typeStart, startChunk["type"])

	mid := chunkStr(t, startChunk, "messageId")
	require.NotEmpty(t, mid, "start chunk has no messageId")
	assert.Len(t, mid, 16)
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

	assert.Equal(t, http.StatusBadRequest, rec.Code, "oversized body should be rejected")
}

func TestHandler_EmptyMessagesArray(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	h := Handler(model)

	body := `{"id":"chat-none","messages":[]}`
	req := newPostRequest(t, body)

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code, "empty messages should be rejected")
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
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")
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
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

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
	require.Equal(t, expected, types, "chunk types mismatch")
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

	assert.Equal(t, "Hello World", msg.textContent())
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
	}, ew, nil, nil)

	chunks, _ := parseSSE(t, rec.Body)

	for _, c := range chunks {
		if c["type"] == typeError {
			et := chunkStr(t, c, "errorText")
			assert.NotContains(t, et, "secret", "error text leaks internals")
			assert.NotContains(t, et, "password", "error text leaks internals")
			assert.Equal(t, sanitizedError, et)
		}
	}
}

func TestStreamModel_ToolCall(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("Let me check the weather.")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewToolRequestPart("call-1", "getWeather", json.RawMessage(`{"location":"San Francisco"}`))},
			llm.ContentPartEvent{Index: 2, Part: llm.NewToolResponsePart("call-1", "getWeather", json.RawMessage(`{"temp":72,"condition":"sunny"}`), false)},
			llm.ContentPartEvent{Index: 3, Part: llm.NewTextPart("It's 72F and sunny!")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("weather?"))},
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeToolInputStart, typeToolInputAvailable,
		typeToolOutputAvail,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

	for _, c := range chunks {
		if c["type"] == typeToolInputStart {
			assert.Equal(t, "call-1", c["toolCallId"])
			assert.Equal(t, "getWeather", c["toolName"])
		}
	}

	for _, c := range chunks {
		if c["type"] == typeToolInputAvailable {
			input, ok := c["input"].(map[string]any)
			require.True(t, ok, "tool-input-available input is not map: %T", c["input"])
			assert.Equal(t, "San Francisco", input["location"])
		}
	}

	for _, c := range chunks {
		if c["type"] == typeToolOutputAvail {
			output, ok := c["output"].(map[string]any)
			require.True(t, ok, "tool-output-available output is not map: %T", c["output"])
			assert.Equal(t, "sunny", output["condition"])
		}
	}
}

func TestStreamModel_ToolCallError(t *testing.T) {
	t.Parallel()

	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewToolRequestPart("call-2", "failTool", json.RawMessage(`{}`))},
			llm.ContentPartEvent{Index: 1, Part: llm.NewToolResponsePart("call-2", "failTool", json.RawMessage("tool execution failed"), true)},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("fail"))},
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeToolInputStart, typeToolInputAvailable,
		typeToolOutputError,
		typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

	for _, c := range chunks {
		if c["type"] == typeToolOutputError {
			assert.Equal(t, "tool execution failed", c["errorText"])
			assert.Equal(t, "call-2", c["toolCallId"])
		}
	}
}

func TestStreamModel_IteratorErrorClosesSpans(t *testing.T) {
	t.Parallel()

	// When the iterator yields an error mid-stream, open text/reasoning spans
	// must be closed before the error/finish-step/finish sequence.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("partial")},
		},
	}

	// Wrap with an iterator that yields the events then an error.
	iterErrModel := &iteratorErrorModel{
		inner: model.events,
		err:   errors.New("network blip"),
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), iterErrModel, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	// text-end must appear before error to close the open span.
	expected := []string{
		typeStart, typeStartStep,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeError, typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")
}

// iteratorErrorModel yields events then a terminal error from the iterator.
type iteratorErrorModel struct {
	errorStreamModel

	inner []llm.Event
	err   error
}

func (m *iteratorErrorModel) GenerateEvents(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, e := range m.inner {
			if !yield(e, nil) {
				return
			}
		}

		yield(nil, m.err)
	}
}

func TestStreamModel_ReasoningIDsIncrement(t *testing.T) {
	t.Parallel()

	// reasoning → text → reasoning should produce distinct reasoning IDs.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart("think1")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewTextPart("middle")},
			llm.ContentPartEvent{Index: 2, Part: llm.NewReasoningPart("think2")},
			llm.ContentPartEvent{Index: 3, Part: llm.NewTextPart("final")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("think"))},
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeReasoningStart, typeReasoningDelta, typeReasoningEnd,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeReasoningStart, typeReasoningDelta, typeReasoningEnd,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

	// Collect reasoning-start IDs and verify they differ.
	var reasoningIDs []string

	for _, c := range chunks {
		if c["type"] == typeReasoningStart {
			reasoningIDs = append(reasoningIDs, chunkStr(t, c, "id"))
		}
	}

	require.Len(t, reasoningIDs, 2, "expected 2 reasoning-start chunks")
	assert.NotEqual(t, reasoningIDs[0], reasoningIDs[1], "reasoning IDs should differ")
}

func TestStreamModel_TextIDsIncrementAcrossToolCalls(t *testing.T) {
	t.Parallel()

	// text → tool → text should produce distinct text IDs.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("before tool")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewToolRequestPart("call-1", "lookup", json.RawMessage(`{}`))},
			llm.ContentPartEvent{Index: 2, Part: llm.NewToolResponsePart("call-1", "lookup", json.RawMessage(`"ok"`), false)},
			llm.ContentPartEvent{Index: 3, Part: llm.NewTextPart("after tool")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
	}, ew, nil, nil)

	chunks, _ := parseSSE(t, rec.Body)

	// Collect text-start IDs and verify they differ.
	var textIDs []string

	for _, c := range chunks {
		if c["type"] == typeTextStart {
			textIDs = append(textIDs, chunkStr(t, c, "id"))
		}
	}

	require.Len(t, textIDs, 2, "expected 2 text-start chunks")
	assert.NotEqual(t, textIDs[0], textIDs[1], "text IDs should differ")
}

func TestStreamModel_ReasoningThenToolRequest(t *testing.T) {
	t.Parallel()

	// reasoning → tool-request must close the reasoning span before emitting tool events.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewReasoningPart("thinking about tool")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewToolRequestPart("call-1", "lookup", json.RawMessage(`{}`))},
			llm.ContentPartEvent{Index: 2, Part: llm.NewToolResponsePart("call-1", "lookup", json.RawMessage(`"found"`), false)},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeReasoningStart, typeReasoningDelta, typeReasoningEnd,
		typeToolInputStart, typeToolInputAvailable,
		typeToolOutputAvail,
		typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")
}

func TestHandler_WithMaxBodyBytes(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	h := Handler(model, WithMaxBodyBytes(50))

	// Valid JSON structure that exceeds the 50-byte limit.
	body := `{"id":"x","messages":[{"role":"user","content":"this payload is definitely longer than fifty bytes"}]}`
	req := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code, "body exceeding custom limit should be rejected")
}

func TestGenerateMessageID_Length(t *testing.T) {
	t.Parallel()

	id := generateMessageID()
	assert.Len(t, id, 16, "generateMessageID() value = %q", id)
}

// toolCallModel returns tool calls on the first N turns, then stops.
type toolCallModel struct {
	errorStreamModel

	turnsWithTools int
	callCount      int
}

func (m *toolCallModel) GenerateEvents(_ context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
	m.callCount++
	turn := m.callCount

	return func(yield func(llm.Event, error) bool) {
		if turn <= m.turnsWithTools {
			if !yield(llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("calling tool")}, nil) {
				return
			}

			if !yield(llm.ContentPartEvent{Index: 1, Part: llm.NewToolRequestPart("call-1", "lookup", json.RawMessage(`{}`))}, nil) {
				return
			}

			yield(llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonToolCalls}}, nil)

			return
		}

		if !yield(llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("done")}, nil) {
			return
		}

		yield(llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}}, nil)
	}
}

func TestStreamModelWithTools_HappyPath(t *testing.T) {
	t.Parallel()

	model := &toolCallModel{turnsWithTools: 1}

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`"result"`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 0, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart,
		// Turn 1: tool call
		typeStartStep, typeTextStart, typeTextDelta, typeTextEnd,
		typeToolInputStart, typeToolInputAvailable,
		typeFinishStep,
		typeToolOutputAvail,
		// Turn 2: final answer
		typeStartStep, typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep,
		typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

	finishChunk := chunks[len(chunks)-1]
	assert.Equal(t, finishReasonStop, chunkStr(t, finishChunk, "finishReason"))
}

func TestStreamModelWithTools_MaxTurnsExhaustion(t *testing.T) {
	t.Parallel()

	// Model always returns tool calls — will exhaust maxTurns.
	model := &toolCallModel{turnsWithTools: 100}

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`"ok"`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 2, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	// Count paired start-step/finish-step.
	startSteps := 0
	finishSteps := 0

	for _, tp := range types {
		switch tp {
		case typeStartStep:
			startSteps++
		case typeFinishStep:
			finishSteps++
		}
	}

	assert.Equal(t, 2, startSteps, "expected 2 start-step chunks")
	assert.Equal(t, 2, finishSteps, "expected 2 finish-step chunks")

	// Exactly one finish chunk.
	finishCount := 0

	for _, tp := range types {
		if tp == typeFinish {
			finishCount++
		}
	}

	assert.Equal(t, 1, finishCount, "expected 1 finish chunk")

	// On maxTurns exhaustion the adapter surfaces the last turn's real finish
	// reason (the model kept calling tools, so "tool-calls"), matching the
	// reference rather than emitting a synthetic "other".
	finishChunk := chunks[len(chunks)-1]
	assert.Equal(t, "tool-calls", chunkStr(t, finishChunk, "finishReason"))
}

func TestStreamModelWithTools_IteratorErrorNoDuplicateTerminal(t *testing.T) {
	t.Parallel()

	// Model yields text then iterator error — exercises the streamToolTurn error
	// path and verifies the caller doesn't emit a second finish+[DONE].
	model := &iteratorErrorModel{
		inner: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("partial")},
		},
		err: errors.New("network blip"),
	}

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`"ok"`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 0, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	// Exactly one finish chunk.
	finishCount := 0

	for _, c := range chunks {
		if c["type"] == typeFinish {
			finishCount++
		}
	}

	require.Equal(t, 1, finishCount, "expected exactly 1 finish chunk")

	// Verify it's an error finish.
	for _, c := range chunks {
		if c["type"] == typeFinish {
			assert.Equal(t, finishReasonError, chunkStr(t, c, "finishReason"))
		}
	}
}

func TestStreamModelWithTools_IteratorErrorPairsToolChunks(t *testing.T) {
	t.Parallel()

	// When a tool-input has been emitted and then the iterator errors,
	// the orphaned tool-input must get a matching tool-output-error.
	model := &iteratorErrorModel{
		inner: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewToolRequestPart("call-orphan", "lookup", json.RawMessage(`{}`))},
		},
		err: errors.New("connection reset"),
	}

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`"ok"`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 0, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeToolInputStart, typeToolInputAvailable,
		typeToolOutputError,
		typeError, typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

	// Verify the tool-output-error references the right call.
	for _, c := range chunks {
		if c["type"] == typeToolOutputError {
			assert.Equal(t, "call-orphan", c["toolCallId"])
		}
	}
}

func TestStreamModelWithTools_StreamEndEventErrorPairsToolChunks(t *testing.T) {
	t.Parallel()

	// StreamEndEvent{Error} after tool-input has been emitted must pair
	// the orphaned tool-input with a tool-output-error.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewToolRequestPart("call-orphan", "lookup", json.RawMessage(`{}`))},
			llm.StreamEndEvent{Error: errors.New("message too long")},
		},
	}

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`"ok"`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 0, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeToolInputStart, typeToolInputAvailable,
		typeError, typeFinishStep,
		typeToolOutputError,
		typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

	for _, c := range chunks {
		if c["type"] == typeToolOutputError {
			assert.Equal(t, "call-orphan", c["toolCallId"])
		}
	}
}

func TestStreamModelWithTools_StreamEndEventWithError(t *testing.T) {
	t.Parallel()

	// StreamEndEvent{Error: ...} (as opposed to iterator error) must also
	// produce exactly one finish+[DONE]. This exercises the writeToolTurnEnd
	// path where the caller emits finish, not streamToolTurn itself.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("partial")},
			llm.StreamEndEvent{Error: errors.New("provider exploded")},
		},
	}

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`"ok"`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 0, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	finishCount := 0

	for _, c := range chunks {
		if c["type"] == typeFinish {
			finishCount++
		}
	}

	require.Equal(t, 1, finishCount, "expected exactly 1 finish chunk")

	for _, c := range chunks {
		if c["type"] == typeFinish {
			assert.Equal(t, finishReasonError, chunkStr(t, c, "finishReason"))
		}
	}
}

func TestStreamModelWithTools_StreamResetEvent(t *testing.T) {
	t.Parallel()

	// StreamResetEvent during a tool-calling turn must close spans, reset
	// collected tool requests, and not double-fire tools from the discarded attempt.
	executorCalls := 0

	model := &errorStreamModel{
		events: []llm.Event{
			// First attempt: text + tool request, then reset.
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("attempt1")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewToolRequestPart("call-discarded", "lookup", json.RawMessage(`{}`))},
			llm.StreamResetEvent{Attempt: 1, Reason: "retrying"},
			// Second attempt: just text, no tool call.
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("attempt2")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		executorCalls++

		return json.RawMessage(`"ok"`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 0, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	// Tool executor must NOT have been called — the tool request was from the
	// discarded attempt before the reset.
	assert.Equal(t, 0, executorCalls, "executor should not be called for discarded tool requests")

	types := chunkTypes(chunks)

	// Verify text spans are properly closed across the reset, and the
	// discarded tool request gets a tool-output-error to satisfy the
	// protocol pairing invariant.
	expected := []string{
		typeStart,
		typeStartStep,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeToolInputStart, typeToolInputAvailable,
		typeToolOutputError,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

	assert.Equal(t, finishReasonStop, chunkStr(t, chunks[len(chunks)-1], "finishReason"))
}

func TestStreamModel_ToolResponseClosesSpans(t *testing.T) {
	t.Parallel()

	// PartToolResponse should close any open text/reasoning span.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("before")},
			llm.ContentPartEvent{Index: 1, Part: llm.NewToolResponsePart("call-1", "lookup", json.RawMessage(`"found"`), false)},
			llm.ContentPartEvent{Index: 2, Part: llm.NewTextPart("after")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
	}, ew, nil, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeToolOutputAvail,
		typeTextStart, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

	// Verify text IDs differ across the tool response boundary.
	var textIDs []string

	for _, c := range chunks {
		if c["type"] == typeTextStart {
			textIDs = append(textIDs, chunkStr(t, c, "id"))
		}
	}

	require.Len(t, textIDs, 2)
	assert.NotEqual(t, textIDs[0], textIDs[1], "text IDs should differ across tool response")
}

func TestStreamModelWithTools_ErrorEventNonTerminal(t *testing.T) {
	t.Parallel()

	// ErrorEvent in tool-calling mode must be forwarded to the wire,
	// not silently dropped.
	model := &errorStreamModel{
		events: []llm.Event{
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("before")},
			llm.ErrorEvent{Message: "recoverable warning"},
			llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart(" after")},
			llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
		},
	}

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`"ok"`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 0, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	types := chunkTypes(chunks)

	expected := []string{
		typeStart, typeStartStep,
		typeTextStart, typeTextDelta, typeError, typeTextDelta, typeTextEnd,
		typeFinishStep, typeFinish,
	}
	require.Equal(t, expected, types, "chunk types mismatch")

	for _, c := range chunks {
		if c["type"] == typeError {
			assert.Equal(t, sanitizedError, chunkStr(t, c, "errorText"))
		}
	}
}

func TestStreamModelWithTools_RequestFieldsForwarded(t *testing.T) {
	t.Parallel()

	// Verify that Options, ResponseFormat, and Metadata from the original
	// request are forwarded to subsequent model turns, not dropped.
	type testOptions struct {
		MaxTokens int
	}

	var capturedOptions []any

	// Model that returns tool call on first turn, stop on second.
	callCount := 0
	model := &callbackModel{
		fn: func(_ context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
			callCount++

			capturedOptions = append(capturedOptions, req.Options)

			return func(yield func(llm.Event, error) bool) {
				if callCount == 1 {
					if !yield(llm.ContentPartEvent{Index: 0, Part: llm.NewToolRequestPart("call-1", "lookup", json.RawMessage(`{}`))}, nil) {
						return
					}

					yield(llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonToolCalls}}, nil)

					return
				}

				if !yield(llm.ContentPartEvent{Index: 0, Part: llm.NewTextPart("done")}, nil) {
					return
				}

				yield(llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}}, nil)
			}
		},
	}

	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`"ok"`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	opts := testOptions{MaxTokens: 1000}

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
		Options:  opts,
	}, ew, nil, executor, 0, nil)

	require.Len(t, capturedOptions, 2, "expected 2 model calls")
	assert.Equal(t, opts, capturedOptions[0], "turn 1 should have options")
	assert.Equal(t, opts, capturedOptions[1], "turn 2 should have options")
}

// callbackModel delegates GenerateEvents to a callback function.
type callbackModel struct {
	errorStreamModel

	fn func(context.Context, *llm.Request) iter.Seq2[llm.Event, error]
}

func (m *callbackModel) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return m.fn(ctx, req)
}

// ── Conformance invariant checkers ──────────────────────────────────
//
// These mirror the strict UIMessageChunk schema (ui-message-chunks.ts) and the
// client parse state machine (process-ui-message-stream.ts) from ai@7.0.6, so
// Go-level tests can prove conformance without spinning up the JS client.

var validFinishReasons = map[string]bool{
	"stop": true, "length": true, "content-filter": true,
	"tool-calls": true, "error": true, "other": true,
}

type chunkKeySpec struct {
	required []string
	optional []string
}

// emittedChunkSchema lists the keys the strict schema permits for every chunk
// type this adapter emits. "type" is always allowed.
var emittedChunkSchema = map[string]chunkKeySpec{
	typeStart:              {optional: []string{"messageId", "messageMetadata"}},
	typeStartStep:          {},
	typeFinishStep:         {},
	typeFinish:             {optional: []string{"finishReason", "messageMetadata"}},
	typeTextStart:          {required: []string{"id"}, optional: []string{"providerMetadata"}},
	typeTextDelta:          {required: []string{"id", "delta"}, optional: []string{"providerMetadata"}},
	typeTextEnd:            {required: []string{"id"}, optional: []string{"providerMetadata"}},
	typeReasoningStart:     {required: []string{"id"}, optional: []string{"providerMetadata"}},
	typeReasoningDelta:     {required: []string{"id", "delta"}, optional: []string{"providerMetadata"}},
	typeReasoningEnd:       {required: []string{"id"}, optional: []string{"providerMetadata"}},
	typeToolInputStart:     {required: []string{"toolCallId", "toolName"}, optional: []string{"providerExecuted", "providerMetadata", "toolMetadata", "dynamic", "title"}},
	typeToolInputAvailable: {required: []string{"toolCallId", "toolName", "input"}, optional: []string{"providerExecuted", "providerMetadata", "toolMetadata", "dynamic", "title"}},
	typeToolOutputAvail:    {required: []string{"toolCallId", "output"}, optional: []string{"providerExecuted", "providerMetadata", "toolMetadata", "dynamic", "preliminary"}},
	typeToolOutputError:    {required: []string{"toolCallId", "errorText"}, optional: []string{"providerExecuted", "providerMetadata", "toolMetadata", "dynamic"}},
	typeError:              {required: []string{"errorText"}},
	typeAbort:              {optional: []string{"reason"}},
}

// assertSchemaValid checks every chunk against the strict schema: known type,
// all required keys present, no key outside required+optional.
func assertSchemaValid(t *testing.T, chunks []Chunk) {
	t.Helper()

	for i, c := range chunks {
		typ, _ := c["type"].(string)

		spec, ok := emittedChunkSchema[typ]
		require.Truef(t, ok, "chunk %d has unknown type %q", i, typ)

		allowed := map[string]bool{"type": true}

		for _, k := range spec.required {
			allowed[k] = true

			_, present := c[k]
			assert.Truef(t, present, "chunk %d (%s) missing required key %q", i, typ, k)
		}

		for _, k := range spec.optional {
			allowed[k] = true
		}

		for k := range c {
			assert.Truef(t, allowed[k], "chunk %d (%s) has key %q not permitted by the strict schema", i, typ, k)
		}

		if typ == typeFinish {
			if fr, present := c["finishReason"]; present {
				s, _ := fr.(string)
				assert.Truef(t, validFinishReasons[s], "chunk %d finishReason %q not in enum", i, s)
			}
		}
	}
}

// assertOrderingValid replays the client parse state machine: every text and
// reasoning delta/end needs a matching start since the last finish-step (which
// resets active text/reasoning parts); every tool-output needs a prior
// tool-input creating the invocation (tool parts persist across steps).
func assertOrderingValid(t *testing.T, chunks []Chunk) {
	t.Helper()

	activeText := map[string]bool{}
	activeReasoning := map[string]bool{}
	toolParts := map[string]bool{}

	str := func(c Chunk, k string) string { s, _ := c[k].(string); return s }

	for i, c := range chunks {
		typ, _ := c["type"].(string)
		switch typ {
		case typeTextStart:
			activeText[str(c, "id")] = true
		case typeTextDelta:
			assert.Truef(t, activeText[str(c, "id")], "chunk %d text-delta for id %q without active text-start", i, str(c, "id"))
		case typeTextEnd:
			assert.Truef(t, activeText[str(c, "id")], "chunk %d text-end for id %q without active text-start", i, str(c, "id"))
			delete(activeText, str(c, "id"))
		case typeReasoningStart:
			activeReasoning[str(c, "id")] = true
		case typeReasoningDelta:
			assert.Truef(t, activeReasoning[str(c, "id")], "chunk %d reasoning-delta for id %q without active reasoning-start", i, str(c, "id"))
		case typeReasoningEnd:
			assert.Truef(t, activeReasoning[str(c, "id")], "chunk %d reasoning-end for id %q without active reasoning-start", i, str(c, "id"))
			delete(activeReasoning, str(c, "id"))
		case typeToolInputStart, typeToolInputAvailable:
			toolParts[str(c, "toolCallId")] = true
		case typeToolOutputAvail, typeToolOutputError:
			assert.Truef(t, toolParts[str(c, "toolCallId")], "chunk %d %s for toolCallId %q without prior tool-input", i, typ, str(c, "toolCallId"))
		case typeFinishStep:
			activeText = map[string]bool{}
			activeReasoning = map[string]bool{}
		}
	}
}

// assertConformant runs both the schema and ordering invariant checks.
func assertConformant(t *testing.T, chunks []Chunk) {
	t.Helper()
	assertSchemaValid(t, chunks)
	assertOrderingValid(t, chunks)
}

func streamModelChunks(t *testing.T, events []llm.Event, onError ErrorMapper) []Chunk {
	t.Helper()

	model := &errorStreamModel{events: events}
	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModel(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}, ew, nil, onError)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")

	return chunks
}

func TestStreamModel_AllEmittedChunksAreSchemaValid(t *testing.T) {
	t.Parallel()

	stop := llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}}

	cases := []struct {
		name   string
		events []llm.Event
	}{
		{
			name:   "text",
			events: []llm.Event{llm.ContentPartEvent{Part: llm.NewTextPart("hello")}, stop},
		},
		{
			name: "reasoning then text",
			events: []llm.Event{
				llm.ContentPartEvent{Part: llm.NewReasoningPart("thinking")},
				llm.ContentPartEvent{Part: llm.NewTextPart("answer")},
				stop,
			},
		},
		{
			name: "tool request and response",
			events: []llm.Event{
				llm.ContentPartEvent{Part: llm.NewToolRequestPart("c1", "lookup", json.RawMessage(`{"q":"x"}`))},
				llm.ContentPartEvent{Part: llm.NewToolResponsePart("c1", "lookup", json.RawMessage(`{"ok":true}`), false)},
				stop,
			},
		},
		{
			name:   "error",
			events: []llm.Event{llm.ContentPartEvent{Part: llm.NewTextPart("partial")}, llm.StreamEndEvent{Error: errors.New("boom")}},
		},
		{
			name: "every finish reason",
			events: []llm.Event{
				llm.ContentPartEvent{Part: llm.NewTextPart("x")},
				llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonContentFilter}},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assertConformant(t, streamModelChunks(t, tc.events, nil))
		})
	}
}

func TestStreamModelWithTools_AllEmittedChunksAreSchemaValid(t *testing.T) {
	t.Parallel()

	model := &toolCallModel{turnsWithTools: 1}
	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"ok":true}`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 5, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done, "stream not terminated with [DONE]")
	assertConformant(t, chunks)
}

func TestWithOnError_SurfacesCustomErrorText(t *testing.T) {
	t.Parallel()

	chunks := streamModelChunks(t, []llm.Event{
		llm.StreamEndEvent{Error: errors.New("rate limit exceeded")},
	}, func(err error) string { return err.Error() })

	var got string

	for _, c := range chunks {
		if c["type"] == typeError {
			got = chunkStr(t, c, "errorText")
		}
	}

	assert.Equal(t, "rate limit exceeded", got, "custom mapper should surface the real error")
}

func TestWithOnError_DefaultSanitizes(t *testing.T) {
	t.Parallel()

	chunks := streamModelChunks(t, []llm.Event{
		llm.StreamEndEvent{Error: errors.New("sensitive internal detail")},
	}, nil)

	for _, c := range chunks {
		if c["type"] == typeError {
			assert.Equal(t, sanitizedError, chunkStr(t, c, "errorText"), "default mapper should sanitize")
			assert.NotContains(t, chunkStr(t, c, "errorText"), "sensitive")
		}
	}
}

func TestStreamModelWithTools_ToolErrorTextRoutedThroughOnError(t *testing.T) {
	t.Parallel()

	run := func(onError ErrorMapper) string {
		model := &toolCallModel{turnsWithTools: 1}
		executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("db connection failed")
		}

		rec := httptest.NewRecorder()
		ew := NewEventWriter(rec)

		StreamModelWithTools(context.Background(), model, &llm.Request{
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
			Tools:    []llm.ToolDefinition{{Name: "lookup"}},
		}, ew, nil, executor, 5, onError)

		chunks, _ := parseSSE(t, rec.Body)
		assertConformant(t, chunks)

		for _, c := range chunks {
			if c["type"] == typeToolOutputError {
				return chunkStr(t, c, "errorText")
			}
		}

		return ""
	}

	assert.Equal(t, sanitizedError, run(nil), "default should sanitize tool error text")
	assert.Equal(t, "db connection failed", run(func(err error) string { return err.Error() }), "custom mapper should surface tool error text")
}

func TestStreamModel_ToolOutputNullWhenEmpty(t *testing.T) {
	t.Parallel()

	chunks := streamModelChunks(t, []llm.Event{
		llm.ContentPartEvent{Part: llm.NewToolRequestPart("c1", "lookup", json.RawMessage(`{}`))},
		llm.ContentPartEvent{Part: llm.NewToolResponsePart("c1", "lookup", nil, false)}, // empty Result
		llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
	}, nil)

	assertConformant(t, chunks)

	var found bool

	for _, c := range chunks {
		if c["type"] == typeToolOutputAvail {
			found = true
			v, present := c["output"]
			assert.True(t, present, "output key must be present even when empty")
			assert.Nil(t, v, "empty tool result should serialize as JSON null")
		}
	}

	assert.True(t, found, "expected a tool-output-available chunk")
}

func TestStreamModel_ReasoningTextInterleaving(t *testing.T) {
	t.Parallel()

	chunks := streamModelChunks(t, []llm.Event{
		llm.ContentPartEvent{Part: llm.NewReasoningPart("think1")},
		llm.ContentPartEvent{Part: llm.NewTextPart("answer")},
		llm.ContentPartEvent{Part: llm.NewReasoningPart("think2")},
		llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
	}, nil)

	assertConformant(t, chunks)

	// Reasoning resuming after text within the same step must use a NEW id.
	var reasoningStartIDs []string

	for _, c := range chunks {
		if c["type"] == typeReasoningStart {
			reasoningStartIDs = append(reasoningStartIDs, chunkStr(t, c, "id"))
		}
	}

	require.Len(t, reasoningStartIDs, 2, "expected two reasoning spans")
	assert.Equal(t, "reasoning-0", reasoningStartIDs[0])
	assert.Equal(t, "reasoning-1", reasoningStartIDs[1])
	assert.NotEqual(t, reasoningStartIDs[0], reasoningStartIDs[1])
}

func TestStreamModel_EmptyTextDeltaIsValid(t *testing.T) {
	t.Parallel()

	chunks := streamModelChunks(t, []llm.Event{
		llm.ContentPartEvent{Part: llm.NewTextPart("")},
		llm.StreamEndEvent{Response: &llm.Response{FinishReason: llm.FinishReasonStop}},
	}, nil)

	assertConformant(t, chunks)

	for _, c := range chunks {
		if c["type"] == typeTextDelta {
			assert.Empty(t, chunkStr(t, c, "delta"), "empty delta is permitted by the schema")
		}
	}
}

func TestStreamModelWithTools_MultiStepResetsTextID(t *testing.T) {
	t.Parallel()

	// turnsWithTools:1 -> turn 1 emits "calling tool" text + a tool call, turn 2
	// emits "done" text. The turn-2 text run must start with a fresh text-start
	// (different id) after the turn-1 finish-step reset.
	model := &toolCallModel{turnsWithTools: 1}
	executor := func(_ context.Context, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"ok":true}`), nil
	}

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	StreamModelWithTools(context.Background(), model, &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
		Tools:    []llm.ToolDefinition{{Name: "lookup"}},
	}, ew, nil, executor, 5, nil)

	chunks, done := parseSSE(t, rec.Body)
	assert.True(t, done)
	assertConformant(t, chunks)

	var textStartIDs []string

	for _, c := range chunks {
		if c["type"] == typeTextStart {
			textStartIDs = append(textStartIDs, chunkStr(t, c, "id"))
		}
	}

	require.Len(t, textStartIDs, 2, "expected one text run per step")
	assert.NotEqual(t, textStartIDs[0], textStartIDs[1], "text ids must reset across steps")
}

func TestWriteChunk_DoesNotHTMLEscape(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	require.NoError(t, ew.WriteChunk(Chunk{"type": typeTextDelta, "id": "text-0", "delta": "if a < b && c > d"}))

	body := rec.Body.String()
	assert.Contains(t, body, `"delta":"if a < b && c > d"`, "markup must be emitted verbatim, matching JSON.stringify")
	// HTML escaping (Go's default) would replace <, >, & with <, >,
	// &. Disabling it keeps the bytes identical to JSON.stringify.
	assert.NotContains(t, body, "\\u003c", "< must not be escaped")
	assert.NotContains(t, body, "\\u003e", "> must not be escaped")
	assert.NotContains(t, body, "\\u0026", "& must not be escaped")
}

func TestConvertMessages_ReconstructsToolHistory(t *testing.T) {
	t.Parallel()

	msgs := []chatMessage{
		{Role: "user", Parts: []messagePart{{Type: "text", Text: "weather in SF?"}}},
		{Role: "assistant", Parts: []messagePart{
			{Type: "step-start"},
			{
				Type:       "tool-getWeather",
				ToolCallID: "call-1",
				State:      "output-available",
				Input:      json.RawMessage(`{"city":"SF"}`),
				Output:     json.RawMessage(`{"temp":"72F"}`),
			},
			{Type: "step-start"},
			{Type: "text", Text: "It is 72F in SF."},
		}},
		{Role: "user", Parts: []messagePart{{Type: "text", Text: "thanks"}}},
	}

	out := convertMessages(msgs, "")

	// user, assistant(tool-call), user(tool-result), assistant(text), user
	require.Len(t, out, 5)

	assert.Equal(t, llm.RoleUser, out[0].Role)
	assert.Equal(t, "weather in SF?", out[0].TextContent())

	require.Equal(t, llm.RoleAssistant, out[1].Role)
	require.Len(t, out[1].Content, 1)
	req, ok := out[1].Content[0].(*llm.ToolRequestPart)
	require.True(t, ok, "expected *llm.ToolRequestPart, got %T", out[1].Content[0])
	assert.Equal(t, "call-1", req.ID)
	assert.Equal(t, "getWeather", req.Name)
	assert.JSONEq(t, `{"city":"SF"}`, string(req.Arguments))

	require.Equal(t, llm.RoleUser, out[2].Role)
	require.Len(t, out[2].Content, 1)
	resp, ok := out[2].Content[0].(*llm.ToolResponsePart)
	require.True(t, ok, "expected *llm.ToolResponsePart, got %T", out[2].Content[0])
	assert.False(t, resp.IsError)
	assert.Equal(t, "call-1", resp.ID)
	assert.JSONEq(t, `{"temp":"72F"}`, string(resp.Result))

	assert.Equal(t, llm.RoleAssistant, out[3].Role)
	assert.Equal(t, "It is 72F in SF.", out[3].TextContent())

	assert.Equal(t, llm.RoleUser, out[4].Role)
	assert.Equal(t, "thanks", out[4].TextContent())
}

func TestConvertMessages_ToolErrorAndDynamicTool(t *testing.T) {
	t.Parallel()

	msgs := []chatMessage{
		{Role: "assistant", Parts: []messagePart{
			{
				Type:       "dynamic-tool",
				ToolName:   "search",
				ToolCallID: "c2",
				State:      "output-error",
				Input:      json.RawMessage(`{"q":"x"}`),
				ErrorText:  "search backend down",
			},
		}},
	}

	out := convertMessages(msgs, "")

	require.Len(t, out, 2, "assistant tool-call + user tool-result(error)")
	req, ok := out[0].Content[0].(*llm.ToolRequestPart)
	require.True(t, ok, "expected *llm.ToolRequestPart, got %T", out[0].Content[0])
	assert.Equal(t, "search", req.Name)

	resp, ok := out[1].Content[0].(*llm.ToolResponsePart)
	require.True(t, ok, "expected *llm.ToolResponsePart, got %T", out[1].Content[0])
	assert.True(t, resp.IsError, "error tool result should set IsError")
	assert.JSONEq(t, `{"error":"search backend down"}`, string(resp.Result))
}

func TestConvertMessages_SkipsIncompleteToolCalls(t *testing.T) {
	t.Parallel()

	msgs := []chatMessage{
		{Role: "assistant", Parts: []messagePart{
			{Type: "tool-foo", ToolCallID: "c", State: "input-streaming"},
		}},
	}

	assert.Empty(t, convertMessages(msgs, ""), "a still-streaming tool call yields no model message")
}

func TestConvertMessages_CoalescesConsecutiveAssistantSteps(t *testing.T) {
	t.Parallel()

	// Two text-only assistant steps (no intervening tool result) must coalesce
	// into a single assistant message so roles stay strictly alternating.
	msgs := []chatMessage{
		{Role: "user", Parts: []messagePart{{Type: "text", Text: "hi"}}},
		{Role: "assistant", Parts: []messagePart{
			{Type: "step-start"},
			{Type: "text", Text: "a"},
			{Type: "step-start"},
			{Type: "text", Text: "b"},
		}},
		{Role: "user", Parts: []messagePart{{Type: "text", Text: "next"}}},
	}

	out := convertMessages(msgs, "")

	require.Len(t, out, 3)
	assert.Equal(t, llm.RoleUser, out[0].Role)
	assert.Equal(t, llm.RoleAssistant, out[1].Role)
	assert.Equal(t, "ab", out[1].TextContent(), "the two assistant steps should be merged")
	assert.Equal(t, llm.RoleUser, out[2].Role)

	assertRolesAlternate(t, out)
}

func TestConvertMessages_DroppedAssistantTurnDoesNotBreakAlternation(t *testing.T) {
	t.Parallel()

	// An assistant turn that reconstructs to nothing (only a streaming tool call)
	// must not leave two consecutive user messages.
	msgs := []chatMessage{
		{Role: "user", Parts: []messagePart{{Type: "text", Text: "q1"}}},
		{Role: "assistant", Parts: []messagePart{{Type: "tool-x", ToolCallID: "c", State: "input-streaming"}}},
		{Role: "user", Parts: []messagePart{{Type: "text", Text: "q2"}}},
	}

	out := convertMessages(msgs, "")

	require.Len(t, out, 1, "the two user turns should coalesce")
	assert.Equal(t, llm.RoleUser, out[0].Role)
	assert.Equal(t, "q1q2", out[0].TextContent())
}

// assertRolesAlternate verifies no two consecutive messages share a role, which
// providers such as Anthropic require.
func assertRolesAlternate(t *testing.T, msgs []llm.Message) {
	t.Helper()

	for i := 1; i < len(msgs); i++ {
		assert.NotEqualf(t, msgs[i-1].Role, msgs[i].Role, "messages %d and %d must not share a role", i-1, i)
	}
}
