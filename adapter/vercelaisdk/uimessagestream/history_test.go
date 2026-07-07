// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package uimessagestream

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// passthroughErrors surfaces the raw error text, so projection tests can
// assert unwrapping without the default sanitizer in the way.
func passthroughErrors(err error) string { return err.Error() }

func TestProjectUIMessages_MultiStepToolRun(t *testing.T) {
	t.Parallel()

	// One invocation spanning a tool step and an answer step must project as
	// ONE assistant UI message with step-start delimiters — the exact shape
	// useChat assembles from the stream.
	msgs := []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("weather?")),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("c1", "getWeather", []byte(`{"city":"SF"}`))),
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("c1", "getWeather", []byte(`{"temp":"72F"}`), false)),
		llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("It is 72F.")),
	}

	out := projectUIMessages(msgs, passthroughErrors)
	require.Len(t, out, 2)

	assert.Equal(t, "user", out[0].Role)
	assert.Equal(t, []messagePart{{Type: "text", Text: "weather?"}}, out[0].Parts)
	assert.Equal(t, "msg-0", out[0].ID)

	require.Equal(t, "assistant", out[1].Role)
	assert.Equal(t, "msg-1", out[1].ID)
	require.Len(t, out[1].Parts, 4)

	assert.Equal(t, "step-start", out[1].Parts[0].Type)

	tool := out[1].Parts[1]
	assert.Equal(t, "dynamic-tool", tool.Type)
	assert.Equal(t, "getWeather", tool.ToolName)
	assert.Equal(t, "c1", tool.ToolCallID)
	assert.Equal(t, "output-available", tool.State)
	assert.JSONEq(t, `{"city":"SF"}`, string(tool.Input))
	assert.JSONEq(t, `{"temp":"72F"}`, string(tool.Output))

	assert.Equal(t, "step-start", out[1].Parts[2].Type)
	assert.Equal(t, messagePart{Type: "text", Text: "It is 72F.", State: "done"}, out[1].Parts[3])
}

func TestProjectUIMessages_ToolErrorUnwrapped(t *testing.T) {
	t.Parallel()

	msgs := []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("q")),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("c1", "t", []byte(`{}`))),
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("c1", "t", []byte(`{"error":"backend down"}`), true)),
	}

	out := projectUIMessages(msgs, passthroughErrors)
	require.Len(t, out, 2)

	tool := out[1].Parts[1]
	assert.Equal(t, "output-error", tool.State)
	assert.Equal(t, "backend down", tool.ErrorText, "the registry's error envelope must be unwrapped")
}

func TestProjectUIMessages_DanglingToolRequestClosed(t *testing.T) {
	t.Parallel()

	// An interrupted run can persist a tool request with no result; the
	// projection must not hand the client an eternal input-available spinner.
	msgs := []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("q")),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("c1", "t", []byte(`{}`))),
	}

	out := projectUIMessages(msgs, passthroughErrors)
	require.Len(t, out, 2)

	tool := out[1].Parts[1]
	assert.Equal(t, "output-error", tool.State)
	assert.Equal(t, "tool call did not complete", tool.ErrorText)
}

func TestProjectUIMessages_ReasoningBeforeText(t *testing.T) {
	t.Parallel()

	msgs := []llm.Message{
		llm.NewMessage(llm.RoleAssistant, llm.NewReasoningPart("thinking"), llm.NewTextPart("answer")),
	}

	out := projectUIMessages(msgs, passthroughErrors)
	require.Len(t, out, 1)
	require.Len(t, out[0].Parts, 3)
	assert.Equal(t, messagePart{Type: "reasoning", Text: "thinking", State: "done"}, out[0].Parts[1])
	assert.Equal(t, messagePart{Type: "text", Text: "answer", State: "done"}, out[0].Parts[2])
}

func TestProjectUIMessages_RoundTrip(t *testing.T) {
	t.Parallel()

	// What GET returns, convertMessages must reconstruct to the original model
	// messages (modulo reasoning, which inbound conversion drops by design).
	msgs := []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("weather?")),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("c1", "getWeather", []byte(`{"city":"SF"}`))),
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("c1", "getWeather", []byte(`{"temp":"72F"}`), false)),
		llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("It is 72F.")),
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("thanks")),
	}

	projected := projectUIMessages(msgs, passthroughErrors)

	back := make([]chatMessage, 0, len(projected))
	for _, m := range projected {
		back = append(back, chatMessage{Role: m.Role, Parts: m.Parts})
	}

	got := convertMessages(back)

	require.Len(t, got, len(msgs))

	for i := range msgs {
		assert.Equal(t, msgs[i].Role, got[i].Role, "message %d role", i)
		assert.Equal(t, msgs[i].TextContent(), got[i].TextContent(), "message %d text", i)
	}

	req, ok := got[1].Content[0].(*llm.ToolRequestPart)
	require.True(t, ok)
	assert.Equal(t, "c1", req.ID)

	resp, ok := got[2].Content[0].(*llm.ToolResponsePart)
	require.True(t, ok)
	assert.Equal(t, "c1", resp.ID)
	assert.False(t, resp.IsError)
}

func TestHandler_GetHistorySanitizesToolErrors(t *testing.T) {
	t.Parallel()

	// The live stream routes tool error text through onError; a resumed page
	// must not see server-side detail the stream sanitized. GET history goes
	// through the same mapper.
	ctx := context.Background()
	store := session.NewInMemoryStore()
	require.NoError(t, store.Save(ctx, &session.State{ID: "chat-1", Messages: []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("q")),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("c1", "t", []byte(`{}`))),
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("c1", "t", []byte(`{"error":"secret stack trace"}`), true)),
	}}))

	get := func(h http.Handler) string {
		req := httptest.NewRequestWithContext(ctx, http.MethodGet, "/chat-1", nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)

		var resp chatHistoryResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		require.Len(t, resp.Messages, 2)

		tool := resp.Messages[1].Parts[1]
		require.Equal(t, "output-error", tool.State)

		return tool.ErrorText
	}

	assert.Equal(t, defaultErrorText, get(Handler(&sessionEchoAgent{}, store)),
		"default mapper must sanitize resumed tool errors")
	assert.NotContains(t, get(Handler(&sessionEchoAgent{}, store)), "secret")

	custom := Handler(&sessionEchoAgent{}, store, WithOnError(passthroughErrors))
	assert.Equal(t, "secret stack trace", get(custom), "custom mapper sees the unwrapped error")
}

func TestHandler_GetHistory(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := session.NewInMemoryStore()
	h := Handler(&sessionEchoAgent{}, store)

	require.Equal(t, http.StatusOK, postChat(ctx, h, submitBody("chat-1", "hello")).Code)

	req := httptest.NewRequestWithContext(ctx, http.MethodGet, "/chat-1", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, "application/json", rec.Header().Get("Content-Type"))

	var resp struct {
		ID        string      `json:"id"`
		UpdatedAt string      `json:"updatedAt"` //nolint:tagliatelle // wire format
		Messages  []uiMessage `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

	assert.Equal(t, "chat-1", resp.ID)
	assert.NotEmpty(t, resp.UpdatedAt)
	require.Len(t, resp.Messages, 2)
	assert.Equal(t, "user", resp.Messages[0].Role)
	assert.Equal(t, "assistant", resp.Messages[1].Role)
	assert.Equal(t, "msg-0", resp.Messages[0].ID)

	// Absent chat.
	req = httptest.NewRequestWithContext(ctx, http.MethodGet, "/nope", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusNotFound, rec.Code)
}

func TestHandler_DeleteChat(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	h := Handler(&sessionEchoAgent{}, session.NewInMemoryStore())

	require.Equal(t, http.StatusOK, postChat(ctx, h, submitBody("chat-1", "hello")).Code)

	do := func(method, path string) int {
		req := httptest.NewRequestWithContext(ctx, method, path, nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)

		return rec.Code
	}

	assert.Equal(t, http.StatusNoContent, do(http.MethodDelete, "/chat-1"))
	assert.Equal(t, http.StatusNotFound, do(http.MethodGet, "/chat-1"))
	assert.Equal(t, http.StatusNoContent, do(http.MethodDelete, "/chat-1"), "delete is idempotent")
}

func TestHandler_ListChats(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	h := Handler(&sessionEchoAgent{}, session.NewInMemoryStore())

	require.Equal(t, http.StatusOK, postChat(ctx, h, submitBody("chat-a", "one")).Code)
	require.Equal(t, http.StatusOK, postChat(ctx, h, submitBody("chat-b", "two")).Code)

	get := func(path string) (*httptest.ResponseRecorder, chatListResponse) {
		req := httptest.NewRequestWithContext(ctx, http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)

		var resp chatListResponse
		if rec.Code == http.StatusOK {
			require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		}

		return rec, resp
	}

	rec, list := get("/")
	require.Equal(t, http.StatusOK, rec.Code)
	require.Len(t, list.Chats, 2)
	assert.Equal(t, "chat-b", list.Chats[0].ID, "most recently updated first")
	assert.Equal(t, "chat-a", list.Chats[1].ID)

	rec, list = get("/?pageSize=1")
	require.Equal(t, http.StatusOK, rec.Code)
	require.Len(t, list.Chats, 1)
	require.NotEmpty(t, list.NextPageToken)

	rec, list = get("/?pageSize=1&pageToken=" + list.NextPageToken)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Len(t, list.Chats, 1)
	assert.Equal(t, "chat-a", list.Chats[0].ID)

	rec, _ = get("/?pageSize=abc")
	assert.Equal(t, http.StatusBadRequest, rec.Code)
}
