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
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// scriptedAgent is a fake agent.Agent that yields a fixed sequence of events,
// optionally followed by a terminal iterator error. It ignores the invocation.
type scriptedAgent struct {
	events   []agent.Event
	finalErr error
}

func (*scriptedAgent) Info() agent.Info { return agent.Info{Name: "test-agent"} }

func (*scriptedAgent) InputSchema() map[string]any { return nil }

func (s *scriptedAgent) Run(_ context.Context, _ *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		for _, e := range s.events {
			if !yield(e, nil) {
				return
			}
		}

		if s.finalErr != nil {
			yield(nil, s.finalErr)
		}
	}
}

// serveAgent drives AgentHandler with a one-line user message and returns the
// ordered list of decoded chunks plus whether the terminal [DONE] was seen.
func serveAgent(t *testing.T, ag agent.Agent) ([]Chunk, bool) {
	t.Helper()

	body := `{"id":"chat-1","messages":[{"role":"user","parts":[{"type":"text","text":"hi"}]}]}`
	req := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/chat", strings.NewReader(body))
	rec := httptest.NewRecorder()

	AgentHandler(ag).ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "v1", rec.Header().Get("X-Vercel-Ai-Ui-Message-Stream"))

	return parseSSEChunks(t, rec.Body.String())
}

// parseSSEChunks splits an SSE body into decoded JSON chunks, ignoring comment
// (":") lines and recording the terminal [DONE].
func parseSSEChunks(t *testing.T, raw string) ([]Chunk, bool) {
	t.Helper()

	var (
		chunks  []Chunk
		sawDone bool
	)

	for block := range strings.SplitSeq(raw, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" || strings.HasPrefix(block, ":") {
			continue
		}

		data, ok := strings.CutPrefix(block, "data: ")
		if !ok {
			continue
		}

		if data == "[DONE]" {
			sawDone = true
			continue
		}
		var c Chunk
		require.NoError(t, json.Unmarshal([]byte(data), &c), "chunk: %s", data)
		chunks = append(chunks, c)
	}

	return chunks, sawDone
}

func types(chunks []Chunk) []string {
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i], _ = c["type"].(string)
	}

	return out
}

func TestAgentHandler_TextOnly(t *testing.T) {
	t.Parallel()

	ag := &scriptedAgent{events: []agent.Event{
		agent.AssistantDeltaEvent{Delta: llm.ContentPartEvent{Part: llm.NewTextPart("Hello ")}},
		agent.AssistantDeltaEvent{Delta: llm.ContentPartEvent{Part: llm.NewTextPart("world")}},
		agent.MessageEvent{Response: llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Hello world"))}},
		agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop},
	}}

	chunks, sawDone := serveAgent(t, ag)

	assert.True(t, sawDone, "stream must terminate with [DONE]")
	assert.Equal(t, []string{
		"start", "start-step", "text-start", "text-delta", "text-delta", "text-end", "finish-step", "finish",
	}, types(chunks))

	// Streamed text is not re-emitted from the MessageEvent.
	deltas := deltaTexts(chunks)
	assert.Equal(t, []string{"Hello ", "world"}, deltas)

	finish := chunks[len(chunks)-1]
	assert.Equal(t, "stop", finish["finishReason"])
}

func TestAgentHandler_DynamicToolCall(t *testing.T) {
	t.Parallel()

	toolReq := llm.NewToolRequestPart("call-1", "getWeather", json.RawMessage(`{"city":"NYC"}`))
	toolResp := llm.NewToolResponsePart("call-1", "getWeather", json.RawMessage(`{"tempC":21}`), false)

	ag := &scriptedAgent{events: []agent.Event{
		// Turn 1: the model asks for a tool.
		agent.MessageEvent{Response: llm.Response{Message: llm.NewMessage(llm.RoleAssistant, toolReq)}},
		agent.ToolResponseEvent{Response: *toolResp},
		// Turn 2: the model answers.
		agent.AssistantDeltaEvent{Delta: llm.ContentPartEvent{Part: llm.NewTextPart("It is 21C.")}},
		agent.MessageEvent{Response: llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("It is 21C."))}},
		agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop},
	}}

	chunks, sawDone := serveAgent(t, ag)
	assert.True(t, sawDone)

	assert.Equal(t, []string{
		"start",
		"start-step", "tool-input-start", "tool-input-available", "finish-step",
		"tool-output-available",
		"start-step", "text-start", "text-delta", "text-end", "finish-step",
		"finish",
	}, types(chunks))

	// Tool chunks must be tagged dynamic — agent tools are runtime-discovered.
	for _, c := range chunks {
		switch c["type"] {
		case "tool-input-start", "tool-input-available", "tool-output-available":
			assert.Equal(t, true, c["dynamic"], "tool chunk %v must be dynamic", c["type"])
		}
	}

	// The tool-output-available carries the parsed output and no toolName field.
	out := findChunk(chunks, "tool-output-available")
	require.NotNil(t, out)
	assert.Equal(t, "call-1", out["toolCallId"])
	assert.NotContains(t, out, "toolName")
	assert.Equal(t, map[string]any{"tempC": float64(21)}, out["output"])
}

func TestAgentHandler_MaxTurnsEmitsErrorThenFinish(t *testing.T) {
	t.Parallel()

	ag := &scriptedAgent{events: []agent.Event{
		agent.AssistantDeltaEvent{Delta: llm.ContentPartEvent{Part: llm.NewTextPart("working")}},
		agent.MessageEvent{Response: llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("working"))}},
		agent.InvocationEndEvent{FinishReason: agent.FinishReasonMaxTurns},
	}}

	chunks, sawDone := serveAgent(t, ag)
	assert.True(t, sawDone)

	tt := types(chunks)
	// error must come before finish, and finish must be the terminal chunk.
	errIdx := indexOf(tt, "error")
	finishIdx := indexOf(tt, "finish")

	require.GreaterOrEqual(t, errIdx, 0, "expected an error chunk")
	require.Equal(t, len(tt)-1, finishIdx, "finish must be the last chunk")
	assert.Less(t, errIdx, finishIdx, "error must precede finish")

	assert.Equal(t, "maximum iterations reached", chunks[errIdx]["errorText"])
	assert.Equal(t, "error", chunks[finishIdx]["finishReason"])
}

func TestAgentHandler_TerminalErrorClosesStream(t *testing.T) {
	t.Parallel()

	ag := &scriptedAgent{
		events: []agent.Event{
			agent.AssistantDeltaEvent{Delta: llm.ContentPartEvent{Part: llm.NewTextPart("partial")}},
		},
		finalErr: assert.AnError,
	}

	chunks, sawDone := serveAgent(t, ag)
	assert.True(t, sawDone, "even a mid-stream failure must terminate with [DONE]")

	tt := types(chunks)
	// The open text span/step must be closed before the terminal error+finish.
	assert.Equal(t, "finish", tt[len(tt)-1])
	assert.Contains(t, tt, "text-end")
	assert.Contains(t, tt, "finish-step")
	assert.Contains(t, tt, "error")
	assert.Equal(t, "error", chunks[len(chunks)-1]["finishReason"])
}

func TestAgentHandler_NoInvocationEndStillTerminates(t *testing.T) {
	t.Parallel()

	// A stream that just ends (no InvocationEndEvent, no error) must not leave
	// the client hanging.
	ag := &scriptedAgent{events: []agent.Event{
		agent.MessageEvent{Response: llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("done"))}},
	}}

	chunks, sawDone := serveAgent(t, ag)
	assert.True(t, sawDone)

	tt := types(chunks)
	assert.Equal(t, "finish", tt[len(tt)-1])
	assert.Contains(t, tt, "error")
}

func TestConvertMessages_DropsIncompleteToolCall(t *testing.T) {
	t.Parallel()

	msgs := []chatMessage{
		{Role: "user", Parts: []messagePart{{Type: "text", Text: "hi"}}},
		{Role: "assistant", Parts: []messagePart{
			// A completed prior call — kept, with its result.
			{Type: "tool-getX", ToolCallID: "done", State: "output-available", Input: []byte(`{}`), Output: []byte(`{"ok":true}`)},
			// An unresolved call (aborted turn re-sent by the client) — must be dropped:
			// no bare tool_use without a paired tool_result, and no forged call the
			// agent's crash-recovery path could execute.
			{Type: "tool-getY", ToolCallID: "pending", State: "input-available", Input: []byte(`{"a":1}`)},
		}},
		{Role: "user", Parts: []messagePart{{Type: "text", Text: "again"}}},
	}

	var reqIDs []string

	for _, m := range convertMessages(msgs, "") {
		for _, p := range m.Content {
			if tr, ok := p.(*llm.ToolRequestPart); ok {
				reqIDs = append(reqIDs, tr.ID)
			}
		}
	}

	assert.Equal(t, []string{"done"}, reqIDs, "completed call kept, incomplete call dropped")
}

func TestAgentHandler_ToolErrorSanitized(t *testing.T) {
	t.Parallel()

	toolReq := llm.NewToolRequestPart("c1", "boom", []byte(`{}`))
	toolErr := llm.NewToolResponsePart("c1", "boom", []byte(`secret server-side stack trace`), true)

	ag := &scriptedAgent{events: []agent.Event{
		agent.MessageEvent{Response: llm.Response{Message: llm.NewMessage(llm.RoleAssistant, toolReq)}},
		agent.ToolResponseEvent{Response: *toolErr},
		agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop},
	}}

	chunks, _ := serveAgent(t, ag)

	e := findChunk(chunks, "tool-output-error")
	require.NotNil(t, e)
	// Default onError sanitizes; the raw tool error must not reach the client.
	assert.Equal(t, defaultErrorText, e["errorText"])
	assert.Equal(t, true, e["dynamic"])
}

func TestAgentHandler_LengthFinishIsNotAnError(t *testing.T) {
	t.Parallel()

	ag := &scriptedAgent{events: []agent.Event{
		agent.MessageEvent{Response: llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("partial"))}},
		agent.InvocationEndEvent{FinishReason: agent.FinishReasonLength},
	}}

	chunks, _ := serveAgent(t, ag)

	assert.NotContains(t, types(chunks), "error", "length is a partial completion, not an error")
	assert.Equal(t, "length", chunks[len(chunks)-1]["finishReason"])
}

func TestAgentHandler_RejectsNonPost(t *testing.T) {
	t.Parallel()

	req := httptest.NewRequestWithContext(context.Background(), http.MethodGet, "/chat", nil)
	rec := httptest.NewRecorder()
	AgentHandler(&scriptedAgent{}).ServeHTTP(rec, req)
	assert.Equal(t, http.StatusMethodNotAllowed, rec.Code)
}

// --- helpers ---

func deltaTexts(chunks []Chunk) []string {
	var out []string

	for _, c := range chunks {
		if c["type"] == "text-delta" {
			if d, ok := c["delta"].(string); ok {
				out = append(out, d)
			}
		}
	}

	return out
}

func findChunk(chunks []Chunk, typ string) Chunk {
	for _, c := range chunks {
		if c["type"] == typ {
			return c
		}
	}

	return nil
}

func indexOf(ss []string, target string) int {
	for i, s := range ss {
		if s == target {
			return i
		}
	}

	return -1
}
