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
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestWriteChunk_DoesNotHTMLEscape(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	require.NoError(t, ew.WriteChunk(Chunk{"type": "text-delta", "id": "text-0", "delta": "if a < b && c > d"}))

	body := rec.Body.String()
	assert.Contains(t, body, `"delta":"if a < b && c > d"`, "markup must be emitted verbatim, matching JSON.stringify")
	// HTML escaping (Go's default) would replace <, >, & with their \u00xx
	// escapes. Disabling it keeps the bytes identical to JSON.stringify.
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

	out := convertMessages(msgs)

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

	out := convertMessages(msgs)

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

	assert.Empty(t, convertMessages(msgs), "a still-streaming tool call yields no model message")
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

	out := convertMessages(msgs)

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

	out := convertMessages(msgs)

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
