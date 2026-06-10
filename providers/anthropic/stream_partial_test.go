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

package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
)

// TestStreaming_MaxTokensMidToolUse reproduces the Anthropic wire sequence
// that corrupts session state when parallel tool use hits max_tokens: the
// second tool_use block accumulates a partial input_json_delta and the
// stream ends with stop_reason=max_tokens without a closing
// content_block_stop. Before the fix, the provider emitted a ToolRequest
// carrying five bytes of truncated JSON (`{"q":`). After the fix, the
// partial block is dropped and FinishReasonLength propagates up so the
// caller can react.
func TestStreaming_MaxTokensMidToolUse(t *testing.T) {
	t.Parallel()

	sse := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_ok","name":"query","input":{}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"SELECT 1\"}"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_broken","name":"query","input":{}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":30}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/messages", r.URL.Path)

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sse))

		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer srv.Close()

	provider, err := anthropic.NewProvider(
		"test-key",
		anthropic.WithBaseURL(srv.URL),
		anthropic.WithHTTPClient(srv.Client()),
	)
	require.NoError(t, err)

	model, err := provider.NewModel(anthropic.ModelClaudeOpus45)
	require.NoError(t, err)

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("run it")}},
		},
	}

	var (
		toolRequests []*llm.ToolRequestPart
		streamEnd    *llm.StreamEndEvent
	)

	for event, err := range model.GenerateEvents(context.Background(), req) {
		require.NoError(t, err)

		switch e := event.(type) {
		case llm.ContentPartEvent:
			if tr, ok := e.Part.(*llm.ToolRequestPart); ok {
				toolRequests = append(toolRequests, tr)
			}
		case llm.StreamEndEvent:
			end := e
			streamEnd = &end
		}
	}

	require.Len(t, toolRequests, 1, "partial tool_use must not be emitted as a ContentPartEvent")
	assert.Equal(t, "toolu_ok", toolRequests[0].ID)

	var input map[string]any

	require.NoError(t, json.Unmarshal(toolRequests[0].Arguments, &input),
		"surviving tool_use must carry valid JSON arguments")
	assert.Equal(t, "SELECT 1", input["q"])

	require.NotNil(t, streamEnd)
	require.NotNil(t, streamEnd.Response)

	var finalToolIDs []string

	for _, part := range streamEnd.Response.Message.Content {
		if tr, ok := part.(*llm.ToolRequestPart); ok {
			finalToolIDs = append(finalToolIDs, tr.ID)
			assert.True(t, json.Valid(tr.Arguments),
				"tool_use %s in final response carried invalid JSON: %q", tr.ID, tr.Arguments)
		}
	}

	assert.Equal(t, []string{"toolu_ok"}, finalToolIDs,
		"partial tool_use must not leak into the final response")

	// Truncation signal must survive through to the caller so the agent loop
	// can decide what to do (retry with higher max_tokens, surface to user,
	// etc). Without the fix, hasToolCalls override clobbers this to
	// FinishReasonToolCalls and callers never see that the turn was cut short.
	assert.Equal(t, llm.FinishReasonLength, streamEnd.Response.FinishReason,
		"max_tokens must propagate as FinishReasonLength even when tool calls are present")
}
