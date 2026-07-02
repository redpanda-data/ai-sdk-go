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

package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestMapAssistantMessage_EmptyContentIsRepaired is the mapper-level regression
// for the empty-content replay bug. A max_tokens cut can persist an assistant
// message whose Content slice is empty (its only block was a partial tool_use
// the streaming finalizer dropped). Replaying that history must NOT produce an
// assistant message with an empty content array — Anthropic's Messages API
// rejects it with "messages.N.content: Field required". The mapper substitutes
// a single minimal text block so the outgoing request stays valid and roles
// keep alternating.
func TestMapAssistantMessage_EmptyContentIsRepaired(t *testing.T) {
	t.Parallel()

	m := newWireTestModel(t)

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("run the tools")}},
			// The poisoned turn: assistant message with zero content parts,
			// exactly what a dropped-partial-tool_use max_tokens cut persists.
			{Role: llm.RoleAssistant, Content: []llm.Part{}},
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("continue")}},
		},
	}

	apiReq, err := m.requestMapper.ToProvider(req)
	require.NoError(t, err)

	require.Len(t, apiReq.Messages, 3, "all three history messages must map through")

	for i, msg := range apiReq.Messages {
		assert.NotEmpty(t, msg.Content,
			"message %d (role %s) mapped to an empty content array — Anthropic rejects this with 'content: Field required'",
			i, msg.Role)
	}

	// The repaired assistant turn must carry exactly one text block.
	assistant := apiReq.Messages[1]
	require.Equal(t, anthropic.BetaMessageParamRoleAssistant, assistant.Role)
	require.Len(t, assistant.Content, 1)
	require.NotNil(t, assistant.Content[0].OfText, "repaired block must be a text block")
	assert.NotEmpty(t, assistant.Content[0].OfText.Text, "substituted text block must be non-empty")
}

// TestStreaming_MaxTokensAllBlocksDropped_ReplayStaysValid is the end-to-end
// reproducer. The stream contains a SINGLE tool_use block that is cut off by
// stop_reason=max_tokens before its JSON args finish (no closing
// content_block_stop). The finalizer drops the unparseable partial block, so
// the finalized assistant message ends up with EMPTY content and
// FinishReasonLength. This mirrors providers/anthropic/stream_partial_test.go
// but with NO surviving valid block — the exact shape that poisons the session.
//
// The test then maps that finalized message back through the request mapper
// (as replayed history) and asserts it does NOT yield an empty-content
// assistant message.
func TestStreaming_MaxTokensAllBlocksDropped_ReplayStaysValid(t *testing.T) {
	t.Parallel()

	sse := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_broken","name":"query","input":{}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}` + "\n\n" +
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

	provider, err := NewProvider(
		"test-key",
		WithBaseURL(srv.URL),
		WithHTTPClient(srv.Client()),
	)
	require.NoError(t, err)

	model, err := provider.NewModel(ModelClaudeOpus45)
	require.NoError(t, err)

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("run it")}},
		},
	}

	var streamEnd *llm.StreamEndEvent

	for event, err := range model.GenerateEvents(context.Background(), req) {
		require.NoError(t, err)

		if e, ok := event.(llm.StreamEndEvent); ok {
			end := e
			streamEnd = &end
		}
	}

	require.NotNil(t, streamEnd)
	require.NotNil(t, streamEnd.Response)

	// The finalized turn has empty content (partial tool_use dropped) and the
	// truncation signal survives.
	assert.Empty(t, streamEnd.Response.Message.Content,
		"the sole block was a dropped partial tool_use, so finalized content must be empty")
	assert.Equal(t, llm.FinishReasonLength, streamEnd.Response.FinishReason)

	// Now replay: feed the poisoned assistant turn back as history and confirm
	// the request mapper does not emit an empty-content assistant message.
	replay := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("run it")}},
			streamEnd.Response.Message,
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("try again")}},
		},
	}

	m, ok := model.(*Model)
	require.True(t, ok)

	apiReq, err := m.requestMapper.ToProvider(replay)
	require.NoError(t, err)

	for i, msg := range apiReq.Messages {
		assert.NotEmpty(t, msg.Content,
			"replayed message %d (role %s) mapped to empty content — this is the 400 'content: Field required'",
			i, msg.Role)
	}
}
