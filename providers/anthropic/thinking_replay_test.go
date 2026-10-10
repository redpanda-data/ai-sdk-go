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
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Claude Opus 5.5 and Sonnet 5.5 think by default with display "omitted":
// every thinking block arrives with an empty thinking field and only a
// signature. Anthropic requires thinking and redacted_thinking blocks to be
// passed back unchanged within a tool-use loop.

func TestResponseMapper_ThinkingBlocks(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		block anthropic.BetaContentBlockUnion
		want  []llm.Part
	}{
		{
			name:  "omitted thinking keeps its signature",
			block: anthropic.BetaContentBlockUnion{Type: blockTypeThinking, Signature: "sig-omitted"},
			want:  []llm.Part{&llm.ReasoningPart{Signature: "sig-omitted"}},
		},
		{
			name:  "summarized thinking keeps text and signature",
			block: anthropic.BetaContentBlockUnion{Type: blockTypeThinking, Thinking: "step 1", Signature: "sig-summary"},
			want:  []llm.Part{&llm.ReasoningPart{Text: "step 1", Signature: "sig-summary"}},
		},
		{
			name:  "thinking with neither text nor signature is dropped",
			block: anthropic.BetaContentBlockUnion{Type: blockTypeThinking},
			want:  []llm.Part{},
		},
		{
			name:  "redacted thinking keeps its data and is stamped with the provider",
			block: anthropic.BetaContentBlockUnion{Type: "redacted_thinking", Data: "opaque-data"},
			want: []llm.Part{&llm.ReasoningPart{
				Text:      "[redacted thinking]",
				Signature: "opaque-data",
				Metadata:  map[string]any{"redacted": true, "redacted_provider": "anthropic"},
			}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, err := NewResponseMapper().FromProvider(&anthropic.BetaMessage{
				ID:         "msg_thinking",
				Model:      anthropic.Model(ModelClaudeOpus55),
				Content:    []anthropic.BetaContentBlockUnion{tc.block},
				StopReason: anthropic.BetaStopReasonEndTurn,
			})
			require.NoError(t, err)
			assert.Equal(t, tc.want, resp.Message.Content)
		})
	}
}

// TestThinkingBlocksReplayAfterPersistence follows a tool-use turn from the
// provider response through session-style JSON persistence and back onto the
// wire: the replayed assistant message must carry the original blocks.
func TestThinkingBlocksReplayAfterPersistence(t *testing.T) {
	t.Parallel()

	resp, err := NewResponseMapper().FromProvider(&anthropic.BetaMessage{
		ID:    "msg_tool_turn",
		Model: anthropic.Model(ModelClaudeOpus55),
		Content: []anthropic.BetaContentBlockUnion{
			{Type: blockTypeThinking, Signature: "sig-omitted"},
			{Type: "redacted_thinking", Data: "opaque-data"},
			{Type: blockTypeToolUse, ID: "toolu_1", Name: "get_weather", Input: json.RawMessage(`{"city":"Paris"}`)},
		},
		StopReason: anthropic.BetaStopReasonToolUse,
	})
	require.NoError(t, err)

	persisted, err := json.Marshal(resp.Message)
	require.NoError(t, err)

	var restored llm.Message
	require.NoError(t, json.Unmarshal(persisted, &restored))

	apiReq, err := NewRequestMapper(&Config{ModelName: ModelClaudeOpus55, MaxTokens: 1024}).ToProvider(&llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Weather in Paris?")),
			restored,
			llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("toolu_1", "get_weather", json.RawMessage(`"sunny"`), false)),
		},
	})
	require.NoError(t, err)

	assert.JSONEq(t, `[
		{"type":"thinking","thinking":"","signature":"sig-omitted"},
		{"type":"redacted_thinking","data":"opaque-data"},
		{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}
	]`, wireContent(t, apiReq.Messages[1]))
}

// TestRequestMapper_SkipsUnreplayableRedactedThinking covers redacted parts
// with no data, or data this provider is not known to have produced. Sending
// another provider's payload as redacted_thinking is rejected, so the mapper
// drops the block and the turn goes on.
func TestRequestMapper_SkipsUnreplayableRedactedThinking(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		part *llm.ReasoningPart
	}{
		{
			name: "persisted before redacted data was kept",
			part: &llm.ReasoningPart{Text: "[redacted thinking]", Metadata: map[string]any{"redacted": true}},
		},
		{
			name: "persisted before the provider stamp",
			part: &llm.ReasoningPart{Text: "[redacted thinking]", Signature: "opaque-data", Metadata: map[string]any{"redacted": true}},
		},
		{
			// Bedrock stores base64 of the raw Converse redactedContent bytes.
			name: "produced by Bedrock",
			part: &llm.ReasoningPart{
				Text:      "[redacted thinking]",
				Signature: "AP8QeMM=",
				Metadata:  map[string]any{"redacted": true, "redacted_provider": "aws.bedrock"},
			},
		},
		{
			name: "stamped without data",
			part: &llm.ReasoningPart{Text: "[redacted thinking]", Metadata: map[string]any{"redacted": true, "redacted_provider": "anthropic"}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			apiReq, err := NewRequestMapper(&Config{ModelName: ModelClaudeOpus55, MaxTokens: 1024}).ToProvider(&llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")),
					llm.NewMessage(llm.RoleAssistant, tc.part, llm.NewTextPart("hello")),
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("again")),
				},
			})
			require.NoError(t, err)

			assert.JSONEq(t, `[{"type":"text","text":"hello"}]`, wireContent(t, apiReq.Messages[1]))
		})
	}
}

// TestStreaming_OmittedThinkingKeepsSignature feeds the wire sequence of an
// omitted-display thinking block (empty thinking_delta, then signature_delta)
// and a redacted_thinking block, and checks the final response carries both.
func TestStreaming_OmittedThinkingKeepsSignature(t *testing.T) {
	t.Parallel()

	sse := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_stream","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-omitted"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque-data"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":1}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":2}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":30}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sse))
	}))
	defer srv.Close()

	provider, err := NewProvider("test-key", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	require.NoError(t, err)

	model, err := provider.NewModel(ModelClaudeOpus55)
	require.NoError(t, err)

	streamer, ok := model.(llm.EventsGenerator)
	require.True(t, ok, "Anthropic models stream")

	var (
		deltas []llm.Part
		end    *llm.StreamEndEvent
	)

	for event, err := range streamer.GenerateEvents(context.Background(), &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Weather in Paris?"))},
	}) {
		require.NoError(t, err)

		switch e := event.(type) {
		case llm.ContentPartEvent:
			deltas = append(deltas, e.Part)
		case llm.StreamEndEvent:
			end = &e
		}
	}

	require.Len(t, deltas, 1, "an omitted thinking block streams no displayable reasoning")
	assert.IsType(t, &llm.ToolRequestPart{}, deltas[0])

	require.NotNil(t, end)
	require.NotNil(t, end.Response)
	require.Len(t, end.Response.Message.Content, 3)
	assert.Equal(t, &llm.ReasoningPart{Signature: "sig-omitted"}, end.Response.Message.Content[0])
	assert.Equal(t, &llm.ReasoningPart{
		Text:      "[redacted thinking]",
		Signature: "opaque-data",
		Metadata:  map[string]any{"redacted": true, "redacted_provider": "anthropic"},
	}, end.Response.Message.Content[1])
	assert.IsType(t, &llm.ToolRequestPart{}, end.Response.Message.Content[2])
}

// TestRequestMapper_ThinkingBlockBinding pins the request shape for models
// that check replayed thinking against an unchanged prefix: every request
// asks the API to drop a failing block instead of rejecting the request.
// Other models must not gain a thinking config, which would turn thinking on.
func TestRequestMapper_ThinkingBlockBinding(t *testing.T) {
	t.Parallel()

	const bound = `{"type":"adaptive","block_binding":{"prefix_mismatch_behavior":"drop_block"}}`

	bindingBeta := []string{"thinking-binding-controls-2026-08-01"}

	cases := []struct {
		name         string
		model        string
		opts         []Option
		wantModel    string
		wantThinking string // empty: no thinking field on the wire
		wantBetas    []string
	}{
		{name: "Sonnet 5.5 without thinking option", model: ModelClaudeSonnet55, wantThinking: bound, wantBetas: bindingBeta},
		{name: "Sonnet 5.5 with thinking", model: ModelClaudeSonnet55, opts: []Option{WithThinking(true)}, wantThinking: bound, wantBetas: bindingBeta},
		{name: "Opus 5.5", model: ModelClaudeOpus55, wantThinking: bound, wantBetas: bindingBeta},
		{name: "Fable 5.1", model: ModelClaudeFable51, wantThinking: bound, wantBetas: bindingBeta},
		{
			name:         "custom model name keeps the offering's binding",
			model:        ModelClaudeSonnet55,
			opts:         []Option{WithCustomModelName("claude-sonnet-5-5-preview")},
			wantModel:    "claude-sonnet-5-5-preview",
			wantThinking: bound,
			wantBetas:    bindingBeta,
		},
		{name: "opt-out sends no thinking config", model: ModelClaudeOpus55, opts: []Option{WithThinkingBlockBinding(false)}},
		{
			name:         "opt-out keeps requested thinking",
			model:        ModelClaudeSonnet55,
			opts:         []Option{WithThinking(true), WithThinkingBlockBinding(false)},
			wantThinking: `{"type":"adaptive"}`,
		},
		{name: "opting in does not bind other models", model: ModelClaudeSonnet46, opts: []Option{WithThinkingBlockBinding(true)}},
		{name: "Sonnet 4.6 without thinking sends none", model: ModelClaudeSonnet46},
		{name: "Sonnet 4.6 adaptive thinking is not bound", model: ModelClaudeSonnet46, opts: []Option{WithThinking(true)}, wantThinking: `{"type":"adaptive"}`},
		{name: "Opus 5 is not bound", model: ModelClaudeOpus5},
		{
			name:         "explicit budget is left as is",
			model:        ModelClaudeSonnet46,
			opts:         []Option{WithThinkingBudget(2048)},
			wantThinking: `{"type":"enabled","budget_tokens":2048}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			provider, err := NewProvider("test-key")
			require.NoError(t, err)

			model, err := provider.NewModel(tc.model, tc.opts...)
			require.NoError(t, err)

			m, ok := model.(*Model)
			require.True(t, ok)

			apiReq, err := m.requestMapper.ToProvider(&llm.Request{
				Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
			})
			require.NoError(t, err)

			wantModel := tc.wantModel
			if wantModel == "" {
				wantModel = tc.model
			}

			assertThinkingWire(t, apiReq, wantModel, tc.wantThinking)
			assert.Equal(t, tc.wantBetas, apiReq.Betas)
		})
	}
}

// TestRequestMapper_ThinkingBlockBindingSkipsHandBuiltBudget covers the
// defensive budget check in applyThinkingBlockBinding with a Config NewModel
// cannot produce: it rejects WithThinkingBudget for the prefix-checked models.
// block_binding is invalid on enabled thinking, so a manual budget set by hand
// is kept and the request gains nothing.
func TestRequestMapper_ThinkingBlockBindingSkipsHandBuiltBudget(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	_, err = provider.NewModel(ModelClaudeSonnet55, WithThinkingBudget(2048))
	require.Error(t, err, "if NewModel accepts a budget here, the check is no longer defensive")

	budget := int64(2048)

	apiReq, err := NewRequestMapper(&Config{
		ModelName:           ModelClaudeSonnet55,
		MaxTokens:           4096,
		EnableThinking:      true,
		ThinkingBudget:      &budget,
		AdaptiveThinking:    true,
		ThinkingPrefixCheck: true,
	}).ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	})
	require.NoError(t, err)

	assertThinkingWire(t, apiReq, ModelClaudeSonnet55, `{"type":"enabled","budget_tokens":2048}`)
	assert.Empty(t, apiReq.Betas)
}

// TestThinkingBlockBinding_RoundTrip drives a streamed request end to end:
// the binding reaches the wire as body and header, and a block the API
// dropped for a prefix mismatch is reported on the response.
func TestThinkingBlockBinding_RoundTrip(t *testing.T) {
	t.Parallel()

	sse := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_bound","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0},"input_transformations":[{"type":"thinking_dropped","path":"messages.1.content.0","reason":"prefix_binding_mismatch"}]}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	var (
		gotBetas []string
		gotBody  []byte
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBetas = r.Header.Values("anthropic-beta")
		gotBody, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sse))
	}))
	defer srv.Close()

	provider, err := NewProvider("test-key", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	require.NoError(t, err)

	model, err := provider.NewModel(ModelClaudeOpus55)
	require.NoError(t, err)

	streamer, ok := model.(llm.EventsGenerator)
	require.True(t, ok, "Anthropic models stream")

	var end *llm.StreamEndEvent

	for event, err := range streamer.GenerateEvents(context.Background(), &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}) {
		require.NoError(t, err)

		if e, ok := event.(llm.StreamEndEvent); ok {
			end = &e
		}
	}

	assert.Contains(t, gotBetas, "thinking-binding-controls-2026-08-01")

	var body struct {
		Thinking json.RawMessage `json:"thinking"`
	}
	require.NoError(t, json.Unmarshal(gotBody, &body))
	assert.JSONEq(t, `{"type":"adaptive","block_binding":{"prefix_mismatch_behavior":"drop_block"}}`, string(body.Thinking))

	require.NotNil(t, end)
	require.NotNil(t, end.Response)
	assert.Equal(t, []map[string]any{{
		"type":   "thinking_dropped",
		"path":   "messages.1.content.0",
		"reason": "prefix_binding_mismatch",
	}}, end.Response.Raw["input_transformations"])
}

// TestResponseMapper_InputTransformations checks the non-streaming path
// surfaces dropped thinking blocks and stays quiet when nothing was dropped.
func TestResponseMapper_InputTransformations(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
		want any
	}{
		{
			name: "dropped block is reported",
			body: `{"input_transformations":[{"type":"thinking_dropped","path":"messages.3.content.0","reason":"prefix_binding_mismatch"}]}`,
			want: []map[string]any{{"type": "thinking_dropped", "path": "messages.3.content.0", "reason": "prefix_binding_mismatch"}},
		},
		{name: "empty list is not reported", body: `{"input_transformations":[]}`},
		{name: "absent field is not reported", body: `{}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var msg anthropic.BetaMessage
			require.NoError(t, json.Unmarshal([]byte(tc.body), &msg))

			resp, err := NewResponseMapper().FromProvider(&msg)
			require.NoError(t, err)

			if tc.want == nil {
				assert.Nil(t, resp.Raw)
				return
			}

			assert.Equal(t, tc.want, resp.Raw["input_transformations"])
		})
	}
}

// assertThinkingWire checks the serialized request's model and thinking
// fields; an empty wantThinking means the field must be absent.
func assertThinkingWire(t *testing.T, apiReq anthropic.BetaMessageNewParams, wantModel, wantThinking string) {
	t.Helper()

	raw, err := json.Marshal(apiReq)
	require.NoError(t, err)

	var body struct {
		Model    string          `json:"model"`
		Thinking json.RawMessage `json:"thinking"`
	}
	require.NoError(t, json.Unmarshal(raw, &body))

	assert.Equal(t, wantModel, body.Model)

	if wantThinking == "" {
		assert.Empty(t, body.Thinking, "no thinking config may be sent")
		return
	}

	assert.JSONEq(t, wantThinking, string(body.Thinking))
}

// wireContent returns the serialized content array of one API message.
func wireContent(t *testing.T, msg anthropic.BetaMessageParam) string {
	t.Helper()

	raw, err := json.Marshal(msg)
	require.NoError(t, err)

	var body struct {
		Content json.RawMessage `json:"content"`
	}
	require.NoError(t, json.Unmarshal(raw, &body))

	return string(body.Content)
}
