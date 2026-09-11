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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestNativeToolSearchWire(t *testing.T) {
	t.Parallel()
	const searchCall = `{"type":"server_tool_use","id":"srv_1","name":"tool_search_tool_bm25","input":{"query":"fetch a record"}}`
	const searchResult = `{"type":"tool_search_tool_result","tool_use_id":"srv_1","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"fetch"}]}}`
	const toolCall = `{"type":"tool_use","id":"call_1","name":"fetch","input":{}}`
	blocks := []json.RawMessage{json.RawMessage(searchCall), json.RawMessage(searchResult), json.RawMessage(toolCall)}
	payload, err := json.Marshal(map[string]any{"id": "msg_1", "type": "message", "role": "assistant", "model": ModelClaudeSonnet46, "content": blocks, "stop_reason": "tool_use", "usage": map[string]int{"input_tokens": 25, "output_tokens": 15}})
	require.NoError(t, err)

	for _, streaming := range []bool{false, true} {
		t.Run(strconv.FormatBool(streaming), func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]json.RawMessage
				assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				var tools []map[string]json.RawMessage
				assert.NoError(t, json.Unmarshal(body["tools"], &tools))
				assert.JSONEq(t, `true`, string(tools[0]["defer_loading"]))
				assert.JSONEq(t, `"tool_search_tool_bm25_20251119"`, string(tools[1]["type"]))
				assert.JSONEq(t, `"tool_search_tool_bm25"`, string(tools[1]["name"]))

				if !streaming {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(payload)

					return
				}

				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":25,\"output_tokens\":0}}}\n\n")

				for i, block := range blocks {
					if i == 0 {
						block = json.RawMessage(`{"type":"server_tool_use","id":"srv_1","name":"tool_search_tool_bm25","input":{}}`)
					}

					_, _ = fmt.Fprintf(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":%d,\"content_block\":%s}\n\n", i, block)
					if i == 0 {
						_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\":\\\"fetch a record\\\"}\"}}\n\n")
					}

					_, _ = fmt.Fprintf(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":%d}\n\n", i)
				}

				_, _ = fmt.Fprint(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":15}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
			}))
			defer srv.Close()

			provider, err := NewProvider("test-key", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
			require.NoError(t, err)
			model, err := provider.NewModel(ModelClaudeSonnet46)
			require.NoError(t, err)

			req := &llm.Request{ToolSearch: true, Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("fetch"))}, Tools: []llm.ToolDefinition{
				{Name: "fetch", Description: "Fetch a record", Parameters: json.RawMessage(`{"type":"object","properties":{}}`), Deferred: true},
			}}
			concrete, ok := model.(*Model)
			require.True(t, ok)

			before, err := concrete.requestMapper.ToProvider(req)
			require.NoError(t, err)
			var response *llm.Response

			if streaming {
				var parts []llm.Part

				for event, err := range model.GenerateEvents(t.Context(), req) {
					require.NoError(t, err)

					switch e := event.(type) {
					case llm.ContentPartEvent:
						parts = append(parts, e.Part)
					case llm.StreamEndEvent:
						require.NoError(t, e.Error)
						response = e.Response
					}
				}

				require.Len(t, parts, 3)
				require.IsType(t, &llm.ToolSearchPart{}, parts[0])
				require.IsType(t, &llm.ToolSearchPart{}, parts[1])
			} else {
				response, err = model.Generate(t.Context(), req)
				require.NoError(t, err)
			}

			require.NotNil(t, response)
			require.Len(t, response.ToolRequests(), 1)
			search, ok := response.Message.Content[1].(*llm.ToolSearchPart)
			require.True(t, ok)
			require.Equal(t, []string{"fetch"}, search.Tools)

			req.Messages = append(req.Messages, response.Message, llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("call_1", "fetch", json.RawMessage(`{}`), false)))
			after, err := concrete.requestMapper.ToProvider(req)
			require.NoError(t, err)
			a, err := json.Marshal(before.Tools)
			require.NoError(t, err)
			b, err := json.Marshal(after.Tools)
			require.NoError(t, err)
			require.JSONEq(t, string(a), string(b))

			for i, want := range []string{searchCall, searchResult} {
				raw, err := json.Marshal(after.Messages[1].Content[i])
				require.NoError(t, err)
				require.JSONEq(t, want, string(raw))
			}

			require.Len(t, after.Messages[2].Content, 1, "only client tool calls get tool results")
		})
	}
	// A hosted pause is continuable; an output-token cut must still stop the agent.
	var paused anthropic.BetaMessage
	require.NoError(t, json.Unmarshal(payload, &paused))
	paused.StopReason = anthropic.BetaStopReasonPauseTurn
	response, err := NewResponseMapper().FromProvider(&paused)
	require.NoError(t, err)
	require.Equal(t, llm.FinishReasonToolCalls, response.FinishReason)

	paused.StopReason = anthropic.BetaStopReasonMaxTokens
	response, err = NewResponseMapper().FromProvider(&paused)
	require.NoError(t, err)
	require.Equal(t, llm.FinishReasonLength, response.FinishReason)
}
