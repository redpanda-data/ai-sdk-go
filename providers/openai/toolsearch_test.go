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

package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestNativeToolSearchWire(t *testing.T) {
	t.Parallel()
	const searchCall = `{"type":"tool_search_call","id":"tsc_1","call_id":null,"execution":"server","status":"completed","arguments":{"paths":["svc"]}}`
	const searchOutput = `{"type":"tool_search_output","id":"tso_1","call_id":null,"execution":"server","status":"completed","tools":[{"type":"namespace","name":"svc","description":"Service","tools":[{"type":"function","name":"fetch","description":"Fetch a record","parameters":{"type":"object","properties":{}},"defer_loading":false}]}]}`
	const functionCall = `{"type":"function_call","id":"fc_1","call_id":"call_1","name":"fetch","namespace":"svc","arguments":"{}","status":"completed"}`
	output := []json.RawMessage{json.RawMessage(searchCall), json.RawMessage(searchOutput), json.RawMessage(functionCall)}
	payload, err := json.Marshal(map[string]any{"id": "resp_1", "status": "completed", "output": output, "usage": map[string]any{"input_tokens": 25, "output_tokens": 15, "total_tokens": 40}})
	require.NoError(t, err)

	for _, streaming := range []bool{false, true} {
		t.Run(strconv.FormatBool(streaming), func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]json.RawMessage
				assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				var tools []map[string]json.RawMessage
				assert.NoError(t, json.Unmarshal(body["tools"], &tools))
				assert.JSONEq(t, `"namespace"`, string(tools[0]["type"]))
				assert.JSONEq(t, `"tool_search"`, string(tools[1]["type"]))
				assert.Contains(t, string(tools[0]["tools"]), `"defer_loading":true`)

				if !streaming {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(payload)

					return
				}

				w.Header().Set("Content-Type", "text/event-stream")

				for i, item := range output {
					event, _ := json.Marshal(map[string]any{"type": "response.output_item.done", "output_index": i, "item": item})
					_, _ = fmt.Fprintf(w, "event: response.output_item.done\ndata: %s\n\n", event)
				}

				_, _ = fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":%s}\n\n", payload)
			}))
			defer srv.Close()

			provider, err := NewProvider("test-key", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
			require.NoError(t, err)
			model, err := provider.NewModel(ModelGPT5_4)
			require.NoError(t, err)

			req := &llm.Request{ToolSearch: true, Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("fetch"))}, Tools: []llm.ToolDefinition{
				{Name: "fetch", Description: "Fetch a record", Parameters: json.RawMessage(`{"type":"object","properties":{}}`), Deferred: true, Group: llm.ToolGroup{Name: "svc", Description: "Service"}},
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
			require.Equal(t, "svc", response.ToolRequests()[0].Metadata["openai_namespace"])
			search, ok := response.Message.Content[1].(*llm.ToolSearchPart)
			require.True(t, ok)
			require.Equal(t, []string{"fetch"}, search.Tools)
			require.JSONEq(t, searchOutput, string(search.Data))

			req.Messages = append(req.Messages, response.Message, llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("call_1", "fetch", json.RawMessage(`{}`), false)))
			after, err := concrete.requestMapper.ToProvider(req)
			require.NoError(t, err)
			a, err := json.Marshal(before.Tools)
			require.NoError(t, err)
			b, err := json.Marshal(after.Tools)
			require.NoError(t, err)
			require.JSONEq(t, string(a), string(b), "native discovery keeps the catalog prefix stable")

			inputs := after.Input.OfInputItemList
			require.Len(t, inputs, 5)

			for i, want := range []string{searchCall, searchOutput} {
				raw, err := json.Marshal(inputs[i+1])
				require.NoError(t, err)
				require.JSONEq(t, want, string(raw), "native replay preserves nulls and complete schemas")
			}

			require.Equal(t, "svc", inputs[3].OfFunctionCall.Namespace.Value)
		})
	}
}
