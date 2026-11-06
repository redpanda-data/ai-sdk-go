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

package openaicompat_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
)

func TestReasoningHistoryPublicRoundTrip(t *testing.T) {
	t.Parallel()

	for _, enabled := range []bool{false, true} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("enabled=%t/stream=%t", enabled, streaming), func(t *testing.T) {
				t.Parallel()

				var calls atomic.Int32

				requests := make(chan map[string]any, 2)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						http.Error(w, err.Error(), http.StatusBadRequest)
						return
					}

					requests <- body

					first := calls.Add(1) == 1

					if !streaming {
						w.Header().Set("Content-Type", "application/json")

						if first {
							_, _ = fmt.Fprint(w, `{"id":"chat_1","object":"chat.completion","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","reasoning_content":"think first","content":"checking","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"x\"}"}},{"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"y\"}"}}]}}]}`)
						} else {
							_, _ = fmt.Fprint(w, `{"id":"chat_2","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"finished"}}]}`)
						}

						return
					}

					w.Header().Set("Content-Type", "text/event-stream")

					if first {
						for _, delta := range []string{
							`{"reasoning_content":"think "}`,
							`{"reasoning_content":"first"}`,
							`{"content":"checking"}`,
							`{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"y\"}"}}]}`,
							`{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"x\"}"}}]}`,
						} {
							_, _ = fmt.Fprintf(w, "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":%s,\"finish_reason\":null}]}\n\n", delta)
						}

						_, _ = fmt.Fprint(w, "data: "+`{"id":"chat_1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
					} else {
						_, _ = fmt.Fprint(w, "data: "+`{"id":"chat_2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"finished"},"finish_reason":"stop"}]}`+"\n\n")
					}

					_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
				}))
				t.Cleanup(server.Close)

				provider, err := openaicompat.NewProvider("placeholder-key", openaicompat.WithBaseURL(server.URL))
				require.NoError(t, err)

				var options []openaicompat.Option
				if enabled {
					options = append(options, openaicompat.WithReasoning())
				}

				model, err := provider.NewModel("native-reasoning-model", options...)
				require.NoError(t, err)

				generate := func(request *llm.Request) *llm.Response {
					t.Helper()

					if !streaming {
						response, generateErr := model.Generate(t.Context(), request)
						require.NoError(t, generateErr)

						return response
					}

					var response *llm.Response
					var reasoning strings.Builder
					var toolIDs []string
					var terminal int

					for event, streamErr := range model.GenerateEvents(t.Context(), request) {
						require.NoError(t, streamErr)

						switch e := event.(type) {
						case llm.ContentPartEvent:
							switch part := e.Part.(type) {
							case *llm.ReasoningPart:
								reasoning.WriteString(part.Text)
							case *llm.ToolRequestPart:
								toolIDs = append(toolIDs, part.ID)
							}
						case llm.StreamEndEvent:
							terminal++

							require.NoError(t, e.Error)

							response = e.Response
						}
					}

					require.Equal(t, 1, terminal)
					require.NotNil(t, response)

					if response.FinishReason == llm.FinishReasonToolCalls {
						assert.Equal(t, "think first", reasoning.String())
						assert.Equal(t, []string{"call_1", "call_2"}, toolIDs)
					}

					return response
				}

				system := llm.NewMessage(llm.RoleSystem, llm.NewTextPart("use lookup"))
				question := llm.NewMessage(llm.RoleUser, llm.NewTextPart("look up x and y"))
				tools := []llm.ToolDefinition{{Name: "lookup", Parameters: json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}`)}}
				first := generate(&llm.Request{Messages: []llm.Message{system, question}, Tools: tools})
				assert.Equal(t, llm.FinishReasonToolCalls, first.FinishReason)
				require.Len(t, first.Message.Content, 4)

				reasoning, ok := first.Message.Content[0].(*llm.ReasoningPart)
				require.True(t, ok)
				assert.Equal(t, "think first", reasoning.Text)

				first.Message.Content = append(first.Message.Content,
					&llm.ReasoningPart{ID: "foreign-id", Text: "foreign-id-text"},
					&llm.ReasoningPart{Signature: "foreign-signature", Text: "foreign-signature-text"},
					&llm.ReasoningPart{Metadata: map[string]any{"foreign": true}, Text: "foreign-metadata-text"},
				)
				history := []llm.Message{
					system, question, first.Message,
					llm.NewMessage(llm.RoleUser,
						llm.NewToolResponsePart("call_1", "lookup", json.RawMessage(`"first-result"`), false),
						llm.NewToolResponsePart("call_2", "lookup", json.RawMessage(`"second-result"`), false)),
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("now finish")),
				}
				encoded, err := json.Marshal(history)
				require.NoError(t, err)

				var persisted []llm.Message
				require.NoError(t, json.Unmarshal(encoded, &persisted))

				second := generate(&llm.Request{Messages: persisted, Tools: tools})
				assert.Equal(t, "finished", second.TextContent())
				assert.Equal(t, llm.FinishReasonStop, second.FinishReason)

				<-requests

				body := <-requests

				assistant := map[string]any{
					"role": "assistant", "content": "checking",
					"tool_calls": []any{
						map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "lookup", "arguments": `{"key":"x"}`}},
						map[string]any{"id": "call_2", "type": "function", "function": map[string]any{"name": "lookup", "arguments": `{"key":"y"}`}},
					},
				}
				if enabled {
					assistant["reasoning_content"] = "think first"
				}

				assert.Equal(t, []any{
					map[string]any{"role": "system", "content": "use lookup"},
					map[string]any{"role": "user", "content": "look up x and y"},
					assistant,
					map[string]any{"role": "tool", "tool_call_id": "call_1", "content": `"first-result"`},
					map[string]any{"role": "tool", "tool_call_id": "call_2", "content": `"second-result"`},
					map[string]any{"role": "user", "content": "now finish"},
				}, body["messages"])
				assert.EqualValues(t, 2, calls.Load())
			})
		}
	}
}
