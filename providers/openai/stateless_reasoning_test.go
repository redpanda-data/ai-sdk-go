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

package openai_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

func TestStatelessReasoningToolReplay(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{false, true} {
		for _, summary := range []string{`[{"type":"summary_text","text":"look up x"}]`, `[]`, `[{"type":"summary_text","text":""}]`} {
			t.Run(fmt.Sprintf("stream=%t/summary=%s", streaming, summary), func(t *testing.T) {
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

					output := `[{"type":"message","id":"msg_2","role":"assistant","status":"completed","content":[{"type":"output_text","text":"found","annotations":[]}]}]`
					if calls.Add(1) == 1 {
						output = `[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque-encrypted-reasoning","summary":` + summary + `},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"key\":\"x\"}","status":"completed"}]`
					}

					response := `{"id":"resp_test","object":"response","status":"completed","model":"gpt-5.4-mini","output":` + output + `,"usage":{"input_tokens":12,"output_tokens":8,"total_tokens":20}}`

					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":%s}\n\n", response)
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = fmt.Fprint(w, response)
					}
				}))
				t.Cleanup(server.Close)

				provider, err := openai.NewProvider("placeholder-key", openai.WithBaseURL(server.URL))
				require.NoError(t, err)

				model, err := provider.NewModel(openai.ModelGPT5_4Mini,
					openai.WithStore(false), openai.WithReasoningSummary(openai.ReasoningSummaryDetailed))
				require.NoError(t, err)

				question := llm.NewMessage(llm.RoleUser, llm.NewTextPart("look up x"))
				tools := []llm.ToolDefinition{{Name: "lookup", Parameters: json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}`)}}
				generate := func(request *llm.Request) *llm.Response {
					t.Helper()

					if !streaming {
						response, generateErr := model.Generate(t.Context(), request)
						require.NoError(t, generateErr)

						return response
					}

					var response *llm.Response
					var terminal int

					for event, streamErr := range model.GenerateEvents(t.Context(), request) {
						require.NoError(t, streamErr)

						if end, ok := event.(llm.StreamEndEvent); ok {
							terminal++

							require.NoError(t, end.Error)

							response = end.Response
						}
					}

					require.Equal(t, 1, terminal)
					require.NotNil(t, response)

					return response
				}

				first := generate(&llm.Request{Messages: []llm.Message{question}, Tools: tools})
				assert.Equal(t, llm.FinishReasonToolCalls, first.FinishReason)

				firstBody := <-requests
				assert.Equal(t, false, firstBody["store"])
				assert.Equal(t, []any{"reasoning.encrypted_content"}, firstBody["include"])
				require.Len(t, first.Message.Content, 2)

				reasoning, ok := first.Message.Content[0].(*llm.ReasoningPart)
				require.True(t, ok)
				assert.Equal(t, "rs_1", reasoning.ID)
				assert.Equal(t, "opaque-encrypted-reasoning", reasoning.Signature)

				encoded, err := json.Marshal(first.Message)
				require.NoError(t, err)

				var persisted llm.Message
				require.NoError(t, json.Unmarshal(encoded, &persisted))

				second := generate(&llm.Request{Messages: []llm.Message{
					question,
					persisted,
					llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("call_1", "lookup", json.RawMessage(`"found"`), false)),
				}, Tools: tools})
				assert.Equal(t, "found", second.TextContent())

				secondBody := <-requests
				assert.Equal(t, []any{"reasoning.encrypted_content"}, secondBody["include"])

				input, ok := secondBody["input"].([]any)
				require.True(t, ok)
				require.Len(t, input, 4)

				replayed, ok := input[1].(map[string]any)
				require.True(t, ok)
				assert.Equal(t, "reasoning", replayed["type"])
				assert.Equal(t, "rs_1", replayed["id"])
				assert.Equal(t, "opaque-encrypted-reasoning", replayed["encrypted_content"])

				if summary == `[{"type":"summary_text","text":"look up x"}]` {
					assert.Equal(t, []any{map[string]any{"type": "summary_text", "text": "look up x"}}, replayed["summary"])
				} else {
					assert.Equal(t, []any{}, replayed["summary"])
				}

				assert.Equal(t, map[string]any{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": `{"key":"x"}`}, input[2])
				assert.Equal(t, map[string]any{"type": "function_call_output", "call_id": "call_1", "output": `"found"`}, input[3])
				assert.EqualValues(t, 2, calls.Load())
			})
		}
	}
}

func TestEncryptedReasoningIncludeScope(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		model   string
		compat  bool
		opts    []openai.Option
		include bool
	}{
		{name: "native stateless reasoning", model: openai.ModelGPT5_4Mini, opts: []openai.Option{openai.WithStore(false)}, include: true},
		{name: "native default storage", model: openai.ModelGPT5_4Mini},
		{name: "native stored reasoning", model: openai.ModelGPT5_4Mini, opts: []openai.Option{openai.WithStore(true)}},
		{name: "native stateless nonreasoning", model: openai.ModelGPT41, opts: []openai.Option{openai.WithStore(false)}},
		{name: "foreign stateless reasoning", model: "foreign-model", compat: true, opts: []openai.Option{openai.WithStore(false)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			requests := make(chan map[string]any, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}

				requests <- body

				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"id":"resp_test","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`)
			}))
			t.Cleanup(server.Close)

			provider, err := openai.NewProvider("placeholder-key", openai.WithBaseURL(server.URL))
			require.NoError(t, err)

			var model llm.Model

			if tt.compat {
				model, err = provider.NewCompatModel(tt.model, openai.CompatModelDefinition{Capabilities: llm.ModelCapabilities{Reasoning: true}}, tt.opts...)
			} else {
				model, err = provider.NewModel(tt.model, tt.opts...)
			}

			require.NoError(t, err)

			_, err = model.Generate(t.Context(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello"))}})
			require.NoError(t, err)

			body := <-requests
			if tt.include {
				assert.Equal(t, []any{"reasoning.encrypted_content"}, body["include"])
			} else {
				assert.NotContains(t, body, "include")
			}
		})
	}
}
