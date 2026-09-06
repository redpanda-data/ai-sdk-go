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

package meta

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

func TestMuseSparkResponsesTransport(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", streaming), func(t *testing.T) {
			t.Parallel()
			const response = `{"id":"resp_meta","object":"response","status":"completed","model":"muse-spark-1.3","output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello","annotations":[]}]}],"usage":{"input_tokens":100,"output_tokens":20,"total_tokens":120,"input_tokens_details":{"cached_tokens":40},"output_tokens_details":{"reasoning_tokens":10}}}`

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/v1/responses", r.URL.Path)
				assert.Equal(t, http.MethodPost, r.Method)
				assert.Equal(t, "Bearer test-meta-key", r.Header.Get("Authorization"))

				var body struct {
					Model           string `json:"model"`
					Stream          bool   `json:"stream"`
					MaxOutputTokens int    `json:"max_output_tokens"`
					Reasoning       struct {
						Effort string `json:"effort"`
					} `json:"reasoning"`
					Tools []struct {
						Type string `json:"type"`
						Name string `json:"name"`
					} `json:"tools"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Errorf("decode request: %v", err)
					http.Error(w, "bad request", http.StatusBadRequest)

					return
				}

				assert.Equal(t, "muse-spark-1.3", body.Model)
				assert.Equal(t, streaming, body.Stream)
				assert.Equal(t, 32768, body.MaxOutputTokens)
				assert.Equal(t, "max", body.Reasoning.Effort)

				if assert.Len(t, body.Tools, 1) {
					assert.Equal(t, "function", body.Tools[0].Type)
					assert.Equal(t, "lookup", body.Tools[0].Name)
				}

				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					_, err := fmt.Fprintf(w, "data: %s\n\ndata: %s\n\n", `{"type":"response.output_text.delta","delta":"Hello","output_index":0,"content_index":0,"item_id":"msg_1","sequence_number":1}`, `{"type":"response.completed","sequence_number":2,"response":`+response+`}`)
					assert.NoError(t, err)
				} else {
					w.Header().Set("Content-Type", "application/json")
					_, err := fmt.Fprint(w, response)
					assert.NoError(t, err)
				}
			}))
			defer server.Close()

			p, err := NewProvider("test-meta-key", openai.WithBaseURL(server.URL+"/v1"), openai.WithHTTPClient(server.Client()))
			require.NoError(t, err)
			m, err := p.NewModel(ModelMuseSpark13, openai.WithReasoningEffort(openai.ReasoningEffortMax))
			require.NoError(t, err)

			req := &llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("Hello")}}},
				Tools:    []llm.ToolDefinition{{Name: "lookup", Parameters: json.RawMessage(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)}},
			}
			var result *llm.Response

			if streaming {
				var delta strings.Builder

				for event, err := range m.GenerateEvents(t.Context(), req) {
					require.NoError(t, err)

					switch e := event.(type) {
					case llm.ContentPartEvent:
						if text, ok := e.Part.(*llm.TextPart); ok {
							delta.WriteString(text.Text)
						}
					case llm.StreamEndEvent:
						require.NoError(t, e.Error)
						result = e.Response
					}
				}

				assert.Equal(t, "Hello", delta.String())
			} else {
				result, err = m.Generate(t.Context(), req)
				require.NoError(t, err)
			}

			require.NotNil(t, result)
			assert.Equal(t, "Hello", result.TextContent())
			assert.Equal(t, "muse-spark-1.3", result.InvokedModelID)
			require.NotNil(t, result.Usage)
			assert.Equal(t, 40, result.Usage.CachedInputTokens)
			assert.Equal(t, 10, result.Usage.ReasoningTokens)
		})
	}
}
