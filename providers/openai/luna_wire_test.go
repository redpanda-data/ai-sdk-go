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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

// TestGPT6LunaRuntimeWireIsUnchanged pins the request gpt-6-luna's runtime
// sends, so catalog-only WireAPIs metadata can never move its transport.
func TestGPT6LunaRuntimeWireIsUnchanged(t *testing.T) {
	t.Parallel()

	for streaming, want := range map[bool]string{
		false: "POST /responses\n" +
			`{"max_output_tokens":64,"input":[{"content":"Say ok.","role":"user"}],"model":"gpt-6-luna","reasoning":{"effort":"low"}}`,
		true: "POST /responses\n" +
			`{"max_output_tokens":64,"input":[{"content":"Say ok.","role":"user"}],"model":"gpt-6-luna","reasoning":{"effort":"low"},"stream":true}`,
	} {
		t.Run(fmt.Sprintf("stream=%t", streaming), func(t *testing.T) {
			t.Parallel()

			requests := make(chan string, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}

				requests <- r.Method + " " + r.URL.Path + "\n" + string(body)

				response := `{"id":"resp_luna","object":"response","status":"completed","model":"gpt-6-luna","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}`

				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":%s}\n\n", response)

					return
				}

				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, response)
			}))
			t.Cleanup(server.Close)

			provider, err := openai.NewProvider("placeholder-key", openai.WithBaseURL(server.URL))
			require.NoError(t, err)

			model, err := provider.NewModel(openai.ModelGPT6Luna, openai.WithMaxTokens(64), openai.WithReasoningEffort(openai.ReasoningEffortLow))
			require.NoError(t, err)

			request := &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Say ok."))}}
			if streaming {
				for _, streamErr := range model.GenerateEvents(t.Context(), request) {
					require.NoError(t, streamErr)
				}
			} else {
				_, err = model.Generate(t.Context(), request)
				require.NoError(t, err)
			}

			got := <-requests
			assert.Equal(t, want, got)
		})
	}
}
