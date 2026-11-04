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

package fireworks

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestFactoryDefaultOutputLimit(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		protocol catalog.WireProtocol
		field    string
		response string
	}{
		{catalog.WireProtocolAnthropicMessages, "max_tokens", `{"id":"m1","type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`},
		{catalog.WireProtocolOpenAIResponses, "max_output_tokens", `{"id":"r1","object":"response","status":"completed","output":[{"id":"o1","type":"message","role":"assistant","content":[{"type":"output_text","text":"OK"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`},
		{catalog.WireProtocolOpenAIChatCompletions, "max_tokens", `{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`},
	} {
		for _, outputLimit := range []int{8192, 32} {
			t.Run(fmt.Sprintf("%s/limit=%d", tt.protocol, outputLimit), func(t *testing.T) {
				t.Parallel()

				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
					var body map[string]any
					if !assert.NoError(t, json.NewDecoder(req.Body).Decode(&body)) {
						return
					}

					assert.InDelta(t, min(4096, outputLimit), body[tt.field], 0)
					w.Header().Set("Content-Type", "application/json")
					_, err := io.WriteString(w, tt.response)
					assert.NoError(t, err)
				}))
				defer server.Close()

				provider, err := NewProvider(t.Context(), WithAPIKey("fake-token"), WithBaseURL(server.URL+"/v1"), WithHTTPClient(server.Client()))
				require.NoError(t, err)

				offering, ok := Catalog().Resolve(ModelGLM53)
				require.True(t, ok)

				offering.WireAPIs = []catalog.WireAPI{{Protocol: tt.protocol, Transport: catalog.WireTransportDirect}}
				offering.Constraints.MaxOutputTokens = outputLimit
				model, err := provider.newModelFromOffering(offering.ID, offering)
				require.NoError(t, err)
				response, err := model.Generate(t.Context(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("OK"))}})
				require.NoError(t, err)
				assert.Equal(t, "OK", response.TextContent())
			})
		}
	}
}

func TestFactoryResponsesStreamingEffort(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		id     string
		effort llm.ReasoningEffort
	}{
		{ModelGLM53, ReasoningEffortLow},
		{ModelDeepSeekV41Flash1017, ReasoningEffortNone},
	} {
		t.Run(tt.id, func(t *testing.T) {
			t.Parallel()

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				var body map[string]any
				if !assert.NoError(t, json.NewDecoder(req.Body).Decode(&body)) {
					return
				}

				assert.Equal(t, map[string]any{"effort": string(tt.effort)}, body["reasoning"])
				assert.Equal(t, false, body["store"])
				w.Header().Set("Content-Type", "text/event-stream")
				_, err := io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"OK\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n")
				assert.NoError(t, err)
			}))
			defer server.Close()

			provider, err := NewProvider(t.Context(), WithAPIKey("fake-token"), WithBaseURL(server.URL+"/v1"), WithHTTPClient(server.Client()))
			require.NoError(t, err)

			offering, ok := Catalog().Resolve(tt.id)
			require.True(t, ok)

			offering.WireAPIs = []catalog.WireAPI{{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect}}
			model, err := provider.newModelFromOffering(offering.ID, offering, WithReasoningEffort(tt.effort))
			require.NoError(t, err)

			var final *llm.Response

			for event, err := range model.GenerateEvents(t.Context(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("OK"))}}) {
				require.NoError(t, err)

				if end, ok := event.(llm.StreamEndEvent); ok {
					require.NoError(t, end.Error)
					final = end.Response
				}
			}

			require.NotNil(t, final)
			assert.Equal(t, "OK", final.TextContent())
		})
	}
}
