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
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestFactoryStreamingDispatch(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		protocol catalog.WireProtocol
		path     string
		sse      string
	}{
		{catalog.WireProtocolAnthropicMessages, "messages", "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"native-id\",\"content\":[],\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"OK\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"},
		{catalog.WireProtocolOpenAIResponses, "responses", "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"OK\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"status\":\"completed\",\"model\":\"native-id\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"OK\"}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"total_tokens\":12}}}\n\n"},
		{catalog.WireProtocolOpenAIChatCompletions, "chat/completions", "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"OK\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\ndata: [DONE]\n\n"},
	} {
		t.Run(string(tt.protocol), func(t *testing.T) {
			t.Parallel()

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/gateway/v1/"+tt.path, r.URL.Path)
				assert.Equal(t, "Bearer gateway-token", r.Header.Get("Authorization"))

				var body map[string]any
				if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&body)) {
					return
				}

				assert.Equal(t, true, body["stream"])
				assert.Equal(t, ModelGLM53, body["model"])

				budgetField := "max_tokens"
				if tt.protocol == catalog.WireProtocolOpenAIResponses {
					budgetField = "max_output_tokens"
				}

				assert.InDelta(t, 4096, body[budgetField], 0)
				w.Header().Set("Content-Type", "text/event-stream")
				_, err := io.WriteString(w, tt.sse)
				assert.NoError(t, err)
			}))
			defer server.Close()

			provider, err := NewProvider(t.Context(), WithAPIKey("gateway-token"), WithBaseURL(server.URL+"/gateway/v1"), WithHTTPClient(server.Client()))
			require.NoError(t, err)

			offering, ok := Catalog().Resolve(ModelGLM53)
			require.True(t, ok)

			offering.WireAPIs = []catalog.WireAPI{{Protocol: tt.protocol, Transport: catalog.WireTransportDirect}}
			runtime, err := provider.newModelFromOffering(ModelGLM53, offering)
			require.NoError(t, err)
			var final *llm.Response
			text := ""

			for event, err := range runtime.GenerateEvents(t.Context(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("OK"))}}) {
				require.NoError(t, err)
				require.Nil(t, final, "no event may follow a final response")

				switch e := event.(type) {
				case llm.ContentPartEvent:
					part, ok := e.Part.(*llm.TextPart)
					require.True(t, ok)

					text += part.Text
				case llm.StreamEndEvent:
					require.NoError(t, e.Error)
					final = e.Response
				case llm.ErrorEvent, llm.StreamResetEvent:
					t.Fatalf("unexpected event %T", event)
				}
			}

			require.NotNil(t, final)
			assert.Equal(t, "OK", text)
			assert.Equal(t, text, final.TextContent())
			assert.Equal(t, ModelGLM53, final.InvokedModelID)
			assert.Equal(t, llm.FinishReasonStop, final.FinishReason)
			assert.Equal(t, 10, final.Usage.InputTokens)
			assert.Equal(t, 2, final.Usage.OutputTokens)
		})
	}
}
