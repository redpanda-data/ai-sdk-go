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

package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Exercise the real transports: region metadata must survive both response paths
// so SelectorFromResponse can select the regional card without caller overrides.
func TestInRegionResponsePricing(t *testing.T) {
	t.Parallel()
	const region = "eu-central-1"

	for _, id := range []string{ModelGemma431B, ModelMistralLarge3, ModelClaudeSonnet55Global} {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			response := `{"id":"resp_1","model":"` + id + `","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hi","annotations":[]}]}],"usage":{"input_tokens":1000,"output_tokens":1000,"total_tokens":2000}}`
			var awsStream bytes.Buffer
			encoder := eventstream.NewEncoder()

			for _, event := range []struct{ name, payload string }{
				{"messageStop", `{"stopReason":"end_turn"}`},
				{"metadata", `{"usage":{"inputTokens":1000,"outputTokens":1000,"totalTokens":2000},"metrics":{"latencyMs":1}}`},
			} {
				headers := eventstream.Headers{}
				headers.Set(":message-type", eventstream.StringValue("event"))
				headers.Set(":event-type", eventstream.StringValue(event.name))
				headers.Set(":content-type", eventstream.StringValue("application/json"))
				require.NoError(t, encoder.Encode(&awsStream, eventstream.Message{Headers: headers, Payload: []byte(event.payload)}))
			}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}

				switch {
				case strings.HasSuffix(r.URL.Path, "/converse-stream"):
					w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
					_, _ = w.Write(awsStream.Bytes())
				case strings.HasSuffix(r.URL.Path, "/converse"):
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"hi"}]}},"stopReason":"end_turn","usage":{"inputTokens":1000,"outputTokens":1000,"totalTokens":2000},"metrics":{"latencyMs":1}}`))
				case string(body["stream"]) == "true":
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":%s}\n\n", response)
				default:
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(response))
				}
			}))
			defer server.Close()

			provider, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{Region: region, BaseEndpoint: aws.String(server.URL), Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
				return aws.Credentials{AccessKeyID: "test", SecretAccessKey: "test"}, nil
			})}), WithNoAuth())
			require.NoError(t, err)
			model, err := provider.NewModel(id)
			require.NoError(t, err)

			req := &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))}}
			resp, err := model.Generate(context.Background(), req)
			require.NoError(t, err)
			var streamed *llm.Response

			for event, err := range model.GenerateEvents(context.Background(), req) {
				require.NoError(t, err)

				if end, ok := event.(llm.StreamEndEvent); ok {
					require.NoError(t, end.Error)
					streamed = end.Response
				}
			}

			require.NotNil(t, streamed)

			want := region
			if id == ModelClaudeSonnet55Global {
				want = ""
			} // source region is not execution region for CRIS

			for _, got := range []*llm.Response{resp, streamed} {
				require.NotNil(t, got)
				assert.Equal(t, want, pricing.SelectorFromResponse(got).Region)
			}
		})
	}
}
