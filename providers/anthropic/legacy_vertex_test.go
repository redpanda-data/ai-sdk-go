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

package anthropic_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

func TestVertexContinuesPersistedLegacyRedactedThinking(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{false, true} {
		t.Run(map[bool]string{false: "generate", true: "stream"}[streaming], func(t *testing.T) {
			t.Parallel()

			// Vertex previously used this native mapper without assigning its
			// host, so saved redacted payloads carry the Anthropic stamp.
			previous, err := anthropic.NewResponseMapper().FromProvider(&sdk.BetaMessage{
				ID: "old-vertex-message", StopReason: sdk.BetaStopReasonToolUse,
				Content: []sdk.BetaContentBlockUnion{
					{Type: "redacted_thinking", Data: "opaque-legacy-vertex-data"},
					{Type: "tool_use", ID: "saved-tool-id", Name: "lookup", Input: json.RawMessage(`{"key":"x"}`)},
				},
			})
			require.NoError(t, err)
			persisted, err := json.Marshal(previous.Message)
			require.NoError(t, err)
			var restored llm.Message
			require.NoError(t, json.Unmarshal(persisted, &restored))
			redacted, ok := restored.Content[0].(*llm.ReasoningPart)
			require.True(t, ok)
			require.Equal(t, "anthropic", redacted.Metadata["redacted_provider"])

			requests := make(chan []byte, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, readErr := io.ReadAll(r.Body)
				assert.NoError(t, readErr)

				requests <- body

				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, messagesStream(`{"input_tokens":2,"output_tokens":0}`, []string{`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`}, true))
				} else {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"id":"continued","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":2}}`)
				}
			}))
			t.Cleanup(server.Close)
			provider, err := vertex.NewProvider(t.Context(), vertex.WithProject("test-project"), vertex.WithLocation("us-east5"), vertex.WithBaseURL(server.URL), vertex.WithHTTPClient(server.Client()))
			require.NoError(t, err)
			model, err := provider.NewModel(vertex.ModelClaudeHaiku45)
			require.NoError(t, err)

			request := &llm.Request{Messages: []llm.Message{
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("lookup")),
				restored,
				llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("saved-tool-id", "lookup", json.RawMessage(`"found"`), false)),
			}}
			if streaming {
				for _, callErr := range model.GenerateEvents(t.Context(), request) {
					require.NoError(t, callErr)
				}
			} else {
				_, callErr := model.Generate(t.Context(), request)
				require.NoError(t, callErr)
			}

			var wire struct {
				Messages []struct {
					Content []struct {
						Type      string `json:"type"`
						Data      string `json:"data"`
						ID        string `json:"id"`
						ToolUseID string `json:"tool_use_id"`
					} `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(<-requests, &wire))
			require.Len(t, wire.Messages, 3)
			require.Len(t, wire.Messages[1].Content, 2)
			assert.Equal(t, "redacted_thinking", wire.Messages[1].Content[0].Type)
			assert.Equal(t, "opaque-legacy-vertex-data", wire.Messages[1].Content[0].Data)
			assert.Equal(t, "saved-tool-id", wire.Messages[1].Content[1].ID)
			assert.Equal(t, "saved-tool-id", wire.Messages[2].Content[0].ToolUseID)
		})
	}
}
