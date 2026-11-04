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
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

func foreignOffering(t *testing.T) catalog.Offering {
	t.Helper()

	base, ok := Catalog().Lookup(ModelClaudeOpus45)
	require.True(t, ok)

	e := base.Entry
	e.ID = "accounts/host/models/reasoning-model"
	e.Aliases = nil
	e.Life.ReplacedBy = ""
	c := catalog.MustNew("foreign-host", []catalog.Entry{e})
	offering, ok := c.Lookup(e.ID)
	require.True(t, ok)

	return offering
}

func TestForeignOfferingOpaqueHistoryReplay(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{false, true} {
		t.Run(map[bool]string{false: "generate", true: "stream"}[streaming], func(t *testing.T) {
			t.Parallel()

			requests := make(chan []byte, 2)
			const response = `{"id":"msg_foreign","type":"message","role":"assistant","model":"served-model","content":[{"type":"thinking","thinking":"plan","signature":"foreign-signature"},{"type":"redacted_thinking","data":"foreign-redacted"},{"type":"tool_use","id":"foreign-tool","name":"lookup","input":{"key":"x"}}],"stop_reason":"tool_use","usage":{"input_tokens":3,"cache_read_input_tokens":4,"output_tokens":6}}`
			const stream = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_foreign\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"served-model\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"pl\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"an\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"foreign-signature\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"foreign-redacted\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"foreign-tool\",\"name\":\"lookup\",\"input\":{}}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"key\\\":\\\"x\\\"}\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n" +
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"input_tokens\":3,\"cache_read_input_tokens\":4,\"output_tokens\":6}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				data, err := io.ReadAll(r.Body)
				assert.NoError(t, err)

				requests <- data

				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte(stream))
				} else {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(response))
				}
			}))
			t.Cleanup(server.Close)
			client := sdk.NewClient(option.WithAPIKey("synthetic-test-key"), option.WithBaseURL(server.URL), option.WithHTTPClient(server.Client()), option.WithMaxRetries(0))
			provider, err := NewProviderWithClient(&client)
			require.NoError(t, err)

			provider.EnableCaching = false
			offering := foreignOffering(t)
			model, err := provider.NewModelFromOffering(offering.ID, offering, WithMaxTokens(64))
			require.NoError(t, err)

			generate := func(request *llm.Request) *llm.Response {
				if !streaming {
					result, err := model.Generate(t.Context(), request)
					require.NoError(t, err)

					return result
				}

				var result *llm.Response
				var reasoningDeltas []*llm.ReasoningPart

				for event, err := range model.GenerateEvents(t.Context(), request) {
					require.NoError(t, err)

					if delta, ok := event.(llm.ContentPartEvent); ok {
						if part, ok := delta.Part.(*llm.ReasoningPart); ok {
							reasoningDeltas = append(reasoningDeltas, part)
						}
					}

					if end, ok := event.(llm.StreamEndEvent); ok {
						require.NoError(t, end.Error)
						result = end.Response
					}
				}

				require.NotNil(t, result)
				require.Len(t, reasoningDeltas, 2)
				assert.Equal(t, "pl", reasoningDeltas[0].Text)
				assert.Equal(t, "an", reasoningDeltas[1].Text)

				for _, part := range reasoningDeltas {
					assert.Equal(t, map[string]any{reasoningMetadataProvider: "foreign-host"}, part.Metadata)
					assert.Empty(t, part.Signature, "the opaque signature arrives after the thinking deltas")
				}

				return result
			}
			request := &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("lookup"))}}
			first := generate(request)

			<-requests
			assert.Equal(t, offering.Provider(), model.Provider())
			assert.Equal(t, offering.ID, first.InvokedModelID)
			assert.Equal(t, llm.FinishReasonToolCalls, first.FinishReason)
			require.Len(t, first.Message.Content, 3)
			thinking, ok := first.Message.Content[0].(*llm.ReasoningPart)
			require.True(t, ok)
			assert.Equal(t, "foreign-host", thinking.Metadata[reasoningMetadataProvider])

			redacted, ok := first.Message.Content[1].(*llm.ReasoningPart)
			require.True(t, ok)
			assert.Equal(t, "foreign-host", redacted.Metadata[reasoningMetadataRedactedProvider])
			assert.Equal(t, 13, first.Usage.TotalBilledTokens())
			assert.Zero(t, first.Usage.ReasoningTokens, "Messages does not supply a separate reasoning counter")

			persisted, err := json.Marshal(first.Message)
			require.NoError(t, err)
			var restored llm.Message
			require.NoError(t, json.Unmarshal(persisted, &restored))
			request.Messages = append(request.Messages, restored, llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("foreign-tool", "lookup", json.RawMessage(`"found"`), false)))
			generate(request)
			var replay struct {
				Model    string `json:"model"`
				Messages []struct {
					Content json.RawMessage `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(<-requests, &replay))
			assert.Equal(t, offering.ID, replay.Model)
			require.Len(t, replay.Messages, 3)
			assert.JSONEq(t, `[{"type":"thinking","thinking":"plan","signature":"foreign-signature"},{"type":"redacted_thinking","data":"foreign-redacted"},{"type":"tool_use","id":"foreign-tool","name":"lookup","input":{"key":"x"}}]`, string(replay.Messages[1].Content))
		})
	}
}

func TestNewModelFromOfferingLegacyNativeThinking(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name      string
		model     catalog.ModelID
		transport catalog.WireTransport
		legacy    bool
	}{
		{"native Claude Vertex binding", catalog.ModelClaudeOpus45, catalog.WireTransportVertex, true},
		{"Claude on an unrelated direct host", catalog.ModelClaudeOpus45, catalog.WireTransportDirect, false},
		{"non-Claude with Vertex binding", catalog.ModelGPT5, catalog.WireTransportVertex, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			e := foreignOffering(t).Entry
			e.Model = tt.model
			e.WireAPIs = []catalog.WireAPI{{Protocol: catalog.WireProtocolAnthropicMessages, Transport: tt.transport}}
			c := catalog.MustNew("host-from-metadata", []catalog.Entry{e})
			offering, ok := c.Lookup(e.ID)
			require.True(t, ok)

			provider, err := NewProvider("synthetic-test-key")
			require.NoError(t, err)
			model, err := provider.NewModelFromOffering(e.ID, offering, WithMaxTokens(64))
			require.NoError(t, err)

			concrete, ok := model.(*Model)
			require.True(t, ok)

			for _, producer := range []string{"", "different-host"} {
				part := &llm.ReasoningPart{Signature: "saved-signature"}
				if producer != "" {
					part.Metadata = map[string]any{reasoningMetadataProvider: producer}
				}

				request, err := concrete.requestMapper.ToProvider(&llm.Request{Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")),
					llm.NewMessage(llm.RoleAssistant, part, llm.NewTextPart("keep")),
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")),
				}})
				require.NoError(t, err)

				if tt.legacy && producer == "" {
					assert.Len(t, request.Messages[1].Content, 2)
				} else {
					assert.JSONEq(t, `[{"type":"text","text":"keep"}]`, wireContent(t, request.Messages[1]))
				}
			}
		})
	}
}

func TestLegacyRedactedThinkingRejectsUnrelatedProvenance(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name     string
		legacy   bool
		producer string
		keep     bool
	}{
		{"native legacy stamp", true, ProviderName, true},
		{"native current stamp", true, "native-host", true},
		{"unrelated explicit stamp", true, "different-host", false},
		{"missing stamp", true, "", false},
		{"legacy stamp on foreign binding", false, ProviderName, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			mapper := NewRequestMapper(&Config{ModelName: "served-model", MaxTokens: 64})
			mapper.provider = "native-host"
			mapper.legacyThinking = tt.legacy
			part := &llm.ReasoningPart{Signature: "opaque-data", Metadata: map[string]any{
				reasoningMetadataRedacted: true, reasoningMetadataRedactedProvider: tt.producer,
			}}
			request, err := mapper.ToProvider(&llm.Request{Messages: []llm.Message{
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")),
				llm.NewMessage(llm.RoleAssistant, part, llm.NewTextPart("keep")),
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")),
			}})
			require.NoError(t, err)

			if tt.keep {
				assert.Len(t, request.Messages[1].Content, 2)
				assert.Equal(t, "opaque-data", request.Messages[1].Content[0].OfRedactedThinking.Data)
			} else {
				assert.JSONEq(t, `[{"type":"text","text":"keep"}]`, wireContent(t, request.Messages[1]))
			}
		})
	}
}

func TestOpaqueThinkingDoesNotCrossOfferingProviders(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name     string
		provider llm.ProviderID
		part     *llm.ReasoningPart
		keep     bool
	}{
		{"legacy first-party", ProviderName, &llm.ReasoningPart{Text: "plan", Signature: "legacy-signature"}, true},
		{"legacy signature on foreign host", "foreign-host", &llm.ReasoningPart{Text: "plan", Signature: "legacy-signature"}, false},
		{"same foreign host", "foreign-host", &llm.ReasoningPart{Text: "plan", Signature: "signature", Metadata: map[string]any{reasoningMetadataProvider: "foreign-host"}}, true},
		{"foreign signature on first-party", ProviderName, &llm.ReasoningPart{Text: "plan", Signature: "signature", Metadata: map[string]any{reasoningMetadataProvider: "foreign-host"}}, false},
		{"foreign redacted on first-party", ProviderName, &llm.ReasoningPart{Signature: "redacted", Metadata: map[string]any{reasoningMetadataRedacted: true, reasoningMetadataRedactedProvider: "foreign-host"}}, false},
		{"first-party redacted on foreign host", "foreign-host", &llm.ReasoningPart{Signature: "redacted", Metadata: map[string]any{reasoningMetadataRedacted: true, reasoningMetadataRedactedProvider: ProviderName}}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			mapper := NewRequestMapper(&Config{ModelName: "served-model", MaxTokens: 64})
			mapper.provider = tt.provider
			request, err := mapper.ToProvider(&llm.Request{Messages: []llm.Message{
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")),
				llm.NewMessage(llm.RoleAssistant, tt.part, llm.NewTextPart("keep")),
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("again")),
			}})
			require.NoError(t, err)

			if tt.keep {
				assert.Len(t, request.Messages[1].Content, 2)
			} else {
				assert.JSONEq(t, `[{"type":"text","text":"keep"}]`, wireContent(t, request.Messages[1]))
			}
		})
	}
}
