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
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestProviderDiscoveryAndConfiguration(t *testing.T) {
	t.Parallel()
	provider, err := NewProvider(t.Context())
	require.NoError(t, err)
	assert.Same(t, Catalog(), provider.Catalog())
	assert.Same(t, Catalog(), new(Provider).Catalog())
	assert.Equal(t, llm.ProviderID(ProviderName), provider.Name())
	_, err = provider.NewModel(ModelGLM53)
	require.ErrorContains(t, err, "WithAPIKey")
	_, err = provider.NewModel("glm-5p3")
	require.ErrorContains(t, err, "unsupported Fireworks model")

	for _, option := range []ProviderOption{
		WithAPIKey(""), WithAPIKey(" token"), WithAPIKey("token\r\nheader"), WithHTTPClient(nil), nil,
		WithBaseURL(""), WithBaseURL("file:///tmp/v1"), WithBaseURL("https://user:pass@example.com/v1"),
		WithBaseURL("https://example.com/v1?key=x"), WithBaseURL("https://example.com/v1#fragment"),
		WithBaseURL("https://example.com/no-version"),
	} {
		_, err := NewProvider(t.Context(), option)
		require.Error(t, err)
	}
}

func TestRuntimeMetadataIsSelectedAndNotCatalogMutation(t *testing.T) {
	t.Parallel()
	provider, err := NewProvider(t.Context(), WithAPIKey("gateway-service-account-token"))
	require.NoError(t, err)

	for _, offering := range Catalog().All() {
		t.Run(offering.ID, func(t *testing.T) {
			t.Parallel()

			require.NotEmpty(t, offering.WireAPIs)
			api := offering.WireAPIs[0]
			runtime, err := provider.NewModel(offering.ID)
			if api.Transport != catalog.WireTransportDirect || !slices.Contains([]catalog.WireProtocol{catalog.WireProtocolAnthropicMessages, catalog.WireProtocolOpenAIResponses, catalog.WireProtocolOpenAIChatCompletions}, api.Protocol) {
				require.ErrorContains(t, err, "unsupported first Fireworks wire API")
				return
			}

			require.NoError(t, err)

			support, ok := offering.SupportForWireAPI(api)
			require.True(t, ok)

			expected := support.Capabilities
			expected.Vision = false
			assert.Equal(t, expected, runtime.Capabilities())
			assert.Equal(t, offering.ID, runtime.Name())
			assert.Equal(t, llm.ProviderID(ProviderName), runtime.Provider())
			lister, ok := runtime.(llm.ReasoningEffortLister)
			require.True(t, ok)
			assert.Equal(t, support.Reasoning.Efforts, lister.SupportedReasoningEfforts())

			constraints := runtime.Constraints()
			params := []string{"max_tokens", "temperature"}

			if len(support.Reasoning.Efforts) > 0 {
				params = append(params, "reasoning_effort")
			}

			if support.Reasoning.Adaptive {
				params = append(params, "thinking")
			}

			assert.ElementsMatch(t, params, constraints.SupportedParams)
			assert.NoError(t, constraints.ValidateParameterSupport("max_tokens"))
			assert.NoError(t, constraints.ValidateParameterSupport("temperature"))
			require.Error(t, constraints.ValidateParameterSupport("top_p"))
			constraints.SupportedParams[0] = "caller-mutation"
			assert.NotContains(t, runtime.Constraints().SupportedParams, "caller-mutation")

			fresh, ok := Catalog().Resolve(offering.ID)
			require.True(t, ok)
			assert.Empty(t, fresh.Constraints.SupportedParams)
			assert.Equal(t, offering.Capabilities, fresh.Capabilities)
		})
	}
}

func TestFactoryNeverFallsBackFromFirstBinding(t *testing.T) {
	t.Parallel()
	provider, err := NewProvider(t.Context(), WithAPIKey("placeholder"))
	require.NoError(t, err)

	base, ok := Catalog().Resolve(ModelGLM53)
	require.True(t, ok)

	tests := []struct {
		name    string
		apis    []catalog.WireAPI
		support []catalog.WireAPISupport
		want    string
	}{
		{name: "empty", want: "no wire API"},
		{name: "unsupported protocol before valid", apis: append([]catalog.WireAPI{{Protocol: catalog.WireProtocolGeminiGenerateContent, Transport: catalog.WireTransportDirect}}, base.WireAPIs...), support: base.WireAPISupport, want: "unsupported first"},
		{name: "unsupported transport before valid", apis: append([]catalog.WireAPI{{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportVertex}}, base.WireAPIs...), support: base.WireAPISupport, want: "unsupported first"},
		{name: "missing selected metadata", apis: base.WireAPIs, want: "unknown Fireworks support"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			offering := base
			offering.WireAPIs = tt.apis
			offering.WireAPISupport = tt.support
			_, err := provider.newModelFromOffering(base.ID, offering)
			require.ErrorContains(t, err, tt.want)
		})
	}
}

func TestModelOptionsUseSelectedReasoningSupport(t *testing.T) {
	t.Parallel()
	provider, err := NewProvider(t.Context(), WithAPIKey("placeholder"))
	require.NoError(t, err)

	for _, tt := range []struct {
		name   string
		id     string
		option Option
	}{
		{"zero max", ModelGLM53, WithMaxTokens(0)},
		{"negative max", ModelGLM53, WithMaxTokens(-1)},
		{"excess max", ModelGLM53, WithMaxTokens(131073)},
		{"negative temperature", ModelGLM53, WithTemperature(-0.1)},
		{"excess temperature", ModelGLM53, WithTemperature(2.1)},
		{"NaN", ModelGLM53, WithTemperature(math.NaN())},
		{"infinite", ModelGLM53, WithTemperature(math.Inf(1))},
		{"invalid effort", ModelGLM53, WithReasoningEffort(ReasoningEffortMedium)},
		{"blank effort", ModelGLM53, WithReasoningEffort("")},
		{"MiniMax no effort", ModelMiniMaxM3, WithReasoningEffort(ReasoningEffortLow)},
		{"GLM no adaptive", ModelGLM53, WithThinking(true)},
		{"GLM no disabled adaptive", ModelGLM53, WithThinking(false)},
		{"nil option", ModelGLM53, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := provider.NewModel(tt.id, tt.option)
			require.Error(t, err)
		})
	}

	mini, ok := Catalog().Resolve(ModelMiniMaxM3)
	require.True(t, ok)

	mini.WireAPIs = []catalog.WireAPI{{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect}}
	_, err = provider.newModelFromOffering(mini.ID, mini, WithThinking(true))
	require.ErrorContains(t, err, "adaptive thinking is not supported")

	for _, option := range []Option{WithThinking(true), WithThinking(false)} {
		_, err := provider.NewModel(ModelMiniMaxM3, option)
		require.NoError(t, err)
	}
}

func TestFactoryWireDispatch(t *testing.T) {
	t.Parallel()

	for _, protocol := range []catalog.WireProtocol{catalog.WireProtocolAnthropicMessages, catalog.WireProtocolOpenAIResponses, catalog.WireProtocolOpenAIChatCompletions} {
		t.Run(string(protocol), func(t *testing.T) {
			t.Parallel()

			for _, id := range []string{ModelGLM53, ModelMiniMaxM3, ModelDeepSeekV41Flash1017} {
				t.Run(id, func(t *testing.T) {
					t.Parallel()

					offering, ok := Catalog().Resolve(id)
					require.True(t, ok)

					offering.WireAPIs = []catalog.WireAPI{{Protocol: protocol, Transport: catalog.WireTransportDirect}}
					var body map[string]any

					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						assert.Equal(t, http.MethodPost, r.Method)
						assert.Equal(t, "Bearer service-account-token", r.Header.Get("Authorization"))
						assert.Empty(t, r.Header.Get("X-Api-Key"))
						assert.Empty(t, r.Header.Get("Anthropic-Beta"))

						if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&body)) {
							return
						}

						assert.Equal(t, id, body["model"])
						assert.InDelta(t, 0.4, body["temperature"], 0)
						assert.NotContains(t, body, "cache_control")
						w.Header().Set("Content-Type", "application/json")

						switch protocol {
						case catalog.WireProtocolAnthropicMessages:
							assert.Equal(t, "/gateway/inference/v1/messages", r.URL.Path)
							assert.InDelta(t, 128, body["max_tokens"], 0)

							_, err := io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"native-reported-id","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn","usage":{"input_tokens":20,"cache_read_input_tokens":5,"output_tokens":3}}`)
							assert.NoError(t, err)
						case catalog.WireProtocolOpenAIResponses:
							assert.Equal(t, "/gateway/inference/v1/responses", r.URL.Path)
							assert.Equal(t, false, body["store"])
							assert.InDelta(t, 128, body["max_output_tokens"], 0)

							_, err := io.WriteString(w, `{"id":"resp_1","object":"response","status":"completed","model":"native-reported-id","output":[{"id":"out_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"OK","annotations":[]}]}],"usage":{"input_tokens":25,"input_tokens_details":{"cached_tokens":5},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":28}}`)
							assert.NoError(t, err)
						case catalog.WireProtocolOpenAIChatCompletions:
							assert.Equal(t, "/gateway/inference/v1/chat/completions", r.URL.Path)
							assert.InDelta(t, 128, body["max_tokens"], 0)

							_, err := io.WriteString(w, `{"id":"chat_1","object":"chat.completion","model":"native-reported-id","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":25,"prompt_tokens_details":{"cached_tokens":5},"completion_tokens":3,"total_tokens":28}}`)
							assert.NoError(t, err)
						case catalog.WireProtocolGeminiGenerateContent, catalog.WireProtocolBedrockConverse:
							t.Errorf("unsupported protocol %s", protocol)
						default:
							t.Errorf("unexpected protocol %s", protocol)
						}
					}))
					defer server.Close()

					provider, err := NewProvider(t.Context(), WithAPIKey("service-account-token"), WithBaseURL(server.URL+"/gateway/inference/v1/"), WithHTTPClient(server.Client()))
					require.NoError(t, err)

					opts := []Option{WithMaxTokens(128), WithTemperature(0.4)}

					switch id {
					case ModelGLM53:
						opts = append(opts, WithReasoningEffort(ReasoningEffortLow))
					case ModelMiniMaxM3:
						if protocol != catalog.WireProtocolOpenAIResponses {
							opts = append(opts, WithThinking(true))
						}
					case ModelDeepSeekV41Flash1017:
						opts = append(opts, WithReasoningEffort(ReasoningEffortNone))
					}

					runtime, err := provider.newModelFromOffering(id, offering, opts...)
					require.NoError(t, err)
					response, err := runtime.Generate(t.Context(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleSystem, llm.NewTextPart("Be concise.")), llm.NewMessage(llm.RoleUser, llm.NewTextPart("Reply OK."))}})
					require.NoError(t, err)
					assert.Equal(t, "OK", response.TextContent())
					assert.Equal(t, id, response.InvokedModelID)
					assert.Equal(t, llm.FinishReasonStop, response.FinishReason)
					assert.Equal(t, 20, response.Usage.InputTokens)
					assert.Equal(t, 5, response.Usage.CachedInputTokens)
					assert.Equal(t, 3, response.Usage.OutputTokens)

					encodedBody, err := json.Marshal(body)
					require.NoError(t, err)
					assert.NotContains(t, string(encodedBody), "cache_control")
					assert.NotContains(t, string(encodedBody), "budget_tokens")
					assert.NotContains(t, string(encodedBody), "display")

					if protocol == catalog.WireProtocolOpenAIResponses {
						switch id {
						case ModelGLM53:
							assert.Equal(t, map[string]any{"effort": "low"}, body["reasoning"])
						case ModelDeepSeekV41Flash1017:
							assert.Equal(t, map[string]any{"effort": "none"}, body["reasoning"])
						default:
							assert.NotContains(t, body, "reasoning")
						}
					}

					if protocol == catalog.WireProtocolAnthropicMessages && id == ModelGLM53 {
						assert.Equal(t, map[string]any{"effort": "low"}, body["output_config"])
						assert.NotContains(t, body, "thinking")
					}

					if protocol == catalog.WireProtocolOpenAIChatCompletions && id == ModelGLM53 {
						assert.Equal(t, "low", body["reasoning_effort"])
					}

					if protocol != catalog.WireProtocolOpenAIResponses && id != ModelGLM53 {
						mode := "disabled"
						if id == ModelMiniMaxM3 {
							mode = "adaptive"
						}

						assert.Equal(t, map[string]any{"type": mode}, body["thinking"])
						assert.NotContains(t, body, "reasoning_effort")
						assert.NotContains(t, body, "output_config")
					}
				})
			}
		})
	}
}

func TestRequestAdmission(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, err := io.WriteString(w, `{"id":"m1","type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		assert.NoError(t, err)
	}))
	defer server.Close()

	provider, err := NewProvider(t.Context(), WithAPIKey("placeholder"), WithBaseURL(server.URL+"/v1"), WithHTTPClient(server.Client()))
	require.NoError(t, err)
	runtime, err := provider.NewModel(ModelGLM53)
	require.NoError(t, err)

	messages := []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Reply OK."))}
	for _, request := range []*llm.Request{nil, {Messages: messages, ToolSearch: true}, {Messages: messages, Options: map[string]any{"service_tier": "priority"}}, {Messages: messages, ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}}} {
		want := llm.ErrUnsupportedFeature
		if request == nil {
			want = llm.ErrInvalidInput
		}

		_, err := runtime.Generate(t.Context(), request)
		require.ErrorIs(t, err, want)

		count := 0

		for event, err := range runtime.GenerateEvents(t.Context(), request) {
			assert.Nil(t, event)
			require.ErrorIs(t, err, want)

			count++
		}

		assert.Equal(t, 1, count)
		assert.Zero(t, calls.Load(), "invalid requests must not reach HTTP")
	}
}
