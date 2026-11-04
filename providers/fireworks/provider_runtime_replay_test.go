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

package fireworks_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func sdkRuntimeGenerate(t *testing.T, model llm.Model, request *llm.Request, stream bool, key string) *llm.Response {
	t.Helper()

	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()

	if !stream {
		response, err := model.Generate(ctx, request)
		if err != nil {
			t.Fatalf("SDK generation failed (%T): %s", err, strings.ReplaceAll(err.Error(), key, "[REDACTED]"))
		}

		require.NotNil(t, response)

		return response
	}
	var response *llm.Response
	var text strings.Builder
	ended := false

	for event, err := range model.GenerateEvents(ctx, request) {
		if err != nil {
			t.Fatalf("SDK iterator failed (%T): %s", err, strings.ReplaceAll(err.Error(), key, "[REDACTED]"))
		}

		require.False(t, ended, "SDK emitted an event after its terminal event")

		switch e := event.(type) {
		case llm.ContentPartEvent:
			if part, ok := e.Part.(*llm.TextPart); ok {
				text.WriteString(part.Text)
			}
		case llm.ErrorEvent:
			t.Fatalf("SDK stream error: %s", strings.ReplaceAll(e.Message, key, "[REDACTED]"))
		case llm.StreamEndEvent:
			ended = true

			if e.Error != nil {
				t.Fatalf("SDK terminal failure (%T): %s", e.Error, strings.ReplaceAll(e.Error.Error(), key, "[REDACTED]"))
			}

			response = e.Response
		case llm.StreamResetEvent:
			t.Fatal("unexpected retry in a bounded SDK runtime test")
		}
	}

	require.True(t, ended, "SDK stream must emit exactly one terminal event")
	require.NotNil(t, response)
	require.Equal(t, text.String(), response.TextContent(), "stream deltas must agree with the final SDK response")

	return response
}

func sdkRuntimeAssertUsage(t *testing.T, c contractCase, native contractExchange, response *llm.Response) {
	t.Helper()
	raw := contractParse(t, c, native)
	require.NotNil(t, response.Usage)
	require.Equal(t, int(raw.Input), response.Usage.InputTokens)
	require.Equal(t, int(raw.Cached), response.Usage.CachedInputTokens)
	require.Equal(t, int(raw.Output), response.Usage.OutputTokens+response.Usage.ReasoningTokens)
	require.Equal(t, c.Model, response.InvokedModelID)
	require.Equal(t, llm.RoleAssistant, response.Message.Role)
}

func sdkRuntimeUser(prompt string) llm.Message {
	return llm.NewMessage(llm.RoleUser, llm.NewTextPart(prompt))
}

func sdkRuntimeSchemaRequest() *llm.Request {
	return &llm.Request{Messages: []llm.Message{sdkRuntimeUser(`Return exactly this object, retaining every value, type and field: {"gate":"PROMPT_ONLY","count":"wrong","extra":true}. Do not change it.`)}, ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, JSONSchema: &llm.JSONSchema{Name: "sdk_output", Schema: json.RawMessage(`{"type":"object","properties":{"gate":{"type":"string","enum":["SCHEMA_ACTIVE_73"]},"count":{"type":"integer","enum":[73]}},"required":["gate","count"],"additionalProperties":false}`)}}}
}

func TestFireworksSDKRawMessagesReplay(t *testing.T) {
	t.Parallel()

	report, _ := contractReadReport(t, "testdata/contracts/revalidated-evidence.json")
	for _, row := range report.Evidence {
		if row.Case.API != catalog.WireProtocolAnthropicMessages || row.Case.Feature != contractComplete {
			continue
		}

		t.Run(row.Case.Model+fmt.Sprintf("/stream=%t", row.Case.Stream), func(t *testing.T) {
			t.Parallel()
			require.Len(t, row.Exchanges, 1)
			captured := row.Exchanges[0]

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
				assert.Equal(t, http.MethodPost, request.Method)
				assert.Equal(t, "/inference/v1/messages", request.URL.Path)
				assert.Equal(t, "Bearer sdk-replay-not-a-credential", request.Header.Get("Authorization"))
				assert.Empty(t, request.Header.Get("X-Api-Key"))

				var body map[string]any
				if !assert.NoError(t, json.NewDecoder(request.Body).Decode(&body)) {
					w.WriteHeader(http.StatusBadRequest)
					return
				}

				assert.Equal(t, row.Case.Model, body["model"])
				assert.Equal(t, row.Case.Stream, body["stream"] == true)
				assert.InDelta(t, 512, body["max_tokens"], 0)
				w.Header().Set("Content-Type", captured.ResponseHeaders["Content-Type"])
				w.WriteHeader(captured.Status)
				_, err := io.WriteString(w, captured.Response)
				assert.NoError(t, err)
			}))
			defer server.Close()

			provider, err := fireworks.NewProvider(t.Context(), fireworks.WithAPIKey("sdk-replay-not-a-credential"), fireworks.WithBaseURL(server.URL+"/inference/v1"), fireworks.WithHTTPClient(server.Client()))
			require.NoError(t, err)
			model, err := provider.NewModel(row.Case.Model, fireworks.WithMaxTokens(512))
			require.NoError(t, err)
			require.Equal(t, llm.ProviderID(fireworks.ProviderName), model.Provider())
			require.False(t, model.Capabilities().Vision, "the SDK has no image Part even when the native catalog supports images")
			response := sdkRuntimeGenerate(t, model, &llm.Request{Messages: []llm.Message{sdkRuntimeUser("Reply with exactly OK.")}}, row.Case.Stream, "sdk-replay-not-a-credential")
			require.Equal(t, llm.FinishReasonStop, response.FinishReason)
			require.NotEmpty(t, response.TextContent())
			sdkRuntimeAssertUsage(t, row.Case, captured, response)
		})
	}
}

func TestFireworksSDKRawPositiveCacheReplay(t *testing.T) {
	t.Parallel()
	report, _ := contractReadReport(t, "testdata/contracts/cache-diagnostic-65536.json")
	require.Len(t, report.Evidence, 1)
	row := report.Evidence[0]
	require.Len(t, row.Exchanges, 3)

	var next atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		n := int(next.Add(1) - 1)
		if !assert.Less(t, n, len(row.Exchanges), "unexpected SDK replay request") {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		assert.Equal(t, "/inference/v1/messages", request.URL.Path)

		e := row.Exchanges[n]

		w.Header().Set("Content-Type", e.ResponseHeaders["Content-Type"])
		w.WriteHeader(e.Status)
		_, err := io.WriteString(w, e.Response)
		assert.NoError(t, err)
	}))
	defer server.Close()

	provider, err := fireworks.NewProvider(t.Context(), fireworks.WithAPIKey("sdk-replay-not-a-credential"), fireworks.WithBaseURL(server.URL+"/inference/v1"), fireworks.WithHTTPClient(server.Client()))
	require.NoError(t, err)
	model, err := provider.NewModel(row.Case.Model, fireworks.WithMaxTokens(16), fireworks.WithReasoningEffort(fireworks.ReasoningEffortNone))
	require.NoError(t, err)
	var body map[string]any
	require.NoError(t, json.Unmarshal(row.Exchanges[0].Request, &body))

	prompt := contractString(contractObject(contractArray(body["messages"])[0])["content"])
	for n, native := range row.Exchanges {
		response := sdkRuntimeGenerate(t, model, &llm.Request{Messages: []llm.Message{sdkRuntimeUser(prompt)}}, false, "sdk-replay-not-a-credential")
		sdkRuntimeAssertUsage(t, row.Case, native, response)

		if n > 0 {
			require.Equal(t, 130, response.Usage.InputTokens)
			require.Equal(t, 15130, response.Usage.CachedInputTokens)
		}
	}

	require.EqualValues(t, len(row.Exchanges), next.Load())
}

func TestFireworksSDKMiniMaxSchemaReplay(t *testing.T) {
	t.Parallel()

	for _, mode := range []string{"default", "adaptive"} {
		report, _ := contractReadReport(t, "testdata/sdk-runtime/minimax-schema-"+mode+".json")
		require.Len(t, report.Evidence, 2)

		for _, row := range report.Evidence {
			t.Run(mode+fmt.Sprintf("/stream=%t", row.Case.Stream), func(t *testing.T) {
				t.Parallel()
				require.False(t, row.Failed)
				require.Len(t, row.Exchanges, 1)
				captured := row.Exchanges[0]
				var calls atomic.Int32

				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
					assert.EqualValues(t, 1, calls.Add(1))
					assert.Equal(t, captured.Path, request.URL.RequestURI())
					assert.Equal(t, "Bearer sdk-replay-not-a-credential", request.Header.Get("Authorization"))
					assert.Empty(t, request.Header.Get("X-Api-Key"))

					body, err := io.ReadAll(request.Body)
					if !assert.NoError(t, err) {
						w.WriteHeader(http.StatusBadRequest)
						return
					}

					assert.JSONEq(t, string(captured.Request), string(body))
					w.Header().Set("Content-Type", captured.ResponseHeaders["Content-Type"])
					w.WriteHeader(captured.Status)
					_, err = io.WriteString(w, captured.Response)
					assert.NoError(t, err)
				}))
				defer server.Close()

				provider, err := fireworks.NewProvider(t.Context(), fireworks.WithAPIKey("sdk-replay-not-a-credential"), fireworks.WithBaseURL(server.URL+"/inference/v1"), fireworks.WithHTTPClient(server.Client()))
				require.NoError(t, err)

				opts := []fireworks.Option{fireworks.WithMaxTokens(512), fireworks.WithTemperature(0)}
				if mode == "adaptive" {
					opts = append(opts, fireworks.WithThinking(true))
				}

				model, err := provider.NewModel(fireworks.ModelMiniMaxM3, opts...)
				require.NoError(t, err)
				require.True(t, model.Capabilities().StructuredOutput)
				response := sdkRuntimeGenerate(t, model, sdkRuntimeSchemaRequest(), row.Case.Stream, "sdk-replay-not-a-credential")
				require.Equal(t, llm.FinishReasonStop, response.FinishReason)
				require.JSONEq(t, `{"gate":"SCHEMA_ACTIVE_73","count":73}`, response.TextContent())
				sdkRuntimeAssertUsage(t, row.Case, captured, response)
				require.EqualValues(t, 1, calls.Load())
			})
		}
	}
}

func TestFireworksSDKMiniMaxDisabledSchemaRejected(t *testing.T) {
	t.Parallel()
	report, _ := contractReadReport(t, "testdata/sdk-runtime/messages-live-outcome.json")
	require.Len(t, report.Evidence, 2)

	for _, row := range report.Evidence {
		t.Run(fmt.Sprintf("stream=%t", row.Case.Stream), func(t *testing.T) {
			t.Parallel()
			require.True(t, row.Failed)
			require.Len(t, row.Exchanges, 1)
			captured := row.Exchanges[0]
			var calls atomic.Int32

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", captured.ResponseHeaders["Content-Type"])
				w.WriteHeader(captured.Status)
				_, err := io.WriteString(w, captured.Response)
				assert.NoError(t, err)
			}))
			defer server.Close()

			provider, err := fireworks.NewProvider(t.Context(), fireworks.WithAPIKey("sdk-replay-not-a-credential"), fireworks.WithBaseURL(server.URL+"/inference/v1"), fireworks.WithHTTPClient(server.Client()))
			require.NoError(t, err)

			offering, ok := provider.Catalog().Lookup(fireworks.ModelMiniMaxM3)
			require.True(t, ok)
			support, ok := offering.SupportForWireAPI(offering.WireAPIs[0])
			require.True(t, ok)
			require.True(t, support.Capabilities.StructuredOutput)

			model, err := provider.NewModel(fireworks.ModelMiniMaxM3, fireworks.WithMaxTokens(512), fireworks.WithTemperature(0), fireworks.WithThinking(false))
			require.NoError(t, err)
			assert.False(t, model.Capabilities().StructuredOutput)

			if row.Case.Stream {
				var rejected, emitted bool

				for event, err := range model.GenerateEvents(t.Context(), sdkRuntimeSchemaRequest()) {
					if err != nil {
						require.ErrorIs(t, err, llm.ErrUnsupportedFeature)

						rejected = true
					}

					emitted = emitted || event != nil
				}

				assert.True(t, rejected)
				assert.False(t, emitted)
			} else {
				_, err := model.Generate(t.Context(), sdkRuntimeSchemaRequest())
				require.ErrorIs(t, err, llm.ErrUnsupportedFeature)
			}

			assert.Zero(t, calls.Load(), "unsupported schema and explicit disabled thinking must fail before HTTP")
		})
	}
}
