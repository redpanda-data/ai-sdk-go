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
	"os"
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
	original, source := contractReadReport(t, "testdata/sdk-runtime/messages-live.json")
	require.Equal(t, "e9619e0d2da096b18933217baaddbe22eebf952858f5ab350d14c63395641fa5", source.SHA256)

	for _, row := range report.Evidence {
		t.Run(fmt.Sprintf("stream=%t", row.Case.Stream), func(t *testing.T) {
			t.Parallel()
			require.True(t, row.Failed)
			require.Len(t, row.Exchanges, 1)

			matched := false

			for _, raw := range original.Evidence {
				if raw.Case.key() == row.Case.key() {
					require.Equal(t, raw.Exchanges, row.Exchanges)

					matched = true
				}
			}

			require.True(t, matched)

			captured := row.Exchanges[0]
			native := contractParse(t, row.Case, captured)
			require.False(t, json.Valid([]byte(native.Text)), "historical schema failure is not positive qualification")
			require.Contains(t, native.Text, "PROMPT_ONLY")
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

func sdkRuntimeCapturedServer(t *testing.T, rows []contractEvidence) (*httptest.Server, *atomic.Int32) {
	t.Helper()

	calls := new(atomic.Int32)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		n := int(calls.Add(1) - 1)
		if !assert.Less(t, n, len(rows), "unexpected SDK replay request") {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		e := rows[n].Exchanges[0]
		assert.Equal(t, e.Method, request.Method)
		assert.Equal(t, e.Path, request.URL.RequestURI())
		assert.Equal(t, "Bearer sdk-replay-not-a-credential", request.Header.Get("Authorization"))
		assert.Empty(t, request.Header.Get("X-Api-Key"))

		for name, value := range e.RequestHeaders {
			assert.Equal(t, value, request.Header.Get(name), "captured SDK header %s", name)
		}

		body, err := io.ReadAll(request.Body)
		if !assert.NoError(t, err) {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		assert.JSONEq(t, string(e.Request), string(body), "captured SDK request feature=%s", rows[n].Case.Feature)
		w.Header().Set("Content-Type", e.ResponseHeaders["Content-Type"])
		w.WriteHeader(e.Status)
		_, err = io.WriteString(w, e.Response)
		assert.NoError(t, err)
	}))
	t.Cleanup(server.Close)

	return server, calls
}

// Only the derived replay envelope is corrected: the historical recorder repeatedly
// hashed request IDs. The original capture and all response bytes remain unchanged.
func sdkRuntimeHistoricalHistoryCorrection(t *testing.T, first, history contractEvidence) contractEvidence {
	t.Helper()
	result := contractParse(t, first.Case, first.Exchanges[0])
	require.Len(t, result.Tools, 1)
	var body map[string]any
	require.NoError(t, json.Unmarshal(history.Exchanges[0].Request, &body))
	messages := contractArray(body["messages"])
	require.Len(t, messages, 3)
	var use, reply map[string]any

	for _, item := range contractArray(contractObject(messages[1])["content"]) {
		block := contractObject(item)
		if block["type"] == "tool_use" {
			require.Nil(t, use)
			use = block
		}
	}

	for _, item := range contractArray(contractObject(messages[2])["content"]) {
		block := contractObject(item)
		if block["type"] == "tool_result" {
			require.Nil(t, reply)
			reply = block
		}
	}

	require.NotNil(t, use)
	require.NotNil(t, reply)
	require.Equal(t, use["id"], reply["tool_use_id"])
	recorded := contractString(use["id"])
	rehashed := map[string]any{"id": result.Tools[0].ID}
	iterations := 0

	for iterations < 256 && rehashed["id"] != recorded {
		contractSanitizeValue(rehashed)

		iterations++
	}

	require.Equal(t, recorded, rehashed["id"], "only proven repeated-sanitization corruption may be corrected")
	require.Positive(t, iterations, "the historical linkage defect must remain visible")
	t.Logf("DERIVED HISTORICAL TOOL-ID CORRECTION model=%s stream=%t: response %s -> recorded history %s after %d extra hashes; replay restores response ID, original bytes unchanged", history.Case.Model, history.Case.Stream, result.Tools[0].ID, recorded, iterations)
	use["id"] = result.Tools[0].ID
	reply["tool_use_id"] = result.Tools[0].ID
	corrected, err := json.Marshal(body)
	require.NoError(t, err)

	history.Exchanges = append([]contractExchange(nil), history.Exchanges...)
	history.Exchanges[0].Request = corrected

	return history
}

func sdkRuntimeReplayFeatureConversation(t *testing.T, offering catalog.Offering, rows []contractEvidence) {
	t.Helper()

	stream := rows[0].Case.Stream
	server, calls := sdkRuntimeCapturedServer(t, rows)
	provider, err := fireworks.NewProvider(t.Context(), fireworks.WithAPIKey("sdk-replay-not-a-credential"), fireworks.WithBaseURL(server.URL+"/inference/v1"), fireworks.WithHTTPClient(server.Client()))
	require.NoError(t, err)

	call := func(feature string, request *llm.Request, maxTokens int, reasoning bool) *llm.Response {
		t.Helper()

		n := int(calls.Load())
		require.Less(t, n, len(rows))
		require.Equal(t, feature, rows[n].Case.Feature)
		model := sdkRuntimeModel(t, provider, nil, offering, maxTokens, reasoning)
		response := sdkRuntimeGenerate(t, model, request, stream, "sdk-replay-not-a-credential")
		require.EqualValues(t, n+1, calls.Load())
		sdkRuntimeAssertUsage(t, rows[n].Case, rows[n].Exchanges[0], response)

		return response
	}

	conversation := &llm.Request{Messages: []llm.Message{
		llm.NewMessage(llm.RoleSystem, llm.NewTextPart("Include the word tulip in your final answer, alongside the answer to the user's question.")),
		sdkRuntimeUser("My favorite animal is an otter. Remember that."),
		llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("I will remember your favorite animal.")),
		sdkRuntimeUser("What is my favorite animal?"),
	}}
	response := call("sdk_system_multiturn", conversation, 512, false)
	require.Equal(t, llm.FinishReasonStop, response.FinishReason)
	require.Contains(t, strings.ToLower(response.TextContent()), "otter")
	require.Contains(t, strings.ToLower(response.TextContent()), "tulip")

	name := "lookup"
	question := sdkRuntimeUser(contractAdaptiveTask + " Then call lookup with key x exactly once. Its reply is JSON data. After receiving it, state the exact result field value and finish with the computed n as a plain integer.")
	tool := &llm.Request{Messages: []llm.Message{question}, Tools: []llm.ToolDefinition{{Name: name, Description: "Look up a word by key.", Parameters: json.RawMessage(`{"type":"object","properties":{"key":{"type":"string","enum":["x"]}},"required":["key"],"additionalProperties":false}`)}}, ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceSpecific, Name: &name}}
	first := call("sdk_reasoning_tool", tool, 4096, true)
	require.Equal(t, llm.FinishReasonToolCalls, first.FinishReason)
	tools := first.ToolRequests()
	require.Len(t, tools, 1)
	require.Equal(t, name, tools[0].Name)
	require.JSONEq(t, `{"key":"x"}`, string(tools[0].Arguments))
	native := contractParse(t, rows[1].Case, rows[1].Exchanges[0])
	require.Equal(t, native.Tools[0].ID, tools[0].ID)

	reasoning := false

	for _, part := range first.Message.Content {
		if trace, ok := part.(*llm.ReasoningPart); ok && trace.Text != "" {
			reasoning = true
		}
	}

	require.True(t, reasoning || first.Usage.ReasoningTokens > 0)
	tool.Messages = []llm.Message{question, first.Message, llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(tools[0].ID, name, json.RawMessage(`{"result":"violet-lantern-zebra"}`), false))}
	tool.ToolChoice = &llm.ToolChoice{Type: llm.ToolChoiceAuto}
	second := call("sdk_reasoning_tool_history", tool, 4096, true)
	require.Equal(t, llm.FinishReasonStop, second.FinishReason)
	require.Empty(t, second.ToolRequests())
	require.Contains(t, second.TextContent(), "violet-lantern-zebra")
	contractAssertAdaptiveAnswer(t, second.TextContent())

	request := sdkRuntimeSchemaRequest()
	format := request.ResponseFormat
	request.ResponseFormat = nil
	plain := call("sdk_schema_plain_control", request, 512, false)
	require.Equal(t, llm.FinishReasonStop, plain.FinishReason)
	require.JSONEq(t, `{"gate":"PROMPT_ONLY","count":"wrong","extra":true}`, plain.TextContent())

	if offering.ID != fireworks.ModelMiniMaxM3 {
		request.ResponseFormat = format
		controlled := call("sdk_schema_enforced", request, 512, false)
		require.Equal(t, llm.FinishReasonStop, controlled.FinishReason)
		require.JSONEq(t, `{"gate":"SCHEMA_ACTIVE_73","count":73}`, controlled.TextContent())
	}

	truncated := call("sdk_length", &llm.Request{Messages: []llm.Message{sdkRuntimeUser("Write the numbers from 1 to 10000, separated by commas. Do not stop early.")}}, 1, false)
	require.Equal(t, llm.FinishReasonLength, truncated.FinishReason)
	require.Equal(t, 1, truncated.Usage.OutputTokens+truncated.Usage.ReasoningTokens)
	require.EqualValues(t, len(rows), calls.Load())
}

func TestFireworksSDKCapturedFeatureReplay(t *testing.T) {
	t.Parallel()

	original, err := os.ReadFile("testdata/sdk-runtime/messages-live.json")
	require.NoError(t, err)
	require.Equal(t, "e9619e0d2da096b18933217baaddbe22eebf952858f5ab350d14c63395641fa5", contractHash(original))
	report, _ := contractReadReport(t, "testdata/sdk-runtime/messages-live.json")
	require.Len(t, report.Evidence, 86)
	provider, err := fireworks.NewProvider(t.Context(), fireworks.WithAPIKey("sdk-replay-not-a-credential"))
	require.NoError(t, err)

	offerings := provider.Catalog().All()
	require.Len(t, offerings, 7)

	indexed := make(map[string]contractEvidence)
	var cache []contractEvidence

	for _, row := range report.Evidence {
		require.Len(t, row.Exchanges, 1)
		require.Equal(t, catalog.WireProtocolAnthropicMessages, row.Case.API)
		require.Equal(t, catalog.WireTransportDirect, row.Case.Transport)

		if row.Case.Feature == "sdk_positive_cache" {
			cache = append(cache, row)
			continue
		}

		require.NotContains(t, indexed, row.Case.key(), "duplicate SDK model/mode/feature")
		indexed[row.Case.key()] = row
	}

	require.Len(t, indexed, 84)
	require.Len(t, cache, 2)

	features := []string{"sdk_system_multiturn", "sdk_reasoning_tool", "sdk_reasoning_tool_history", "sdk_schema_plain_control", "sdk_schema_enforced", "sdk_length"}

	for _, offering := range offerings {
		for _, stream := range []bool{false, true} {
			var rows []contractEvidence

			for _, feature := range features {
				c := contractCase{Model: offering.ID, API: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect, Stream: stream, Feature: feature}
				row, ok := indexed[c.key()]
				require.True(t, ok, "missing SDK feature capture %+v", c)

				if offering.ID == fireworks.ModelMiniMaxM3 && feature == "sdk_schema_enforced" {
					continue
				}

				rows = append(rows, row)
			}

			rows[2] = sdkRuntimeHistoricalHistoryCorrection(t, rows[1], rows[2])
			t.Run(offering.ID+fmt.Sprintf("/stream=%t", stream), func(t *testing.T) {
				t.Parallel()
				sdkRuntimeReplayFeatureConversation(t, offering, rows)
			})
		}
	}

	t.Run("actual_sdk_cache", func(t *testing.T) {
		t.Parallel()
		sdkRuntimeReplayCapturedCache(t, cache)
	})

	t.Cleanup(func() {
		current, err := os.ReadFile("testdata/sdk-runtime/messages-live.json")
		require.NoError(t, err)
		require.Equal(t, original, current)
	})
}

type sdkRuntimeReplayHeaders struct {
	http.RoundTripper

	header map[string]string
}

func (r sdkRuntimeReplayHeaders) RoundTrip(request *http.Request) (*http.Response, error) {
	for name, value := range r.header {
		if strings.HasPrefix(name, "X-") {
			request.Header.Set(name, value)
		}
	}

	return r.RoundTripper.RoundTrip(request)
}

func sdkRuntimeReplayCapturedCache(t *testing.T, rows []contractEvidence) {
	t.Helper()
	require.Len(t, rows, 2)
	server, calls := sdkRuntimeCapturedServer(t, rows)
	client := server.Client()
	client.Transport = sdkRuntimeReplayHeaders{RoundTripper: client.Transport, header: rows[0].Exchanges[0].RequestHeaders}
	provider, err := fireworks.NewProvider(t.Context(), fireworks.WithAPIKey("sdk-replay-not-a-credential"), fireworks.WithBaseURL(server.URL+"/inference/v1"), fireworks.WithHTTPClient(client))
	require.NoError(t, err)
	model, err := provider.NewModel(fireworks.ModelDeepSeekV41Flash1017, fireworks.WithMaxTokens(16), fireworks.WithTemperature(0), fireworks.WithReasoningEffort(fireworks.ReasoningEffortNone))
	require.NoError(t, err)
	var body map[string]any
	require.NoError(t, json.Unmarshal(rows[0].Exchanges[0].Request, &body))
	prompt := contractString(contractObject(contractArray(contractObject(contractArray(body["messages"])[0])["content"])[0])["text"])
	require.NotEmpty(t, prompt)

	for n, row := range rows {
		require.Equal(t, fireworks.ModelDeepSeekV41Flash1017, row.Case.Model)
		require.False(t, row.Case.Stream)
		response := sdkRuntimeGenerate(t, model, &llm.Request{Messages: []llm.Message{sdkRuntimeUser(prompt)}}, false, "sdk-replay-not-a-credential")
		require.Equal(t, llm.FinishReasonStop, response.FinishReason)
		sdkRuntimeAssertUsage(t, row.Case, row.Exchanges[0], response)

		if n == 0 {
			require.Zero(t, response.Usage.CachedInputTokens)
		} else {
			require.Equal(t, 130, response.Usage.InputTokens)
			require.Equal(t, 15130, response.Usage.CachedInputTokens)
		}
	}

	require.EqualValues(t, len(rows), calls.Load())
}
