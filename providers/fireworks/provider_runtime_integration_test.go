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
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
	"github.com/redpanda-data/ai-sdk-go/providers/testutil"
)

var (
	sdkRuntimeLive       = flag.Bool("fireworks-sdk-live", false, "explicit local-only Fireworks SDK Generate/GenerateEvents integration tests")
	sdkRuntimeKeyFile    = flag.String("fireworks-sdk-key-file", "", "credential file read only by the opted-in SDK test process")
	sdkRuntimeCache      = flag.Bool("fireworks-sdk-cache", false, "include at most three SDK nonstream cache requests after the feature tests")
	sdkRuntimeMiniSchema = flag.String("fireworks-sdk-mini-schema", "", "explicit MiniMax schema diagnostic: default or adaptive; two SDK HTTP requests")
)

func sdkRuntimeModel(t *testing.T, provider *fireworks.Provider, wire *sdkRuntimeWire, offering catalog.Offering, maxTokens int, reasoning bool) llm.Model {
	t.Helper()

	wire.maxOutput = maxTokens
	api := offering.WireAPIs[0]
	require.Equal(t, catalog.WireProtocolAnthropicMessages, api.Protocol)
	require.Equal(t, catalog.WireTransportDirect, api.Transport)
	support, ok := offering.SupportForWireAPI(api)
	require.True(t, ok)

	opts := []fireworks.Option{fireworks.WithMaxTokens(maxTokens), fireworks.WithTemperature(0)}
	if support.Reasoning.Adaptive {
		opts = append(opts, fireworks.WithThinking(reasoning))
	} else {
		for _, effort := range support.Reasoning.Efforts {
			if reasoning && effort == fireworks.ReasoningEffortNone {
				continue
			}

			opts = append(opts, fireworks.WithReasoningEffort(effort))

			break
		}
	}

	model, err := provider.NewModel(offering.ID, opts...)
	require.NoError(t, err)
	require.False(t, model.Capabilities().Vision, "native vision support cannot be encoded by llm.Part")

	return model
}

func sdkRuntimeConversation(t *testing.T, provider *fireworks.Provider, wire *sdkRuntimeWire, offering catalog.Offering, stream bool) {
	t.Helper()
	model := sdkRuntimeModel(t, provider, wire, offering, 512, false)
	request := &llm.Request{Messages: []llm.Message{
		llm.NewMessage(llm.RoleSystem, llm.NewTextPart("Include the word tulip in your final answer, alongside the answer to the user's question.")),
		sdkRuntimeUser("My favorite animal is an otter. Remember that."),
		llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("I will remember your favorite animal.")),
		sdkRuntimeUser("What is my favorite animal?"),
	}}
	response := sdkRuntimeCall(t, wire, model, request, stream, "sdk_system_multiturn")
	require.Equal(t, llm.FinishReasonStop, response.FinishReason)
	require.Contains(t, strings.ToLower(response.TextContent()), "otter")
	require.Contains(t, strings.ToLower(response.TextContent()), "tulip")
}

func sdkRuntimeReasoningTools(t *testing.T, provider *fireworks.Provider, wire *sdkRuntimeWire, offering catalog.Offering, stream bool) {
	t.Helper()
	model := sdkRuntimeModel(t, provider, wire, offering, 4096, true)
	name := "lookup"
	question := sdkRuntimeUser(contractAdaptiveTask + " Then call lookup with key x exactly once. Its reply is JSON data. After receiving it, state the exact result field value and finish with the computed n as a plain integer.")
	request := &llm.Request{
		Messages:   []llm.Message{question},
		Tools:      []llm.ToolDefinition{{Name: name, Description: "Look up a word by key.", Parameters: json.RawMessage(`{"type":"object","properties":{"key":{"type":"string","enum":["x"]}},"required":["key"],"additionalProperties":false}`)}},
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceSpecific, Name: &name},
	}
	first := sdkRuntimeCall(t, wire, model, request, stream, "sdk_reasoning_tool")
	require.Equal(t, llm.FinishReasonToolCalls, first.FinishReason)
	calls := first.ToolRequests()
	require.Len(t, calls, 1)
	require.Equal(t, name, calls[0].Name)
	require.JSONEq(t, `{"key":"x"}`, string(calls[0].Arguments))

	reasoning := false

	for _, part := range first.Message.Content {
		if trace, ok := part.(*llm.ReasoningPart); ok && trace.Text != "" {
			reasoning = true
		}
	}

	require.True(t, reasoning || first.Usage.ReasoningTokens > 0, "explicit reasoning needs an observable trace or authoritative reasoning tokens")
	request.Messages = []llm.Message{question, first.Message, llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(calls[0].ID, name, json.RawMessage(`{"result":"violet-lantern-zebra"}`), false))}
	request.ToolChoice = &llm.ToolChoice{Type: llm.ToolChoiceAuto}
	second := sdkRuntimeCall(t, wire, model, request, stream, "sdk_reasoning_tool_history")
	require.Equal(t, llm.FinishReasonStop, second.FinishReason)
	require.Empty(t, second.ToolRequests())
	require.Contains(t, second.TextContent(), "violet-lantern-zebra")
	contractAssertAdaptiveAnswer(t, second.TextContent())
}

func sdkRuntimeSchema(t *testing.T, provider *fireworks.Provider, wire *sdkRuntimeWire, offering catalog.Offering, stream bool) {
	t.Helper()

	var model llm.Model

	if offering.ID == fireworks.ModelMiniMaxM3 {
		wire.maxOutput = 512
		var err error

		model, err = provider.NewModel(offering.ID, fireworks.WithMaxTokens(512), fireworks.WithTemperature(0))
		require.NoError(t, err)
	} else {
		model = sdkRuntimeModel(t, provider, wire, offering, 512, false)
	}

	request := sdkRuntimeSchemaRequest()
	format := request.ResponseFormat
	request.ResponseFormat = nil
	plain := sdkRuntimeCall(t, wire, model, request, stream, "sdk_schema_plain_control")
	require.Equal(t, llm.FinishReasonStop, plain.FinishReason)
	require.JSONEq(t, `{"gate":"PROMPT_ONLY","count":"wrong","extra":true}`, plain.TextContent())

	request.ResponseFormat = format
	controlled := sdkRuntimeCall(t, wire, model, request, stream, "sdk_schema_enforced")
	require.Equal(t, llm.FinishReasonStop, controlled.FinishReason)
	require.JSONEq(t, `{"gate":"SCHEMA_ACTIVE_73","count":73}`, controlled.TextContent())
}

func sdkRuntimeTruncation(t *testing.T, provider *fireworks.Provider, wire *sdkRuntimeWire, offering catalog.Offering, stream bool) {
	t.Helper()
	model := sdkRuntimeModel(t, provider, wire, offering, 1, false)
	response := sdkRuntimeCall(t, wire, model, &llm.Request{Messages: []llm.Message{sdkRuntimeUser("Write the numbers from 1 to 10000, separated by commas. Do not stop early.")}}, stream, "sdk_length")
	require.Equal(t, llm.FinishReasonLength, response.FinishReason)
	require.Equal(t, 1, response.Usage.OutputTokens+response.Usage.ReasoningTokens)
}

func sdkRuntimeCacheProbe(t *testing.T, provider *fireworks.Provider, wire *sdkRuntimeWire) {
	t.Helper()
	report, _ := contractReadReport(t, "testdata/contracts/cache-diagnostic-65536.json")
	row := report.Evidence[0]
	offering, ok := provider.Catalog().Lookup(row.Case.Model)
	require.True(t, ok)
	testutil.SkipIfRetired(t, provider.Catalog(), offering.ID)

	wire.limit = 72 * 1024
	wire.header = http.Header{"X-Session-Affinity": {"7a5c2e10-9b84-4f61-8c33-0d72e906ab51"}, "X-Claude-Code-Session-Id": {"7a5c2e10-9b84-4f61-8c33-0d72e906ab51"}, "X-Prompt-Cache-Isolation-Key": {"sdk-cache-crossover-v5"}}
	model := sdkRuntimeModel(t, provider, wire, offering, 16, false)
	var body map[string]any
	require.NoError(t, json.Unmarshal(row.Exchanges[0].Request, &body))
	prompt := contractString(contractObject(contractArray(body["messages"])[0])["content"])
	hit := false

	for n := range 3 {
		if n > 0 {
			time.Sleep(2 * time.Second)
		}

		response := sdkRuntimeCall(t, wire, model, &llm.Request{Messages: []llm.Message{sdkRuntimeUser(prompt)}}, false, "sdk_positive_cache")
		require.Equal(t, llm.FinishReasonStop, response.FinishReason)

		hit = hit || response.Usage.CachedInputTokens > 0
		if n > 0 && hit {
			break
		}
	}

	require.True(t, hit, "bounded actual SDK cache probe was inconclusive; zero is not a cache-hit observation")
}

//nolint:paralleltest,tparallel // Paid SDK calls deliberately run sequentially and reject automatic retries.
func TestFireworksSDKLiveFirstBinding(t *testing.T) {
	t.Parallel()

	if !*sdkRuntimeLive {
		t.Skip("explicit local-only SDK runtime integration; no live coverage is claimed by a skip")
	}

	require.False(t, testing.Short(), "explicit SDK live integration must not run in short mode")
	require.NotEmpty(t, *sdkRuntimeKeyFile)
	raw, err := os.ReadFile(*sdkRuntimeKeyFile)
	require.NoError(t, err)

	key := strings.TrimSpace(string(raw))
	require.NotEmpty(t, key)
	require.False(t, strings.ContainsAny(key, "\r\n"))

	wire := newSDKRuntimeWire(t, key, "testdata/sdk-runtime/messages-live.json", 16*1024)
	defer wire.checkpoint(t)

	client := &http.Client{Transport: wire, Timeout: 90 * time.Second, CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }}
	provider, err := fireworks.NewProvider(t.Context(), fireworks.WithAPIKey(key), fireworks.WithHTTPClient(client))
	require.NoError(t, err)

	for _, offering := range provider.Catalog().All() {
		for _, stream := range []bool{false, true} {
			t.Run(offering.ID+fmt.Sprintf("/stream=%t", stream), func(t *testing.T) {
				testutil.SkipIfRetired(t, provider.Catalog(), offering.ID)
				t.Run("system_multiturn", func(t *testing.T) { sdkRuntimeConversation(t, provider, wire, offering, stream) })
				t.Run("reasoning_tools", func(t *testing.T) { sdkRuntimeReasoningTools(t, provider, wire, offering, stream) })
				t.Run("schema", func(t *testing.T) { sdkRuntimeSchema(t, provider, wire, offering, stream) })
				t.Run("length", func(t *testing.T) { sdkRuntimeTruncation(t, provider, wire, offering, stream) })
			})
		}
	}

	if *sdkRuntimeCache {
		t.Run("actual_sdk_cache", func(t *testing.T) { sdkRuntimeCacheProbe(t, provider, wire) })
	}

	t.Logf("ACTUAL SDK first-binding run: %d HTTP requests; known native-usage USD %.9f", len(wire.report.Evidence), wire.report.KnownRunUSD)
}

//nolint:paralleltest,tparallel // Paid SDK calls deliberately run sequentially and reject automatic retries.
func TestFireworksSDKLiveMiniMaxSchema(t *testing.T) {
	t.Parallel()

	if !*sdkRuntimeLive || *sdkRuntimeMiniSchema == "" {
		t.Skip("explicit local-only two-request MiniMax SDK schema diagnostic; no live coverage is claimed by a skip")
	}

	require.False(t, testing.Short(), "explicit SDK live integration must not run in short mode")
	require.Contains(t, []string{"default", "adaptive"}, *sdkRuntimeMiniSchema)
	require.NotEmpty(t, *sdkRuntimeKeyFile)
	raw, err := os.ReadFile(*sdkRuntimeKeyFile)
	require.NoError(t, err)

	key := strings.TrimSpace(string(raw))
	require.NotEmpty(t, key)
	require.False(t, strings.ContainsAny(key, "\r\n"))

	wire := newSDKRuntimeWire(t, key, "testdata/sdk-runtime/minimax-schema-"+*sdkRuntimeMiniSchema+".json", 2*1024)
	wire.maxOutput = 512
	wire.report.RunKind = "live_sdk_minimax_schema_diagnostic"
	wire.report.Blocked = append(wire.report.Blocked, "Partial corrective diagnostic only; the original 86-request messages-live.json remains a failed SDK batch, including both MiniMax schema cases.")

	defer wire.checkpoint(t)

	client := &http.Client{Transport: wire, Timeout: 90 * time.Second, CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }}
	provider, err := fireworks.NewProvider(t.Context(), fireworks.WithAPIKey(key), fireworks.WithHTTPClient(client))
	require.NoError(t, err)
	testutil.SkipIfRetired(t, provider.Catalog(), fireworks.ModelMiniMaxM3)

	opts := []fireworks.Option{fireworks.WithMaxTokens(512), fireworks.WithTemperature(0)}
	if *sdkRuntimeMiniSchema == "adaptive" {
		opts = append(opts, fireworks.WithThinking(true))
	}

	model, err := provider.NewModel(fireworks.ModelMiniMaxM3, opts...)
	require.NoError(t, err)

	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			request := sdkRuntimeSchemaRequest()
			response := sdkRuntimeCall(t, wire, model, request, stream, "sdk_schema_"+*sdkRuntimeMiniSchema)
			require.Equal(t, llm.FinishReasonStop, response.FinishReason)
			require.JSONEq(t, `{"gate":"SCHEMA_ACTIVE_73","count":73}`, response.TextContent())
		})
	}

	t.Logf("ACTUAL SDK MiniMax %s schema: %d HTTP requests; known native-usage USD %.9f", *sdkRuntimeMiniSchema, len(wire.report.Evidence), wire.report.KnownRunUSD)
}
