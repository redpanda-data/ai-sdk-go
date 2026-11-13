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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func sdkRuntimeExpandedReasoningCatalog(t *testing.T, model string, efforts []llm.ReasoningEffort, adaptive bool) *catalog.Catalog {
	t.Helper()

	offerings := fireworks.Catalog().All()
	entries := make([]catalog.Entry, 0, len(offerings))

	for _, offering := range offerings {
		entry := offering.Entry
		if entry.ID == model {
			entry.Reasoning.Efforts = efforts
			entry.Reasoning.Adaptive = adaptive

			for i := range entry.WireAPISupport {
				entry.WireAPISupport[i].Reasoning.Efforts = efforts
				if entry.WireAPISupport[i].API.Protocol != catalog.WireProtocolOpenAIResponses {
					entry.WireAPISupport[i].Reasoning.Adaptive = adaptive
				}
			}
		}

		entries = append(entries, entry)
	}

	cat, err := catalog.New(fireworks.ProviderName, entries, catalog.WithRequiredWireAPIs())
	require.NoError(t, err)

	return cat
}

func TestFireworksSDKReplayPreservesRecordedReasoning(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name     string
		model    string
		efforts  []llm.ReasoningEffort
		adaptive bool
	}{
		{name: "lower_effort", model: fireworks.ModelDeepSeekV41Flash, efforts: []llm.ReasoningEffort{"none", "minimal", "low", "high", "max"}},
		{name: "disabled_effort", model: fireworks.ModelGLM53Flash, efforts: []llm.ReasoningEffort{"none", "low", "high", "max"}},
		{name: "adaptive_control", model: fireworks.ModelGLM53Flash, efforts: []llm.ReasoningEffort{"low", "high", "max"}, adaptive: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			cat := sdkRuntimeExpandedReasoningCatalog(t, test.model, test.efforts, test.adaptive)
			sdkRuntimeReplayCapturedFeatures(t, cat)
		})
	}
}

func TestFireworksRecordedCasePreservesReasoningControls(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name    string
		api     catalog.WireProtocol
		request string
		effort  string
	}{
		{name: "messages_omitted", api: catalog.WireProtocolAnthropicMessages, request: `{}`},
		{name: "messages_disabled", api: catalog.WireProtocolAnthropicMessages, request: `{"thinking":{"type":"disabled"}}`, effort: "none"},
		{name: "messages_effort", api: catalog.WireProtocolAnthropicMessages, request: `{"output_config":{"effort":"low"}}`, effort: "low"},
		{name: "chat_omitted", api: catalog.WireProtocolOpenAIChatCompletions, request: `{}`},
		{name: "chat_disabled", api: catalog.WireProtocolOpenAIChatCompletions, request: `{"reasoning_effort":"none"}`, effort: "none"},
		{name: "chat_effort", api: catalog.WireProtocolOpenAIChatCompletions, request: `{"reasoning_effort":"low"}`, effort: "low"},
		{name: "responses_omitted", api: catalog.WireProtocolOpenAIResponses, request: `{}`},
		{name: "responses_disabled", api: catalog.WireProtocolOpenAIResponses, request: `{"reasoning":{"effort":"none"}}`, effort: "none"},
		{name: "responses_effort", api: catalog.WireProtocolOpenAIResponses, request: `{"reasoning":{"effort":"low"}}`, effort: "low"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			evidence := contractEvidence{
				Case:      contractCase{Model: fireworks.ModelGLM53Flash, API: test.api, Transport: catalog.WireTransportDirect, Feature: contractComplete, Effort: "high"},
				Options:   contractRequestOptions{UserPrompt: "Retain the recorded request options.", MaxOutputTokens: 73},
				Exchanges: []contractExchange{{Request: json.RawMessage(test.request)}},
			}
			original, err := json.Marshal(evidence.Case)
			require.NoError(t, err)
			recorded := contractRecordedCase(t, evidence)
			require.NotNil(t, recorded.replayEffort)
			assert.Equal(t, test.effort, contractEffort(recorded))
			assert.Equal(t, evidence.Options, recorded.Options)
			assert.Equal(t, evidence.Case.key(), recorded.key())
			encoded, err := json.Marshal(recorded)
			require.NoError(t, err)
			assert.JSONEq(t, string(original), string(encoded))

			request := contractRequest(t, recorded)
			rebuilt := contractEvidence{Case: evidence.Case, Exchanges: []contractExchange{{}}}
			rebuilt.Exchanges[0].Request, err = json.Marshal(request)
			require.NoError(t, err)
			assert.Equal(t, test.effort, contractEffort(contractRecordedCase(t, rebuilt)))
		})
	}
}

func TestFireworksRecordedReasoningIgnoresCatalogDefaults(t *testing.T) {
	t.Parallel()

	cat := sdkRuntimeExpandedReasoningCatalog(t, fireworks.ModelDeepSeekV41Flash, []llm.ReasoningEffort{"none", "minimal", "low", "high", "max"}, false)
	report := contractReadReport(t, "testdata/contracts/revalidated-evidence.json")
	var matched int

	for _, row := range report.Evidence {
		if row.Case.Model != fireworks.ModelDeepSeekV41Flash || row.Case.Feature != contractReasoningHistory {
			continue
		}

		matched++

		t.Run(row.Case.key(), func(t *testing.T) {
			t.Parallel()
			require.Empty(t, row.Case.Effort)
			assert.Equal(t, "minimal", contractEffortForCatalog(row.Case, cat))
			recorded := contractRecordedCase(t, row)
			assert.Equal(t, "low", contractEffortForCatalog(recorded, cat))
			contractReplay(t, row, 0)
		})
	}

	require.Positive(t, matched)
}
