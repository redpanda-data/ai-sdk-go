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
	"math"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func contractOriginalCaptures(t *testing.T) contractManifest {
	t.Helper()

	b, err := os.ReadFile("testdata/contracts/original-incomplete.json")
	require.NoError(t, err)
	var manifest contractManifest
	require.NoError(t, json.Unmarshal(b, &manifest))

	for i := range manifest.Evidence {
		e := &manifest.Evidence[i]
		switch string(e.Case.API) {
		case "anthropic.messages":
			e.Case.API = catalog.WireProtocolAnthropicMessages
		case "openai.chat_completions":
			e.Case.API = catalog.WireProtocolOpenAIChatCompletions
		case "openai.responses":
			e.Case.API = catalog.WireProtocolOpenAIResponses
		}

		e.Case.Transport = catalog.WireTransportDirect
	}

	return manifest
}

func TestFireworksContractOriginalCharacterization(t *testing.T) {
	t.Parallel()

	manifest := contractOriginalCaptures(t)
	for _, evidence := range manifest.Evidence {
		t.Run(evidence.Case.key(), func(t *testing.T) {
			t.Parallel()

			for _, exchange := range evidence.Exchanges {
				require.NotContains(t, string(exchange.Request), "Authorization")

				for name := range exchange.RequestHeaders {
					require.Contains(t, []string{"Content-Type", "Anthropic-Version"}, name)
				}

				for name := range exchange.ResponseHeaders {
					require.Equal(t, "Content-Type", name)
				}

				if exchange.Status != 200 {
					continue
				}

				r := contractParse(t, evidence.Case, exchange)
				contractAssertUsage(t, evidence.Case, r)
				contractAssertPrice(t, evidence.Case, r)
			}
		})
	}
}

func TestFireworksContractSyntheticUsageExtensions(t *testing.T) {
	t.Parallel()

	for _, api := range []catalog.WireProtocol{catalog.WireProtocolAnthropicMessages, catalog.WireProtocolOpenAIChatCompletions, catalog.WireProtocolOpenAIResponses} {
		t.Run(string(api), func(t *testing.T) {
			t.Parallel()

			input, output := "input_tokens", "output_tokens"
			if api == catalog.WireProtocolOpenAIChatCompletions {
				input, output = "prompt_tokens", "completion_tokens"
			}

			r := contractResult{Input: 13, Cached: 5, Output: 3, HasInput: true, HasOutput: true}
			contractUsage(&r, api, map[string]any{output: float64(7)})
			require.Equal(t, int64(13), r.Input)
			require.Equal(t, int64(5), r.Cached)
			require.Equal(t, int64(7), r.Output)
			contractUsage(&r, api, map[string]any{input: float64(0), output: float64(0), "cache_read_input_tokens": float64(0)})
			require.Zero(t, r.Input)
			require.Zero(t, r.Cached)
			require.Zero(t, r.Output)

			final := map[string]any{input: float64(75), output: float64(10)}
			contractUsage(&r, api, final)
			contractUsage(&r, api, final)
			require.Equal(t, int64(75), r.Input, "cumulative records replace, never sum")
			require.Equal(t, int64(10), r.Output)
		})
	}
}

func TestFireworksContractSyntheticIdentityExtensions(t *testing.T) {
	t.Parallel()

	manifest, _ := contractReadReport(t, "testdata/contracts/full-matrix-v2-incomplete.json")
	for _, api := range []catalog.WireProtocol{catalog.WireProtocolAnthropicMessages, catalog.WireProtocolOpenAIChatCompletions, catalog.WireProtocolOpenAIResponses} {
		t.Run(string(api), func(t *testing.T) {
			t.Parallel()
			var chosen contractEvidence

			for _, e := range manifest.Evidence {
				if e.Case.API == api && e.Case.Stream && e.Case.Feature == contractComplete && !e.Failed {
					chosen = e
					break
				}
			}

			require.NotEmpty(t, chosen.Exchanges)

			for _, test := range []struct {
				name, first, last, want string
				usageOnly               bool
			}{
				{"final_only", "", chosen.Case.Model, chosen.Case.Model, false},
				{"early_survives_omissions", chosen.Case.Model, "", chosen.Case.Model, false},
				{"later_replaces_earlier", "unpriced-early", chosen.Case.Model, chosen.Case.Model, false},
				{"usage_only_final_identity", "", chosen.Case.Model, chosen.Case.Model, true},
				{"unknown_is_retained", "", "unpriced-reported-model", "unpriced-reported-model", true},
			} {
				t.Run(test.name, func(t *testing.T) {
					t.Parallel()

					e := chosen.Exchanges[0]
					e.Response = contractIdentityVariant(t, api, e.Response, test.first, test.last, test.usageOnly)
					r := contractParse(t, chosen.Case, e)
					require.Equal(t, test.want, r.Model)

					if test.name == "unknown_is_retained" {
						pc, err := pricing.NewCatalog(pricing.WithSource(fireworks.Catalog()))
						require.NoError(t, err)
						_, err = pc.Lookup(fireworks.ProviderName, r.Model)
						require.ErrorIs(t, err, pricing.ErrUnknownModel, "unknown observed identity must never become the requested priced identity")
					}
				})
			}
		})
	}
}

func contractIdentityVariant(t *testing.T, api catalog.WireProtocol, raw, first, last string, usageOnly bool) string {
	t.Helper()

	lines := strings.Split(raw, "\n")
	indices := []int{}
	bodies := map[int]map[string]any{}
	usageIndex := -1

	for i, line := range lines {
		if !strings.HasPrefix(line, "data:") || strings.Contains(line, contractSSEDone) {
			continue
		}
		var body map[string]any
		require.NoError(t, json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &body))
		delete(body, "model")
		delete(contractObject(body["message"]), "model")
		delete(contractObject(body["response"]), "model")

		if body["usage"] != nil {
			usageIndex = i
		}

		indices = append(indices, i)
		bodies[i] = body
	}

	require.NotEmpty(t, indices)

	setModel := func(body map[string]any, value string) {
		if value == "" {
			return
		}

		if nested := contractObject(body["message"]); nested != nil {
			nested["model"] = value
			return
		}

		if nested := contractObject(body["response"]); nested != nil {
			nested["model"] = value
			return
		}

		body["model"] = value
	}
	setModel(bodies[indices[0]], first)
	target := indices[len(indices)-1]

	if usageOnly && api != catalog.WireProtocolOpenAIResponses {
		require.NotEqual(t, -1, usageIndex)
		target = usageIndex
	}

	if !usageOnly || api != catalog.WireProtocolOpenAIResponses {
		setModel(bodies[target], last)
	}

	for _, i := range indices {
		b, err := json.Marshal(bodies[i])
		require.NoError(t, err)

		lines[i] = "data: " + string(b)
	}

	if usageOnly && api == catalog.WireProtocolOpenAIResponses {
		metadata := map[string]any{"type": "response.in_progress", "response": map[string]any{"model": last, "output": []any{}, "usage": contractObject(bodies[target]["response"])["usage"]}}
		b, err := json.Marshal(metadata)
		require.NoError(t, err)

		lines[target] = "data: " + string(b) + "\n\n" + lines[target]
	}

	return strings.Join(lines, "\n")
}

func contractAssertPrice(t *testing.T, c contractCase, r contractResult) {
	t.Helper()

	pc, err := pricing.NewCatalog(pricing.WithSource(fireworks.Catalog()))
	require.NoError(t, err)

	input := r.Input
	if c.API != catalog.WireProtocolAnthropicMessages {
		input -= r.Cached
	}

	u := &llm.TokenUsage{InputTokens: int(input), CachedInputTokens: int(r.Cached), OutputTokens: int(r.Output - r.Reasoned), ReasoningTokens: int(r.Reasoned)}
	cost, err := pc.Calculate(fireworks.ProviderName, r.Model, u, pricing.CalcRequest{})
	require.NoError(t, err, "unknown response identity is unpriced, never replaced with the requested ID")
	require.Empty(t, cost.Unpriced)

	expectedMicrocents := contractCost(c, r.Input, r.Cached, r.Output) * 1e8
	require.LessOrEqual(t, math.Abs(float64(cost.Total)-expectedMicrocents), float64(3), "independent native-usage arithmetic must match disjoint SDK buckets")
}
