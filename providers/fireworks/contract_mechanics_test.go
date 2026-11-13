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
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func TestFireworksContractSyntheticCancellationMechanics(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			return
		}

		<-r.Context().Done()
	}))
	defer server.Close()

	c := contractCase{Model: fireworks.Catalog().All()[0].ID, API: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect, Feature: contractCancel}
	e := contractHTTP(t, server.Client(), server.URL, "synthetic-only-not-a-credential", c, contractRequest(t, c), nil)
	require.True(t, e.RequestWritten)
	require.True(t, e.Cancelled)
	require.Empty(t, e.ErrorClass)
	require.Empty(t, e.Response)
}

func TestFireworksContractSyntheticLifecycleBoundaries(t *testing.T) {
	t.Parallel()

	cat := contractCatalogWithNativeOnlyOffering(t)
	for _, o := range cat.All() {
		if o.Life.Retires.IsZero() {
			continue
		}

		t.Run(o.ID, func(t *testing.T) {
			t.Parallel()

			before := contractActiveOfferings(cat, o.Life.Retires.Add(-24*time.Hour), "")
			on := contractActiveOfferings(cat, o.Life.Retires, "")
			found := false

			for _, c := range before {
				if c.ID == o.ID {
					found = true
				}
			}

			require.True(t, found)

			for _, c := range on {
				require.NotEqual(t, o.ID, c.ID, "retired offering must not produce paid live cases")
			}
		})
	}
}

func contractCatalogWithNativeOnlyOffering(t *testing.T) *catalog.Catalog {
	t.Helper()

	offerings := fireworks.Catalog().All()

	entries := make([]catalog.Entry, 0, len(offerings)+1)
	for _, o := range offerings {
		entries = append(entries, o.Entry)
	}

	extra := entries[0]
	extra.ID = "accounts/fireworks/models/native-only"
	extra.Aliases = nil
	extra.Life = catalog.Lifecycle{}
	extra.Capabilities = llm.ModelCapabilities{}
	extra.Reasoning = catalog.ReasoningSupport{}
	extra.Modalities = catalog.Modalities{Input: []catalog.Modality{catalog.ModalityText}, Output: []catalog.Modality{catalog.ModalityText}}
	extra.WireAPIs = []catalog.WireAPI{{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect}}
	extra.WireAPISupport = []catalog.WireAPISupport{{API: extra.WireAPIs[0]}}
	extra.Pricing = pricing.TieredInfo(pricing.NewRates(1, 2, .5), pricing.Bracket{MinContextTokens: 4096, Rates: pricing.NewRates(2, 4, 1)})
	entries = append(entries, extra)
	cat, err := catalog.New(fireworks.ProviderName, entries, catalog.WithRequiredWireAPIs())
	require.NoError(t, err)

	return cat
}

func TestFireworksContractSyntheticSelectedInventory(t *testing.T) {
	t.Parallel()

	cat := contractCatalogWithNativeOnlyOffering(t)
	model := fireworks.ModelDeepSeekV41Flash1017
	cases := contractInventoryAt(t, cat, catalog.Today(), model)
	require.NotEmpty(t, cases)
	for _, c := range cases {
		require.Equal(t, model, c.Model)
	}
}

func TestFireworksContractSyntheticRequestControls(t *testing.T) {
	t.Parallel()

	manifest := contractLoadManifest(t)
	for _, evidence := range manifest.Evidence {
		c := evidence.Case
		request := contractRequest(t, c)
		if c.API == catalog.WireProtocolOpenAIResponses {
			require.Equal(t, false, request["store"])
		}

		if c.API == catalog.WireProtocolAnthropicMessages && c.Effort == contractEffortNone {
			require.Equal(t, "disabled", contractObject(request["thinking"])["type"])
			require.Nil(t, contractObject(request["output_config"])["effort"])
		}

		if c.API == catalog.WireProtocolAnthropicMessages && c.Feature == contractStructured {
			require.Equal(t, "json_schema", contractObject(contractObject(request["output_config"])["format"])["type"])
		}

		if c.Feature == contractCancel {
			output := request["max_tokens"]
			if c.API == catalog.WireProtocolOpenAIResponses {
				output = request["max_output_tokens"]
			}

			require.Equal(t, 16, output)
		}
	}
}
