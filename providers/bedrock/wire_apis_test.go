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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

func TestCatalogWireAPIs(t *testing.T) {
	t.Parallel()

	invoke := catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportBedrockInvokeModel}
	messages := catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportBedrockRuntime}
	converse := catalog.WireAPI{Protocol: catalog.WireProtocolBedrockConverse, Transport: catalog.WireTransportBedrockRuntime}
	chatRuntime := catalog.WireAPI{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportBedrockRuntime}
	responsesRuntime := catalog.WireAPI{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportBedrockRuntime}
	chatMantle := catalog.WireAPI{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportBedrockMantle}
	responsesMantle := catalog.WireAPI{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportBedrockMantle}

	for _, tt := range []struct {
		name string
		ids  []string
		want []catalog.WireAPI
	}{
		{
			name: "versioned InvokeModel without native Messages endpoint",
			ids:  []string{ModelClaudeSonnet45US, ModelClaudeSonnet46Global, ModelClaudeOpus45EU, ModelClaudeOpus46AU},
			want: []catalog.WireAPI{invoke, converse},
		},
		{
			name: "native runtime Messages and versioned InvokeModel are separate bindings",
			ids: []string{
				ModelClaudeOpus47US, ModelClaudeOpus48JP, ModelClaudeOpus5AU, ModelClaudeOpus55EU,
				ModelClaudeSonnet5Global, ModelClaudeSonnet55EU, ModelClaudeSonnet55AU, ModelClaudeHaiku45US, ModelClaudeHaiku55JP,
				ModelClaudeFable5US, ModelClaudeFable5Global, ModelClaudeFable51Global,
			},
			want: []catalog.WireAPI{invoke, messages, converse},
		},
		{
			name: "Fable EU wire support is unknown",
			ids:  []string{ModelClaudeFable5EU},
			want: nil,
		},
		{
			name: "Gemma 4 has only Mantle OpenAI bindings",
			ids:  []string{ModelGemma431B, ModelGemma426BA4B, ModelGemma4E2B},
			want: []catalog.WireAPI{chatMantle, responsesMantle},
		},
		{
			name: "bare OpenAI IDs have only Mantle bindings",
			ids:  []string{ModelGPT6Luna, ModelGPT6Astra, ModelGPT61Sol, ModelGPT61Terra, ModelGPT6Sol, ModelGPT56Sol, ModelGPT56Terra, ModelGPT56Luna},
			want: []catalog.WireAPI{chatMantle, responsesMantle},
		},
		{
			name: "OpenAI profile IDs have runtime bindings",
			ids: []string{
				ModelGPT6AstraUS, ModelGPT6AstraGlobal, ModelGPT61SolUS, ModelGPT61SolGlobal,
				ModelGPT61TerraUS, ModelGPT61TerraGlobal,
				ModelGPT6SolUS, ModelGPT6SolGlobal, ModelGPT6LunaUS, ModelGPT6LunaGlobal,
			},
			want: []catalog.WireAPI{converse, chatRuntime, responsesRuntime},
		},
		{
			name: "Nova native Invoke schema is not an Anthropic or OpenAI protocol",
			ids:  []string{ModelNova2LiteUS, ModelNova2LiteEU, ModelNova2LiteJP, ModelNova2LiteGlobal},
			want: []catalog.WireAPI{converse},
		},
		{
			name: "Mistral Large 3 has Chat on both endpoints but no Responses",
			ids:  []string{ModelMistralLarge3},
			want: []catalog.WireAPI{converse, chatRuntime, chatMantle},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			for _, id := range tt.ids {
				offering, ok := Catalog().Lookup(id)
				require.True(t, ok, id)
				assert.ElementsMatch(t, tt.want, offering.WireAPIs, id)
			}
		})
	}
}

func TestCatalogWireAPIUnknownIsExplicit(t *testing.T) {
	t.Parallel()

	var unknown []string

	for _, offering := range Catalog().All() {
		if len(offering.WireAPIs) == 0 {
			unknown = append(unknown, offering.ID)
		}
	}

	assert.Equal(t, []string{ModelClaudeFable5EU}, unknown)
}

func TestWireAPIsDoNotFabricateOfferingIDs(t *testing.T) {
	t.Parallel()

	for _, id := range []string{
		ModelClaudeSonnet5,
		"jp." + ModelClaudeFable5,
		"eu." + ModelGPT6Astra,
		"global." + ModelGemma431B,
		"us." + ModelGPT56Sol,
		"us." + ModelMistralLarge3,
	} {
		offering, ok := Catalog().Lookup(id)
		assert.False(t, ok, id)
		assert.Empty(t, offering.WireAPIs, id)
	}
}

func TestExpandFamiliesPreservesInvokeOnlyDeclaration(t *testing.T) {
	t.Parallel()

	apis := []catalog.WireAPI{{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportBedrockInvokeModel}}
	entries, _ := expandFamilies([]family{{
		BareID:        "anthropic.example",
		BareInvokable: true,
		WireAPIs:      apis,
	}})
	require.Len(t, entries, 1)
	assert.Equal(t, apis, entries[0].WireAPIs)
}

func TestExpandFamiliesWireAPIOverrides(t *testing.T) {
	t.Parallel()

	invoke := catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportBedrockInvokeModel}
	messages := catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportBedrockRuntime}
	entries, _ := expandFamilies([]family{{
		BareID:        "anthropic.example",
		BareInvokable: true,
		Profiles:      []string{"us", "eu"},
		WireAPIs:      []catalog.WireAPI{invoke},
		WireAPIsByID: map[string][]catalog.WireAPI{
			"anthropic.example":    {messages},
			"eu.anthropic.example": {invoke, messages},
		},
	}})
	require.Len(t, entries, 3)
	assert.Equal(t, []catalog.WireAPI{messages}, entries[0].WireAPIs)
	assert.Equal(t, []catalog.WireAPI{invoke}, entries[1].WireAPIs)
	assert.Equal(t, []catalog.WireAPI{invoke, messages}, entries[2].WireAPIs)

	entries[2].WireAPIs[0] = messages
	assert.Equal(t, []catalog.WireAPI{invoke}, entries[1].WireAPIs)
}

func TestExpandFamiliesClonesWireAPIs(t *testing.T) {
	t.Parallel()

	invoke := catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportBedrockInvokeModel}
	apis := []catalog.WireAPI{invoke}
	entries, _ := expandFamilies([]family{{
		BareID:        "anthropic.example",
		BareInvokable: true,
		Profiles:      []string{"us", "eu"},
		WireAPIs:      apis,
	}})
	require.Len(t, entries, 3)

	entries[0].WireAPIs[0].Transport = catalog.WireTransportBedrockRuntime

	assert.Equal(t, []catalog.WireAPI{invoke}, apis)
	assert.Equal(t, []catalog.WireAPI{invoke}, entries[1].WireAPIs)
	assert.Equal(t, []catalog.WireAPI{invoke}, entries[2].WireAPIs)
}

func TestExpandFamiliesRejectsUnregisteredWireAPIOverrides(t *testing.T) {
	t.Parallel()

	for _, id := range []string{"anthropic.example", "jp.anthropic.example", "eu.anthropic.other", ""} {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			f := family{
				BareID:       "anthropic.example",
				Profiles:     []string{"us", "eu"},
				WireAPIsByID: map[string][]catalog.WireAPI{id: nil},
			}
			assert.PanicsWithValue(t,
				"bedrock: family anthropic.example overrides WireAPIs for unregistered offering "+id,
				func() { expandFamilies([]family{f}) })
		})
	}
}
