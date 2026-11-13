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
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func TestProvider(t *testing.T) {
	t.Parallel()

	p, err := fireworks.NewProvider(context.Background())
	require.NoError(t, err)
	assert.Equal(t, llm.ProviderID("fireworks"), p.Name())
	assert.Same(t, fireworks.Catalog(), p.Catalog())
	assert.Same(t, p.Catalog(), (&fireworks.Provider{}).Catalog())
	assert.Equal(t, p.Name(), p.Catalog().Provider())
}

func TestInventoryAndCreatorFacts(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		model     catalog.ModelID
		publisher catalog.Publisher
		input     int
		output    int
		vision    bool
	}{
		fireworks.ModelGLM53:                {catalog.ModelGLM53, catalog.PublisherZAI, 1048576, 131072, false},
		fireworks.ModelGLM53Flash:           {catalog.ModelGLM53Flash, catalog.PublisherZAI, 1048576, 131072, true},
		fireworks.ModelMiniMaxM3:            {catalog.ModelMiniMaxM3, catalog.PublisherMiniMax, 512000, 512000, false},
		fireworks.ModelGPTOSS120B:           {catalog.ModelGPTOSS120B, catalog.PublisherOpenAI, 131072, 131072, false},
		fireworks.ModelDeepSeekV41Flash:     {catalog.ModelDeepSeekV41Flash, catalog.PublisherDeepSeek, 1048576, 393216, true},
		fireworks.ModelDeepSeekV41Flash1017: {catalog.ModelDeepSeekV41Flash1017, catalog.PublisherDeepSeek, 1048576, 393216, true},
		fireworks.ModelKimiK3:               {catalog.ModelKimiK3, catalog.PublisherMoonshotAI, 1048576, 1048576, true},
	}
	for id, want := range cases {
		o, ok := fireworks.Catalog().Lookup(id)
		require.True(t, ok, "missing example offering: %s", id)
		assert.Equal(t, want.model, o.Model)
		assert.Equal(t, want.publisher, o.Facts().Publisher)
		assert.NotEqual(t, catalog.PublisherAnthropic, o.Facts().Publisher)
		assert.Empty(t, o.Capabilities.ToolSearchProtocol, "ordinary tools do not imply hosted tool search")
		assert.True(t, o.Facts().OpenWeights)
		assert.Equal(t, want.input, o.Constraints.MaxInputTokens)
		assert.Equal(t, want.output, o.Constraints.MaxOutputTokens)
		assert.Equal(t, want.vision, o.Capabilities.Vision)
		assert.Equal(t, want.vision, slices.Contains(o.Modalities.Input, catalog.ModalityImage))
		assert.Empty(t, o.Aliases, "moving/router aliases require independent qualification")
		assert.Equal(t, "blocked_missing_effective_tier", o.Attributes["priority_tier_status"])
		assert.ElementsMatch(t, []catalog.WireAPI{
			{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect},
			{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect},
			{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
		}, o.WireAPIs)

		for _, api := range o.WireAPIs {
			support, exists := o.SupportForWireAPI(api)
			require.True(t, exists, "API feature metadata missing: %s / %s", o.ID, api)
			assert.True(t, support.Capabilities.Streaming)
			assert.True(t, support.Capabilities.Tools)
			assert.Equal(t, want.vision, support.Capabilities.Vision)
		}

		messages, exists := o.SupportForWireAPI(catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect})
		require.True(t, exists)
		assert.False(t, messages.Capabilities.JSONMode)
	}

	oss, ok := fireworks.Catalog().Lookup(fireworks.ModelGPTOSS120B)
	require.True(t, ok)
	assert.Equal(t, catalog.MustDate("2026-11-07"), oss.Life.Retires)
	assert.Empty(t, oss.Life.ReplacedBy)
	assert.Equal(t, catalog.DefaultRegistry()[catalog.ModelGPTOSS120B], oss.Facts())
}

func TestCatalogMetadataDoesNotAdmitRuntimeControls(t *testing.T) {
	t.Parallel()

	for _, offering := range fireworks.Catalog().All() {
		require.Nil(t, offering.Constraints.SupportedParams)

		for _, param := range []string{"temperature", "top_p", "max_tokens", "stop", "tools", "reasoning_effort"} {
			require.Error(t, offering.Constraints.ValidateParameterSupport(param), "native catalog metadata does not admit SDK %s", param)
		}

		assert.NotEmpty(t, offering.WireAPIs, "native API discovery remains explicit")
		assert.Equal(t, offering.Capabilities.Vision, slices.Contains(offering.Modalities.Input, catalog.ModalityImage))

		for _, api := range offering.WireAPIs {
			_, exists := offering.SupportForWireAPI(api)
			assert.True(t, exists, "API feature metadata missing: %s / %s", offering.ID, api)
		}
	}
}

func TestUnqualifiedIdentitiesRemainUnknown(t *testing.T) {
	t.Parallel()

	cat, err := pricing.NewCatalog(pricing.WithSource(fireworks.Catalog()))
	require.NoError(t, err)

	for _, id := range []string{
		"auto", "firerouter/auto", "glm-latest", "accounts/fireworks/routers/glm-latest",
		"accounts/fireworks/routers/glm-5p3-fast", "accounts/fireworks/models/gpt-oss-20b",
		"accounts/private/deployments/glm-5p3", fireworks.ModelGLM53 + "@deployment",
		fireworks.ModelGLM53 + "-20261104",
	} {
		_, ok := fireworks.Catalog().Resolve(id)
		assert.False(t, ok, id)
		_, err := cat.Lookup(fireworks.ProviderName, id)
		require.ErrorIs(t, err, pricing.ErrUnknownModel, id)
	}

	_, err = cat.Lookup("anthropic", fireworks.ModelGPTOSS120B)
	assert.ErrorIs(t, err, pricing.ErrUnknownProvider)
}

func TestPricingNamespaceIsolation(t *testing.T) {
	t.Parallel()

	offering, ok := fireworks.Catalog().Lookup(fireworks.ModelGPTOSS120B)
	require.True(t, ok)

	otherEntry := offering.Entry
	otherEntry.Pricing = pricing.FlatInfo(2, 3, 4)
	other, err := catalog.New("test.host", []catalog.Entry{otherEntry})
	require.NoError(t, err)
	combined, err := pricing.NewCatalog(pricing.WithSource(fireworks.Catalog()), pricing.WithSource(other))
	require.NoError(t, err)

	viaFireworks, err := combined.Lookup(fireworks.ProviderName, offering.ID)
	require.NoError(t, err)
	viaOther, err := combined.Lookup("test.host", offering.ID)
	require.NoError(t, err)
	assert.Equal(t, offering.Pricing, viaFireworks)
	assert.NotEqual(t, viaFireworks, viaOther)

	otherOffering, ok := other.Lookup(offering.ID)
	require.True(t, ok)
	assert.Equal(t, offering.Facts(), otherOffering.Facts())
}

func TestCatalogCollectionsAreIsolated(t *testing.T) {
	t.Parallel()

	first, ok := fireworks.Catalog().Lookup(fireworks.ModelGLM53)
	require.True(t, ok)

	first.WireAPIs[0] = catalog.WireAPI{Protocol: catalog.WireProtocolBedrockConverse, Transport: catalog.WireTransportBedrockRuntime}
	first.WireAPISupport[0] = catalog.WireAPISupport{}
	first.Attributes["pricing_checked"] = "mutated"
	first.Pricing.Overrides[0].RateCard.Base.InputPerMillion = 1
	first.Reasoning.Efforts[0] = "mutated"

	again, ok := fireworks.Catalog().Lookup(fireworks.ModelGLM53)
	require.True(t, ok)
	assert.NotContains(t, again.WireAPIs, catalog.WireAPI{Protocol: catalog.WireProtocolBedrockConverse, Transport: catalog.WireTransportBedrockRuntime})
	messages, exists := again.SupportForWireAPI(catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect})
	require.True(t, exists)
	assert.True(t, messages.Capabilities.Tools)
	assert.Equal(t, "2026-11-04", again.Attributes["pricing_checked"])
	assert.NotEqual(t, int64(1), again.Pricing.Overrides[0].RateCard.Base.InputPerMillion)
	assert.Equal(t, llm.ReasoningEffort("low"), again.Reasoning.Efforts[0])
}
