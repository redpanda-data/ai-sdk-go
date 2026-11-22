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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func TestRuneIsDecisionsOnly(t *testing.T) {
	t.Parallel()

	o, ok := fireworks.Catalog().Lookup(fireworks.ModelRune)
	require.True(t, ok)
	assert.True(t, o.DecisionsOnly())
	assert.Equal(t, catalog.ModelRune26BA4B, o.Model)
	assert.Equal(t, catalog.PublisherFireworks, o.Facts().Publisher)
	assert.Equal(t, 262144, o.Constraints.MaxInputTokens)
	assert.Zero(t, o.Constraints.MaxOutputTokens)
	assert.Equal(t, []catalog.Modality{catalog.ModalityText}, o.Modalities.Input)
	assert.Empty(t, o.Modalities.Output)
	assert.Equal(t, llm.ModelCapabilities{}, o.Capabilities)
	assert.Empty(t, o.Aliases, "the rune router is a separate identity")
	assert.Equal(t, []catalog.WireAPI{
		{Protocol: catalog.WireProtocolOpenAIDecisions, Transport: catalog.WireTransportDirect},
		{Protocol: catalog.WireProtocolTypeSafeSystemOne, Transport: catalog.WireTransportDirect},
	}, o.WireAPIs)

	for _, api := range o.WireAPIs {
		support, exists := o.SupportForWireAPI(api)
		require.True(t, exists, api)
		assert.Equal(t, catalog.WireAPISupport{API: api}, support)
	}

	assert.Empty(t, o.Pricing.Overrides, "Priority is not offered")

	_, ok = fireworks.Catalog().Resolve("accounts/fireworks/routers/rune")
	assert.False(t, ok)
}

func TestNewModelRefusesRune(t *testing.T) {
	t.Parallel()

	provider, err := fireworks.NewProvider(context.Background(), fireworks.WithAPIKey("placeholder"))
	require.NoError(t, err)

	_, err = provider.NewModel(fireworks.ModelRune)
	require.ErrorIs(t, err, catalog.ErrDecisionsOnly)
	require.ErrorContains(t, err, fireworks.ModelRune)
}

// TestRuneDocumentedUsagePrice prices the usage blocks of the documented
// example responses, read 2026-11-22: System One
// (https://docs.fireworks.ai/api-reference/post-systemone) reports
// {"input_tokens": 212, "cached_input_tokens": 0, "output_tokens": 0}, and
// Decisions (https://docs.fireworks.ai/api-reference/post-decisions) reports
// input_tokens 212 with cached_tokens 0 and output_tokens 0.
func TestRuneDocumentedUsagePrice(t *testing.T) {
	t.Parallel()

	pc, err := pricing.NewCatalog(pricing.WithSource(fireworks.Catalog()))
	require.NoError(t, err)

	cost, err := pc.Calculate(fireworks.ProviderName, fireworks.ModelRune, &llm.TokenUsage{InputTokens: 212}, pricing.CalcRequest{})
	require.NoError(t, err)
	assert.Equal(t, int64(212*4_200_000/1_000_000), cost.Total, "212 tokens at $0.042/M, in microcents")
	assert.Empty(t, cost.Unpriced)

	cached, err := pc.Calculate(fireworks.ProviderName, fireworks.ModelRune, &llm.TokenUsage{InputTokens: 12, CachedInputTokens: 200, OutputTokens: 3}, pricing.CalcRequest{})
	require.NoError(t, err)
	assert.Equal(t, int64(12*4_200_000/1_000_000+200*840_000/1_000_000), cached.Total, "cached input at $0.0084/M; output free")
	assert.Empty(t, cached.Unpriced)
}

// TestLiveInventorySkipsDecisionsOnly pins the full-catalog native live
// matrix: without -fireworks-model it must build, and Rune has no case in it.
func TestLiveInventorySkipsDecisionsOnly(t *testing.T) {
	t.Parallel()

	cases := contractInventoryAt(t, fireworks.Catalog(), catalog.Today(), "")
	require.NotEmpty(t, cases)

	for _, c := range cases {
		assert.NotEqual(t, fireworks.ModelRune, c.Model)
	}
}
