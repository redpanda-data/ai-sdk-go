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

package google_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
)

func TestGeminiDecision1IsDecisionsOnly(t *testing.T) {
	t.Parallel()

	o, ok := google.Catalog().Lookup(google.ModelGeminiDecision1)
	require.True(t, ok)
	assert.True(t, o.DecisionsOnly())
	assert.Equal(t, catalog.PublisherGoogle, o.Facts().Publisher)
	assert.Equal(t, 262144, o.Constraints.MaxInputTokens)
	assert.Zero(t, o.Constraints.MaxOutputTokens)
	assert.Empty(t, o.Modalities.Output)
	assert.ElementsMatch(t, []catalog.Modality{
		catalog.ModalityText, catalog.ModalityImage, catalog.ModalityAudio, catalog.ModalityVideo, catalog.ModalityDocument,
	}, o.Modalities.Input)
	assert.Equal(t, llm.ModelCapabilities{Vision: true, Audio: true}, o.Capabilities, "input modalities only; nothing generative")
	assert.Equal(t, catalog.ReasoningSupport{}, o.Reasoning)
}

func TestNewModelRefusesDecisionsOnly(t *testing.T) {
	t.Parallel()

	provider, err := google.NewProvider(context.Background(), "test-api-key")
	require.NoError(t, err)

	_, err = provider.NewModel(google.ModelGeminiDecision1)
	require.ErrorIs(t, err, catalog.ErrDecisionsOnly)
	require.ErrorContains(t, err, google.ModelGeminiDecision1)

	offering, ok := google.Catalog().Lookup(google.ModelGeminiDecision1)
	require.True(t, ok)

	_, err = provider.NewModelFromOffering(google.ModelGeminiDecision1, offering)
	require.ErrorIs(t, err, catalog.ErrDecisionsOnly)

	for _, o := range google.Catalog().All() {
		if o.DecisionsOnly() {
			continue
		}

		_, err := provider.NewModel(o.ID)
		require.NoError(t, err, o.ID)
	}
}

// TestGeminiDecision1DocumentedUsagePrice prices the usage block of the
// create-decision reference's example response
// (https://aistudio.google.com/api-reference/decisions/create-decision, read
// 2026-11-22): total_input_tokens 152, total_output_tokens 0.
func TestGeminiDecision1DocumentedUsagePrice(t *testing.T) {
	t.Parallel()

	pc, err := pricing.NewCatalog(pricing.WithSource(google.Catalog()))
	require.NoError(t, err)

	cost, err := pc.Calculate(google.ProviderName, google.ModelGeminiDecision1, &llm.TokenUsage{InputTokens: 152}, pricing.CalcRequest{})
	require.NoError(t, err)
	assert.Equal(t, int64(152*10), cost.Total, "152 tokens at $0.10/M, in microcents")
	assert.Empty(t, cost.Unpriced)
}
