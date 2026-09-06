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

package meta

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

func TestMuseSpark13Catalog(t *testing.T) {
	t.Parallel()

	p, err := NewProvider("test-key")
	require.NoError(t, err)
	assert.Equal(t, "meta", p.Name())
	assert.Equal(t, "https://api.meta.ai/v1", p.transport.BaseURL)
	require.Len(t, p.Catalog().All(), 1)
	o, ok := p.Catalog().Lookup("muse-spark-1.3")
	require.True(t, ok)
	assert.Equal(t, catalog.ModelID("meta/muse-spark-1.3"), o.Model)
	assert.Equal(t, "Muse Spark 1.3", o.DisplayName)
	assert.Equal(t, 1_048_576, o.Constraints.MaxInputTokens)
	assert.Equal(t, 32_768, o.Constraints.MaxOutputTokens)
	assert.Equal(t, "sdk_conservative_limit", o.Attributes["output_token_limit_source"])
	assert.True(t, o.Capabilities.Tools)
	assert.True(t, o.Capabilities.Streaming)
	assert.True(t, o.Capabilities.StructuredOutput)
	assert.Contains(t, o.Modalities.Input, catalog.ModalityImage)
	assert.Contains(t, o.Modalities.Input, catalog.ModalityVideo)
	assert.Contains(t, o.Modalities.Input, catalog.ModalityDocument)
	assert.Equal(t, []catalog.Modality{catalog.ModalityText}, o.Modalities.Output)
	assert.Equal(t, int64(125_000_000), o.Pricing.Default.Base.InputPerMillion)
	assert.Equal(t, int64(425_000_000), o.Pricing.Default.Base.OutputPerMillion)
	assert.Equal(t, int64(15_000_000), o.Pricing.Default.Base.CachedInputPerMillion)
	assert.Empty(t, o.Pricing.Default.Brackets)
}

func TestMuseSpark13Options(t *testing.T) {
	t.Parallel()

	p, err := NewProvider("test-key")
	require.NoError(t, err)

	for _, effort := range []llm.ReasoningEffort{openai.ReasoningEffortMinimal, openai.ReasoningEffortLow, openai.ReasoningEffortMedium, openai.ReasoningEffortHigh, openai.ReasoningEffortXHigh, openai.ReasoningEffortMax} {
		m, err := p.NewModel("muse-spark-1.3", openai.WithReasoningEffort(effort), openai.WithMaxTokens(1024))
		require.NoError(t, err)
		assert.Equal(t, "meta", m.Provider())
		assert.Equal(t, "muse-spark-1.3", m.Name())
		lister, ok := m.(llm.ReasoningEffortLister)
		require.True(t, ok)
		assert.Contains(t, lister.SupportedReasoningEfforts(), effort)
		efforts := lister.SupportedReasoningEfforts()
		efforts[0] = openai.ReasoningEffortNone
		assert.NotContains(t, lister.SupportedReasoningEfforts(), openai.ReasoningEffortNone)
	}

	for _, name := range []string{"", "muse-spark-1.3-contributor", "gpt-5.6-sol", "muse-spark-1.3-unknown"} {
		_, err := p.NewModel(name)
		require.Error(t, err)
	}

	for _, option := range []openai.Option{openai.WithReasoningEffort(openai.ReasoningEffortNone), openai.WithMaxTokens(15), openai.WithMaxTokens(32_769), openai.WithMaxTokens(1_048_577), openai.WithTopP(0.9), openai.WithFrequencyPenalty(0.5), openai.WithPresencePenalty(0.5), openai.WithSeed(1)} {
		_, err := p.NewModel("muse-spark-1.3", option)
		require.Error(t, err)
	}

	_, err = NewProvider("")
	require.Error(t, err)
}
