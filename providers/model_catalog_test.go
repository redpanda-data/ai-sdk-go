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

package providers_test

import (
	"net/url"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

const catalogVerificationDate = "2026-07-13"

type catalogProvider interface {
	Models() []llm.ModelDiscoveryInfo
	ModelCatalog(string) (llm.ModelCatalogMetadata, bool)
}

func TestModelCatalogMetadataIsComplete(t *testing.T) {
	t.Parallel()

	providers := map[string]catalogProvider{
		"anthropic": &anthropic.Provider{},
		"bedrock":   &bedrock.Provider{},
		"google":    &google.Provider{},
		"openai":    &openai.Provider{},
	}

	validPositioning := map[llm.ModelPositioning]bool{
		llm.ModelPositioningFrontier: true,
		llm.ModelPositioningModern:   true,
		llm.ModelPositioningLegacy:   true,
	}
	validLifecycle := map[llm.ModelLifecycle]bool{
		llm.ModelLifecycleActive:     true,
		llm.ModelLifecycleDeprecated: true,
		llm.ModelLifecycleRetired:    true,
	}
	validReleaseStage := map[llm.ModelReleaseStage]bool{
		llm.ModelReleaseStageStable:  true,
		llm.ModelReleaseStagePreview: true,
	}

	for providerName, provider := range providers {
		t.Run(providerName, func(t *testing.T) {
			t.Parallel()

			models := provider.Models()
			require.NotEmpty(t, models)

			for _, model := range models {
				catalog, ok := provider.ModelCatalog(model.Name)
				require.True(t, ok, "%s: catalog metadata", model.Name)
				require.NotEmpty(t, catalog.FamilyKey, "%s: family key", model.Name)
				require.NotEmpty(t, catalog.RecommendationGroup, "%s: recommendation group", model.Name)
				require.True(t, validPositioning[catalog.Positioning], "%s: positioning %q", model.Name, catalog.Positioning)
				require.True(t, validLifecycle[catalog.Lifecycle], "%s: lifecycle %q", model.Name, catalog.Lifecycle)
				require.True(t, validReleaseStage[catalog.ReleaseStage], "%s: release stage %q", model.Name, catalog.ReleaseStage)
				requireHTTPSURL(t, catalog.OfficialSourceURL, model.Name)
				require.Equal(t, catalogVerificationDate, catalog.VerifiedDate, "%s: verified date", model.Name)
				requireISODate(t, catalog.VerifiedDate, model.Name+": verified date")

				if catalog.EndOfLifeDate != "" {
					requireISODate(t, catalog.EndOfLifeDate, model.Name+": end-of-life date")
				}

				if catalog.Lifecycle == llm.ModelLifecycleRetired {
					require.NotEmpty(t, catalog.EndOfLifeDate, "%s: confirmed retirement needs an end-of-life date", model.Name)
					require.NotEmpty(t, catalog.OfficialSourceURL, "%s: confirmed retirement needs an official source", model.Name)
					require.Equal(t, llm.ModelPositioningLegacy, catalog.Positioning, "%s: retired models are legacy", model.Name)
				}
			}
		})
	}
}

func TestCatalogIncludesLatestConfirmedModels(t *testing.T) {
	t.Parallel()

	requireCatalogContains(t, (&google.Provider{}).Models(), "gemini-3.1-flash-lite")

	bedrockModels := (&bedrock.Provider{}).Models()
	requireCatalogContains(t, bedrockModels, "au.anthropic.claude-opus-4-8")
	requireCatalogContains(t, bedrockModels, "au.anthropic.claude-opus-4-7")
	requireCatalogContains(t, bedrockModels, "jp.anthropic.claude-sonnet-4-6")
	requireCatalogContains(t, bedrockModels, "jp.anthropic.claude-haiku-4-5-20251001-v1:0")
}

func TestDirectProviderCatalogRecommendsOnlyNewestUsefulModelsPerFamily(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		provider catalogProvider
		expected []string
	}{
		"anthropic": {
			provider: &anthropic.Provider{},
			expected: []string{
				anthropic.ModelClaudeFable5,
				anthropic.ModelClaudeHaiku45,
				anthropic.ModelClaudeOpus48,
				anthropic.ModelClaudeSonnet5,
			},
		},
		"google": {
			provider: &google.Provider{},
			expected: []string{
				google.ModelGemini31ProPreview,
				google.ModelGemini35Flash,
			},
		},
		"openai": {
			provider: &openai.Provider{},
			expected: []string{
				openai.ModelGPT5_5Pro,
				openai.ModelGPT5_6Luna,
				openai.ModelGPT5_6Sol,
				openai.ModelGPT5_6Terra,
			},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			actual := recommendedModelNames(tt.provider)
			sort.Strings(tt.expected)
			require.Equal(t, tt.expected, actual)
		})
	}
}

func TestLifecycleRequiresProviderConfirmedStatus(t *testing.T) {
	t.Parallel()

	googleModels := catalogByName(t, &google.Provider{})
	require.Equal(t, llm.ModelLifecycleRetired, googleModels[google.ModelGemini3ProPreview].Lifecycle)
	require.Equal(t, "2026-03-09", googleModels[google.ModelGemini3ProPreview].EndOfLifeDate)

	// These dates are in the future as of the catalog verification date. The
	// models are deprecated, not retired; dates never advance lifecycle state.
	require.Equal(t, llm.ModelLifecycleDeprecated, googleModels[google.ModelGemini25Pro].Lifecycle)
	require.Equal(t, "2026-10-16", googleModels[google.ModelGemini25Pro].EndOfLifeDate)
	require.Equal(t, llm.ModelLifecycleDeprecated, googleModels[google.ModelGemini31FlashLite].Lifecycle)
	require.Equal(t, "2027-05-07", googleModels[google.ModelGemini31FlashLite].EndOfLifeDate)

	anthropicModels := catalogByName(t, &anthropic.Provider{})
	require.Equal(t, llm.ModelLifecycleDeprecated, anthropicModels[anthropic.ModelClaudeOpus41].Lifecycle)
	require.Equal(t, "2026-08-05", anthropicModels[anthropic.ModelClaudeOpus41].EndOfLifeDate)

	openAIModels := catalogByName(t, &openai.Provider{})
	require.Equal(t, llm.ModelLifecycleDeprecated, openAIModels[openai.ModelGPT5].Lifecycle)
	require.Equal(t, "2026-12-11", openAIModels[openai.ModelGPT5].EndOfLifeDate)
	require.Equal(t, llm.ModelLifecycleDeprecated, openAIModels[openai.ModelGPT4O].Lifecycle)
}

func TestBedrockRecommendationGroupsKeepNewestModelPerRoutingGeography(t *testing.T) {
	t.Parallel()

	models := catalogByName(t, &bedrock.Provider{})

	require.Equal(t, llm.ModelPositioningModern, models[bedrock.ModelClaudeOpus48AU].Positioning)
	require.Equal(t, "bedrock-claude-opus-au", models[bedrock.ModelClaudeOpus48AU].RecommendationGroup)
	require.Equal(t, llm.ModelPositioningLegacy, models[bedrock.ModelClaudeOpus46AU].Positioning)
	require.Equal(t, llm.ModelLifecycleActive, models[bedrock.ModelClaudeFable5EU].Lifecycle)
	require.Equal(t, llm.ModelPositioningLegacy, models[bedrock.ModelClaudeFable5EU].Positioning)

	require.Equal(t, llm.ModelPositioningModern, models[bedrock.ModelClaudeSonnet46EU].Positioning)
	require.Equal(t, llm.ModelPositioningLegacy, models[bedrock.ModelClaudeSonnet45EU].Positioning)
	require.Equal(t, llm.ModelPositioningModern, models[bedrock.ModelClaudeSonnet46JP].Positioning)
	require.Equal(t, llm.ModelPositioningLegacy, models[bedrock.ModelClaudeSonnet45JP].Positioning)
}

func TestBedrockMistralLarge3UsesExactModelCard(t *testing.T) {
	t.Parallel()

	catalog, ok := (&bedrock.Provider{}).ModelCatalog(bedrock.ModelMistralLarge3)
	require.True(t, ok)
	require.Equal(
		t,
		"https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-mistral-large-3.html",
		catalog.OfficialSourceURL,
	)
}

func TestCatalogReplacementsAreExactDiscoveredModels(t *testing.T) {
	t.Parallel()

	providers := []catalogProvider{&anthropic.Provider{}, &bedrock.Provider{}, &google.Provider{}, &openai.Provider{}}
	for _, provider := range providers {
		names := make(map[string]struct{})

		for _, model := range provider.Models() {
			names[model.Name] = struct{}{}
		}

		for _, model := range provider.Models() {
			catalog, ok := provider.ModelCatalog(model.Name)
			require.True(t, ok)

			if catalog.Replacement != "" {
				require.Contains(t, names, catalog.Replacement, "%s replacement", model.Name)
			}
		}
	}
}

func TestOpenAIGPT52ProRecommendsLatestProModel(t *testing.T) {
	t.Parallel()

	catalog, ok := (&openai.Provider{}).ModelCatalog(openai.ModelGPT5_2Pro)
	require.True(t, ok)
	require.Equal(t, openai.ModelGPT5_5Pro, catalog.Replacement)
}

func TestOpenAIGPT55ProHasExactCapabilitiesAndPricing(t *testing.T) {
	t.Parallel()

	models := catalogDiscoveryByName((&openai.Provider{}).Models())
	pro, ok := models[openai.ModelGPT5_5Pro]
	require.True(t, ok)
	require.False(t, pro.Capabilities.Streaming)
	require.Equal(t, 1_050_000, pro.Constraints.MaxInputTokens)
	require.Equal(t, 128_000, pro.Constraints.MaxOutputTokens)

	price, ok := openai.ModelPricing()[openai.ModelGPT5_5Pro]
	require.True(t, ok)
	require.Equal(t, int64(3_000_000_000), price.Default.Base.InputPerMillion)
	require.Equal(t, int64(18_000_000_000), price.Default.Base.OutputPerMillion)
	require.Zero(t, price.Default.Base.CachedInputPerMillion)
}

func recommendedModelNames(provider catalogProvider) []string {
	models := provider.Models()
	names := make([]string, 0, len(models))

	for _, model := range models {
		catalog, ok := provider.ModelCatalog(model.Name)
		if ok && catalog.Positioning != llm.ModelPositioningLegacy && catalog.Lifecycle == llm.ModelLifecycleActive {
			names = append(names, model.Name)
		}
	}

	sort.Strings(names)

	return names
}

func catalogByName(t *testing.T, provider catalogProvider) map[string]llm.ModelCatalogMetadata {
	t.Helper()

	result := make(map[string]llm.ModelCatalogMetadata)

	for _, model := range provider.Models() {
		catalog, ok := provider.ModelCatalog(model.Name)
		require.True(t, ok, model.Name)
		result[model.Name] = catalog
	}

	return result
}

func catalogDiscoveryByName(models []llm.ModelDiscoveryInfo) map[string]llm.ModelDiscoveryInfo {
	result := make(map[string]llm.ModelDiscoveryInfo, len(models))

	for _, model := range models {
		result[model.Name] = model
	}

	return result
}

func requireCatalogContains(t *testing.T, models []llm.ModelDiscoveryInfo, modelName string) {
	t.Helper()

	for _, model := range models {
		if model.Name == modelName {
			return
		}
	}

	t.Fatalf("catalog does not contain %q", modelName)
}

func requireISODate(t *testing.T, value, field string) {
	t.Helper()

	_, err := time.Parse(time.DateOnly, value)
	require.NoError(t, err, field)
}

func requireHTTPSURL(t *testing.T, value, modelName string) {
	t.Helper()

	parsed, err := url.ParseRequestURI(value)
	require.NoError(t, err, "%s: official source URL", modelName)
	require.Equal(t, "https", parsed.Scheme, "%s: official source URL", modelName)
	require.NotEmpty(t, parsed.Host, "%s: official source URL", modelName)
}
