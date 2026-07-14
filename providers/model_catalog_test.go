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

const catalogVerificationDate = "2026-07-14"

type catalogProvider interface {
	Models() []llm.ModelDiscoveryInfo
	ModelCatalog(string) (llm.ModelCatalogMetadata, bool)
}

type catalogOverridesProvider interface {
	catalogProvider
	llm.ModelCatalogOverridesProvider
}

func TestModelCatalogMetadataIsComplete(t *testing.T) {
	t.Parallel()

	providers := map[string]catalogProvider{
		"anthropic": &anthropic.Provider{},
		"bedrock":   &bedrock.Provider{},
		"google":    &google.Provider{},
		"openai":    &openai.Provider{},
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
				requireISODate(t, catalog.ReleaseDate, model.Name+": release date")
				requireHTTPSURL(t, catalog.OfficialSourceURL, model.Name)
				require.Equal(t, catalogVerificationDate, catalog.VerifiedDate, "%s: verified date", model.Name)
				requireISODate(t, catalog.VerifiedDate, model.Name+": verified date")

				if catalog.EndOfLifeDate != "" {
					requireISODate(t, catalog.EndOfLifeDate, model.Name+": end-of-life date")
				}

				if catalog.Retired {
					require.True(t, catalog.Deprecated, "%s: retired models must also be deprecated", model.Name)
					require.NotEmpty(t, catalog.EndOfLifeDate, "%s: confirmed retirement needs an end-of-life date", model.Name)
					require.NotEmpty(t, catalog.OfficialSourceURL, "%s: confirmed retirement needs an official source", model.Name)
				}
			}
		})
	}
}

func TestCatalogStoresReleaseFactsInsteadOfMutableRanking(t *testing.T) {
	t.Parallel()

	catalog, ok := (&anthropic.Provider{}).ModelCatalog(anthropic.ModelClaudeOpus48)
	require.True(t, ok)
	require.Equal(t, "2026-05-28", catalog.ReleaseDate)
	require.False(t, catalog.Retired)

	retired, ok := (&google.Provider{}).ModelCatalog("gemini-3.1-flash-lite-preview")
	require.True(t, ok)
	require.Equal(t, "2026-03-03", retired.ReleaseDate)
	require.True(t, retired.Retired)
}

func TestCatalogOverridesEnumerateEveryExactNonDiscoveryCatalogID(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		provider catalogOverridesProvider
		expected map[string]bool
		excluded []string
	}{
		"google": {
			provider: &google.Provider{},
			expected: map[string]bool{
				"gemini-3.1-flash-lite-preview": true,
				"gemini-2.0-flash":              true,
				"gemini-2.5-pro-preview-03-25":  true,
				"gemini-2.5-pro-preview-05-06":  true,
				"gemini-2.5-pro-preview-06-05":  true,
			},
			excluded: []string{google.ModelGeminiFlashLatest, google.ModelGemini35Flash},
		},
		"openai": {
			provider: &openai.Provider{},
			expected: map[string]bool{
				"gpt-5-2025-08-07":      false,
				"gpt-5-mini-2025-08-07": false,
				"gpt-5-nano-2025-08-07": false,
				"o3-2025-04-16":         false,
				"o3-pro-2025-06-10":     false,
				"gpt-4-turbo-preview":   true,
			},
			excluded: []string{openai.ModelGPT5, openai.ModelO3, openai.ModelGPT5_6},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			overrides := tt.provider.ModelCatalogOverrides()
			require.Len(t, overrides, len(tt.expected))

			discovered := make(map[string]struct{})
			for _, model := range tt.provider.Models() {
				discovered[model.Name] = struct{}{}
			}

			for model, retired := range tt.expected {
				metadata, ok := overrides[model]
				require.True(t, ok, model)
				require.Equal(t, retired, metadata.Retired, model)
				require.True(t, metadata.Deprecated, model)
				require.NotEmpty(t, metadata.EndOfLifeDate, model)
				requireISODate(t, metadata.ReleaseDate, model+": release date")
				require.NotContains(t, discovered, model)

				resolved, ok := tt.provider.ModelCatalog(model)
				require.True(t, ok, model)
				require.Equal(t, metadata, resolved, model)
			}

			for _, model := range tt.excluded {
				require.NotContains(t, overrides, model)
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
				google.ModelGemini31FlashLite,
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

func TestLifecycleStatusRequiresProviderConfirmation(t *testing.T) {
	t.Parallel()

	googleModels := catalogByName(t, &google.Provider{})
	require.True(t, googleModels[google.ModelGemini3ProPreview].Deprecated)
	require.True(t, googleModels[google.ModelGemini3ProPreview].Retired)
	require.Equal(t, "2026-03-09", googleModels[google.ModelGemini3ProPreview].EndOfLifeDate)

	// Announced end-of-life dates are factual schedule data. They do not imply
	// retirement, but a provider deprecation notice is recorded independently.
	require.True(t, googleModels[google.ModelGemini25Pro].Deprecated)
	require.False(t, googleModels[google.ModelGemini25Pro].Retired)
	require.Equal(t, "2026-10-16", googleModels[google.ModelGemini25Pro].EndOfLifeDate)
	require.True(t, googleModels[google.ModelGemini31FlashLite].Deprecated)
	require.False(t, googleModels[google.ModelGemini31FlashLite].Retired)
	require.Equal(t, "2027-05-07", googleModels[google.ModelGemini31FlashLite].EndOfLifeDate)
	require.False(t, googleModels[google.ModelGemini31ProPreview].Deprecated)
	require.Empty(t, googleModels[google.ModelGemini31ProPreview].EndOfLifeDate)

	anthropicModels := catalogByName(t, &anthropic.Provider{})
	require.True(t, anthropicModels[anthropic.ModelClaudeOpus41].Deprecated)
	require.False(t, anthropicModels[anthropic.ModelClaudeOpus41].Retired)
	require.Equal(t, "2026-08-05", anthropicModels[anthropic.ModelClaudeOpus41].EndOfLifeDate)
	require.False(t, anthropicModels[anthropic.ModelClaudeOpus45].Deprecated)

	openAIModels := catalogByName(t, &openai.Provider{})
	require.False(t, openAIModels[openai.ModelGPT5].Deprecated)
	require.False(t, openAIModels[openai.ModelGPT5].Retired)
	require.Empty(t, openAIModels[openai.ModelGPT5].EndOfLifeDate)
	require.True(t, openAIModels[openai.ModelGPT4O].Deprecated)
	require.False(t, openAIModels[openai.ModelGPT4O].Retired)
	require.Empty(t, openAIModels[openai.ModelGPT4O].EndOfLifeDate)

	bedrockModels := catalogByName(t, &bedrock.Provider{})
	require.False(t, bedrockModels[bedrock.ModelClaudeFable5Global].Deprecated)
	require.False(t, bedrockModels[bedrock.ModelClaudeFable5Global].Retired)
}

func TestBedrockRecommendationGroupsKeepNewestModelPerRoutingGeography(t *testing.T) {
	t.Parallel()

	models := catalogByName(t, &bedrock.Provider{})

	require.Equal(t, "bedrock-claude-opus-au", models[bedrock.ModelClaudeOpus48AU].RecommendationGroup)
	require.Greater(t, models[bedrock.ModelClaudeOpus48AU].ReleaseDate, models[bedrock.ModelClaudeOpus46AU].ReleaseDate)
	require.False(t, models[bedrock.ModelClaudeFable5EU].Retired)

	require.Greater(t, models[bedrock.ModelClaudeSonnet46EU].ReleaseDate, models[bedrock.ModelClaudeSonnet45EU].ReleaseDate)
	require.Greater(t, models[bedrock.ModelClaudeSonnet46JP].ReleaseDate, models[bedrock.ModelClaudeSonnet45JP].ReleaseDate)
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
	latestReleaseByGroup := make(map[string]string)

	for _, model := range models {
		catalog, ok := provider.ModelCatalog(model.Name)
		if ok && !catalog.Retired && catalog.ReleaseDate > latestReleaseByGroup[catalog.RecommendationGroup] {
			latestReleaseByGroup[catalog.RecommendationGroup] = catalog.ReleaseDate
		}
	}

	names := make([]string, 0, len(latestReleaseByGroup))

	for _, model := range models {
		catalog, ok := provider.ModelCatalog(model.Name)
		if ok && !catalog.Retired && catalog.ReleaseDate == latestReleaseByGroup[catalog.RecommendationGroup] {
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
