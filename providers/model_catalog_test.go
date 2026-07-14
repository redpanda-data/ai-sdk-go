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

var metadataVerificationDate = time.Date(2026, time.July, 14, 0, 0, 0, 0, time.UTC)

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
				require.NotEmpty(t, catalog.UpgradeGroup, "%s: upgrade group", model.Name)
				requireCatalogDate(t, catalog.ReleaseDate, model.Name+": release date")
				require.NotEqual(t, llm.ModelLifecycleUnknown, catalog.Lifecycle, "%s: lifecycle", model.Name)
				requireHTTPSURL(t, catalog.OfficialSourceURL, model.Name)
				require.Equal(t, metadataVerificationDate, catalog.MetadataVerifiedDate, "%s: verified date", model.Name)
				requireCatalogDate(t, catalog.MetadataVerifiedDate, model.Name+": verified date")

				if !catalog.EndOfLifeDate.IsZero() {
					requireCatalogDate(t, catalog.EndOfLifeDate, model.Name+": end-of-life date")
				}

				if catalog.Lifecycle == llm.ModelLifecycleRetired {
					require.False(t, catalog.EndOfLifeDate.IsZero(), "%s: confirmed retirement needs an end-of-life date", model.Name)
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
	require.Equal(t, "2026-05-28", catalog.ReleaseDate.Format(time.DateOnly))
	require.Equal(t, llm.ModelLifecycleActive, catalog.Lifecycle)

	retired, ok := (&google.Provider{}).ModelCatalog("gemini-3.1-flash-lite-preview")
	require.True(t, ok)
	require.Equal(t, "2026-03-03", retired.ReleaseDate.Format(time.DateOnly))
	require.Equal(t, llm.ModelLifecycleRetired, retired.Lifecycle)
}

func TestCatalogOverridesEnumerateEveryExactNonDiscoveryCatalogID(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		provider catalogOverridesProvider
		expected map[string]llm.ModelLifecycle
		excluded []string
	}{
		"google": {
			provider: &google.Provider{},
			expected: map[string]llm.ModelLifecycle{
				"gemini-3.1-flash-lite-preview":         llm.ModelLifecycleRetired,
				"gemini-2.0-flash":                      llm.ModelLifecycleRetired,
				"gemini-2.0-flash-001":                  llm.ModelLifecycleRetired,
				"gemini-2.0-flash-lite":                 llm.ModelLifecycleRetired,
				"gemini-2.0-flash-lite-001":             llm.ModelLifecycleRetired,
				"gemini-2.0-flash-lite-preview":         llm.ModelLifecycleRetired,
				"gemini-2.0-flash-lite-preview-02-05":   llm.ModelLifecycleRetired,
				"gemini-2.5-flash-lite-preview-09-2025": llm.ModelLifecycleRetired,
				"gemini-2.5-flash-preview-05-20":        llm.ModelLifecycleRetired,
				"gemini-2.5-flash-preview-09-25":        llm.ModelLifecycleRetired,
				"gemini-2.5-pro-preview-03-25":          llm.ModelLifecycleRetired,
				"gemini-2.5-pro-preview-05-06":          llm.ModelLifecycleRetired,
				"gemini-2.5-pro-preview-06-05":          llm.ModelLifecycleRetired,
			},
			excluded: []string{google.ModelGeminiFlashLatest, google.ModelGemini35Flash},
		},
		"openai": {
			provider: &openai.Provider{},
			expected: map[string]llm.ModelLifecycle{
				"gpt-5-2025-08-07":       llm.ModelLifecycleDeprecated,
				"gpt-5-mini-2025-08-07":  llm.ModelLifecycleDeprecated,
				"gpt-5-nano-2025-08-07":  llm.ModelLifecycleDeprecated,
				"o3-2025-04-16":          llm.ModelLifecycleDeprecated,
				"o3-pro-2025-06-10":      llm.ModelLifecycleDeprecated,
				"gpt-3.5-turbo-0125":     llm.ModelLifecycleDeprecated,
				"gpt-4-turbo-2024-04-09": llm.ModelLifecycleDeprecated,
				"gpt-4o-2024-05-13":      llm.ModelLifecycleDeprecated,
				"o1-pro-2025-03-19":      llm.ModelLifecycleDeprecated,
				"o4-mini-2025-04-16":     llm.ModelLifecycleDeprecated,
				"gpt-4-turbo-preview":    llm.ModelLifecycleRetired,
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

			for model, lifecycle := range tt.expected {
				metadata, ok := overrides[model]
				require.True(t, ok, model)
				require.Equal(t, lifecycle, metadata.Lifecycle, model)
				require.False(t, metadata.EndOfLifeDate.IsZero(), model)
				requireCatalogDate(t, metadata.ReleaseDate, model+": release date")
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

func TestUpgradeGroupsIdentifyLatestActiveModels(t *testing.T) {
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

			actual := latestActiveModelNames(tt.provider)
			sort.Strings(tt.expected)
			require.Equal(t, tt.expected, actual)
		})
	}
}

func TestLifecycleStatusRequiresProviderConfirmation(t *testing.T) {
	t.Parallel()

	googleModels := catalogByName(t, &google.Provider{})
	require.Equal(t, llm.ModelLifecycleRetired, googleModels[google.ModelGemini3ProPreview].Lifecycle)
	require.Equal(t, "2026-03-09", googleModels[google.ModelGemini3ProPreview].EndOfLifeDate.Format(time.DateOnly))

	// Announced end-of-life dates are factual schedule data. They do not imply
	// retirement, but a provider deprecation notice is recorded independently.
	require.Equal(t, llm.ModelLifecycleDeprecated, googleModels[google.ModelGemini25Pro].Lifecycle)
	require.Equal(t, "2026-10-16", googleModels[google.ModelGemini25Pro].EndOfLifeDate.Format(time.DateOnly))
	require.Equal(t, llm.ModelLifecycleDeprecated, googleModels[google.ModelGemini31FlashLite].Lifecycle)
	require.Equal(t, "2027-05-07", googleModels[google.ModelGemini31FlashLite].EndOfLifeDate.Format(time.DateOnly))
	require.Equal(t, llm.ModelLifecycleActive, googleModels[google.ModelGemini31ProPreview].Lifecycle)
	require.True(t, googleModels[google.ModelGemini31ProPreview].EndOfLifeDate.IsZero())

	anthropicModels := catalogByName(t, &anthropic.Provider{})
	require.Equal(t, llm.ModelLifecycleDeprecated, anthropicModels[anthropic.ModelClaudeOpus41].Lifecycle)
	require.Equal(t, "2026-08-05", anthropicModels[anthropic.ModelClaudeOpus41].EndOfLifeDate.Format(time.DateOnly))
	require.Equal(t, llm.ModelLifecycleActive, anthropicModels[anthropic.ModelClaudeOpus45].Lifecycle)

	openAIModels := catalogByName(t, &openai.Provider{})
	require.Equal(t, llm.ModelLifecycleActive, openAIModels[openai.ModelGPT5].Lifecycle)
	require.True(t, openAIModels[openai.ModelGPT5].EndOfLifeDate.IsZero())
	require.Equal(t, llm.ModelLifecycleDeprecated, openAIModels[openai.ModelGPT4O].Lifecycle)
	require.True(t, openAIModels[openai.ModelGPT4O].EndOfLifeDate.IsZero())

	bedrockModels := catalogByName(t, &bedrock.Provider{})
	require.Equal(t, llm.ModelLifecycleActive, bedrockModels[bedrock.ModelClaudeFable5Global].Lifecycle)
}

func TestBedrockUpgradeGroupsKeepNewestModelPerRoutingGeography(t *testing.T) {
	t.Parallel()

	models := catalogByName(t, &bedrock.Provider{})

	require.Equal(t, "bedrock-claude-opus-au", models[bedrock.ModelClaudeOpus48AU].UpgradeGroup)
	require.True(t, models[bedrock.ModelClaudeOpus48AU].ReleaseDate.After(models[bedrock.ModelClaudeOpus46AU].ReleaseDate))

	fableEU, ok := (&bedrock.Provider{}).ModelCatalog(bedrock.ModelClaudeFable5EU)
	require.True(t, ok)
	require.Equal(t, llm.ModelLifecycleActive, fableEU.Lifecycle)

	require.True(t, models[bedrock.ModelClaudeSonnet46EU].ReleaseDate.After(models[bedrock.ModelClaudeSonnet45EU].ReleaseDate))
	require.True(t, models[bedrock.ModelClaudeSonnet46JP].ReleaseDate.After(models[bedrock.ModelClaudeSonnet45JP].ReleaseDate))
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

func TestCatalogProviderReplacementsAreExactDiscoveredModels(t *testing.T) {
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

			if catalog.ProviderReplacement != "" {
				require.Contains(t, names, catalog.ProviderReplacement, "%s replacement", model.Name)
			}
		}
	}
}

func TestOpenAIGPT52ProRecordsProviderMigrationTarget(t *testing.T) {
	t.Parallel()

	catalog, ok := (&openai.Provider{}).ModelCatalog(openai.ModelGPT5_2Pro)
	require.True(t, ok)
	require.Equal(t, openai.ModelGPT5_5Pro, catalog.ProviderReplacement)
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

func latestActiveModelNames(provider catalogProvider) []string {
	models := provider.Models()
	latestReleaseByGroup := make(map[string]time.Time)

	for _, model := range models {
		catalog, ok := provider.ModelCatalog(model.Name)
		if ok && catalog.Lifecycle == llm.ModelLifecycleActive && catalog.ReleaseDate.After(latestReleaseByGroup[catalog.UpgradeGroup]) {
			latestReleaseByGroup[catalog.UpgradeGroup] = catalog.ReleaseDate
		}
	}

	names := make([]string, 0, len(latestReleaseByGroup))

	for _, model := range models {
		catalog, ok := provider.ModelCatalog(model.Name)
		if ok && catalog.Lifecycle == llm.ModelLifecycleActive && catalog.ReleaseDate.Equal(latestReleaseByGroup[catalog.UpgradeGroup]) {
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

func requireCatalogDate(t *testing.T, value time.Time, field string) {
	t.Helper()

	require.False(t, value.IsZero(), field)
	require.Equal(t, time.UTC, value.Location(), field)
	require.Equal(t, 0, value.Hour(), field)
	require.Equal(t, 0, value.Minute(), field)
	require.Equal(t, 0, value.Second(), field)
	require.Equal(t, 0, value.Nanosecond(), field)
}

func requireHTTPSURL(t *testing.T, value, modelName string) {
	t.Helper()

	parsed, err := url.ParseRequestURI(value)
	require.NoError(t, err, "%s: official source URL", modelName)
	require.Equal(t, "https", parsed.Scheme, "%s: official source URL", modelName)
	require.NotEmpty(t, parsed.Host, "%s: official source URL", modelName)
}
