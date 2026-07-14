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

package openai

import (
	"maps"
	"strings"
	"time"

	"github.com/redpanda-data/ai-sdk-go/internal/catalogdate"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	openAIModelsSource         = "https://developers.openai.com/api/docs/models"
	openAIDeprecationsSource   = "https://developers.openai.com/api/docs/deprecations"
	openAIMetadataVerifiedDate = "2026-07-14"
)

var openAIExactModelCatalogOverrides = map[string]llm.ModelCatalogMetadata{
	"gpt-5-2025-08-07":       openAICatalog("gpt", "openai-gpt-flagship", "2025-08-07", "2026-12-11", llm.ModelLifecycleDeprecated, ModelGPT5_5, openAIDeprecationsSource),
	"gpt-5-mini-2025-08-07":  openAICatalog("gpt", "openai-gpt-balanced", "2025-08-07", "2026-12-11", llm.ModelLifecycleDeprecated, ModelGPT5_4Mini, openAIDeprecationsSource),
	"gpt-5-nano-2025-08-07":  openAICatalog("gpt", "openai-gpt-efficient", "2025-08-07", "2026-12-11", llm.ModelLifecycleDeprecated, ModelGPT5_4Nano, openAIDeprecationsSource),
	"o3-2025-04-16":          openAICatalog("o-series", "openai-gpt-flagship", "2025-04-16", "2026-12-11", llm.ModelLifecycleDeprecated, ModelGPT5_5, openAIDeprecationsSource),
	"o3-pro-2025-06-10":      openAICatalog("o-series", "openai-gpt-pro", "2025-06-10", "2026-12-11", llm.ModelLifecycleDeprecated, ModelGPT5_5Pro, openAIDeprecationsSource),
	"gpt-3.5-turbo-0125":     openAICatalog("gpt", "openai-gpt-balanced", "2024-01-25", "2026-10-23", llm.ModelLifecycleDeprecated, ModelGPT5_4Mini, openAIDeprecationsSource),
	"gpt-4-turbo-2024-04-09": openAICatalog("gpt", "openai-gpt-flagship", "2024-04-09", "2026-10-23", llm.ModelLifecycleDeprecated, ModelGPT5_5, openAIDeprecationsSource),
	"gpt-4o-2024-05-13":      openAICatalog("gpt", "openai-gpt-flagship", "2024-05-13", "2026-10-23", llm.ModelLifecycleDeprecated, ModelGPT5_5, openAIDeprecationsSource),
	"o1-pro-2025-03-19":      openAICatalog("o-series", "openai-gpt-pro", "2025-03-19", "2026-10-23", llm.ModelLifecycleDeprecated, ModelGPT5_5Pro, openAIDeprecationsSource),
	"o4-mini-2025-04-16":     openAICatalog("o-series", "openai-gpt-balanced", "2025-04-16", "2026-10-23", llm.ModelLifecycleDeprecated, ModelGPT5_4Mini, openAIDeprecationsSource),
	"gpt-4-turbo-preview":    openAICatalog("gpt", "openai-gpt-flagship", "2023-11-06", "2026-03-26", llm.ModelLifecycleRetired, ModelGPT5_5, openAIDeprecationsSource),
}

// ModelCatalog returns factual catalog metadata for a canonical,
// aliased, or dated OpenAI model identifier.
func (*Provider) ModelCatalog(model string) (llm.ModelCatalogMetadata, bool) {
	if catalog, ok := openAIExactSnapshotCatalog(model); ok {
		return catalog, true
	}

	if alias, ok := modelAliases[model]; ok {
		return openAIModelCatalog(alias)
	}

	if _, ok := supportedModels[model]; ok {
		return openAIModelCatalog(model)
	}

	if family, ok := openAIDatedSnapshotFamily(model); ok {
		return openAIModelCatalog(family)
	}

	return llm.ModelCatalogMetadata{}, false
}

// ModelCatalogOverrides returns exact EOL or retired model IDs that
// remain outside Models() so they are not offered for new selections.
func (*Provider) ModelCatalogOverrides() map[string]llm.ModelCatalogMetadata {
	return maps.Clone(openAIExactModelCatalogOverrides)
}

func openAIExactSnapshotCatalog(model string) (llm.ModelCatalogMetadata, bool) {
	catalog, ok := openAIExactModelCatalogOverrides[model]

	return catalog, ok
}

func openAIDatedSnapshotFamily(model string) (string, bool) {
	best := ""

	for family := range supportedModels {
		prefix := family + "-"
		if !strings.HasPrefix(model, prefix) || len(family) <= len(best) {
			continue
		}

		date := strings.TrimPrefix(model, prefix)
		if len(date) != len(time.DateOnly) {
			continue
		}

		if _, err := time.Parse(time.DateOnly, date); err == nil {
			best = family
		}
	}

	return best, best != ""
}

func openAIModelCatalog(name string) (llm.ModelCatalogMetadata, bool) {
	switch name {
	case ModelGPT5_6Sol:
		return openAICatalog("gpt", "openai-gpt-flagship", "2026-07-09", "", llm.ModelLifecycleActive, "", openAIModelsSource), true
	case ModelGPT5_6Terra:
		return openAICatalog("gpt", "openai-gpt-balanced", "2026-07-09", "", llm.ModelLifecycleActive, "", openAIModelsSource), true
	case ModelGPT5_6Luna:
		return openAICatalog("gpt", "openai-gpt-efficient", "2026-07-09", "", llm.ModelLifecycleActive, "", openAIModelsSource), true

	case ModelGPT5_5:
		return openAICatalog("gpt", "openai-gpt-flagship", "2026-04-23", "", llm.ModelLifecycleActive, ModelGPT5_6Sol, openAIModelsSource), true
	case ModelGPT5_4:
		return openAICatalog("gpt", "openai-gpt-flagship", "2026-03-05", "", llm.ModelLifecycleActive, ModelGPT5_6Sol, openAIModelsSource), true
	case ModelGPT5_2:
		return openAICatalog("gpt", "openai-gpt-flagship", "2025-12-11", "", llm.ModelLifecycleActive, ModelGPT5_6Sol, openAIModelsSource), true
	case ModelGPT5_1:
		return openAICatalog("gpt", "openai-gpt-flagship", "2025-11-13", "", llm.ModelLifecycleActive, ModelGPT5_6Sol, openAIModelsSource), true
	case ModelGPT41:
		return openAICatalog("gpt", "openai-gpt-flagship", "2025-04-14", "", llm.ModelLifecycleActive, ModelGPT5_6Sol, openAIModelsSource), true
	case ModelGPT5_5Pro:
		return openAICatalog("gpt", "openai-gpt-pro", "2026-04-23", "", llm.ModelLifecycleActive, "", openAIModelsSource), true
	case ModelGPT4O:
		return openAICatalog("gpt", "openai-gpt-flagship", "2024-05-13", "", llm.ModelLifecycleDeprecated, ModelGPT5_6Sol, openAIModelsSource), true
	case ModelGPT5_4Mini:
		return openAICatalog("gpt", "openai-gpt-balanced", "2026-03-17", "", llm.ModelLifecycleActive, ModelGPT5_6Terra, openAIModelsSource), true
	case ModelGPT41Mini:
		return openAICatalog("gpt", "openai-gpt-balanced", "2025-04-14", "", llm.ModelLifecycleActive, ModelGPT5_6Terra, openAIModelsSource), true
	case ModelGPT4OMini:
		return openAICatalog("gpt", "openai-gpt-balanced", "2024-07-18", "", llm.ModelLifecycleActive, ModelGPT5_6Terra, openAIModelsSource), true
	case ModelGPT5_4Nano:
		return openAICatalog("gpt", "openai-gpt-efficient", "2026-03-17", "", llm.ModelLifecycleActive, ModelGPT5_6Luna, openAIModelsSource), true
	case ModelGPT5_2Pro:
		return openAICatalog("gpt", "openai-gpt-pro", "2025-12-11", "", llm.ModelLifecycleActive, ModelGPT5_5Pro, openAIModelsSource), true
	case ModelGPT5_3Codex:
		return openAICatalog("gpt", "openai-gpt-flagship", "2026-02-05", "", llm.ModelLifecycleActive, ModelGPT5_6Sol, openAIModelsSource), true

	case ModelGPT5:
		return openAICatalog("gpt", "openai-gpt-flagship", "2025-08-07", "", llm.ModelLifecycleActive, ModelGPT5_5, openAIModelsSource), true
	case ModelGPT5Mini:
		return openAICatalog("gpt", "openai-gpt-balanced", "2025-08-07", "", llm.ModelLifecycleActive, ModelGPT5_4Mini, openAIModelsSource), true
	case ModelGPT5Nano:
		return openAICatalog("gpt", "openai-gpt-efficient", "2025-08-07", "", llm.ModelLifecycleActive, ModelGPT5_4Nano, openAIModelsSource), true
	case ModelGPT5_2Instant:
		return openAICatalog("gpt", "openai-gpt-flagship", "2025-12-11", "2026-08-10", llm.ModelLifecycleDeprecated, ModelGPT5_5, openAIDeprecationsSource), true
	case ModelGPT5_3ChatLatest:
		return openAICatalog("gpt", "openai-gpt-flagship", "2026-03-03", "2026-08-10", llm.ModelLifecycleDeprecated, ModelGPT5_5, openAIDeprecationsSource), true
	case ModelGPT4Turbo:
		return openAICatalog("gpt", "openai-gpt-flagship", "2024-04-09", "2026-10-23", llm.ModelLifecycleDeprecated, ModelGPT5_5, openAIDeprecationsSource), true
	case ModelGPT35Turbo:
		return openAICatalog("gpt", "openai-gpt-balanced", "2024-01-25", "2026-10-23", llm.ModelLifecycleDeprecated, ModelGPT5_4Mini, openAIDeprecationsSource), true

	case ModelO3:
		return openAICatalog("o-series", "openai-gpt-flagship", "2025-04-16", "", llm.ModelLifecycleActive, ModelGPT5_5, openAIModelsSource), true
	case ModelO3Pro:
		return openAICatalog("o-series", "openai-gpt-pro", "2025-06-10", "", llm.ModelLifecycleActive, ModelGPT5_5Pro, openAIModelsSource), true
	case ModelO1Pro:
		return openAICatalog("o-series", "openai-gpt-pro", "2024-12-05", "2026-10-23", llm.ModelLifecycleDeprecated, ModelGPT5_5Pro, openAIDeprecationsSource), true
	case ModelO4Mini:
		return openAICatalog("o-series", "openai-gpt-balanced", "2025-04-16", "2026-10-23", llm.ModelLifecycleDeprecated, ModelGPT5_4Mini, openAIDeprecationsSource), true
	default:
		return llm.ModelCatalogMetadata{}, false
	}
}

func openAICatalog(
	familyKey string,
	upgradeGroup string,
	releaseDate string,
	endOfLifeDate string,
	lifecycle llm.ModelLifecycle,
	providerReplacement string,
	source string,
) llm.ModelCatalogMetadata {
	release, releaseErr := catalogdate.Parse(releaseDate)
	endOfLife, endOfLifeErr := catalogdate.Parse(endOfLifeDate)

	verified, verifiedErr := catalogdate.Parse(openAIMetadataVerifiedDate)
	if releaseErr != nil || endOfLifeErr != nil || verifiedErr != nil {
		return llm.ModelCatalogMetadata{}
	}

	return llm.ModelCatalogMetadata{
		FamilyKey:            familyKey,
		UpgradeGroup:         upgradeGroup,
		ReleaseDate:          release,
		EndOfLifeDate:        endOfLife,
		Lifecycle:            lifecycle,
		ProviderReplacement:  providerReplacement,
		OfficialSourceURL:    source,
		MetadataVerifiedDate: verified,
	}
}
