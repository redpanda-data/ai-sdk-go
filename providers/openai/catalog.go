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

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	openAIModelsSource       = "https://developers.openai.com/api/docs/models"
	openAIDeprecationsSource = "https://developers.openai.com/api/docs/deprecations"
	openAIVerifiedDate       = "2026-07-13"
)

var openAIExactModelCatalogOverrides = map[string]llm.ModelCatalogMetadata{
	"gpt-5-2025-08-07":      openAICatalog("gpt", "openai-gpt-flagship", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-12-11", ModelGPT5_5, openAIDeprecationsSource),
	"gpt-5-mini-2025-08-07": openAICatalog("gpt", "openai-gpt-balanced", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-12-11", ModelGPT5_4Mini, openAIDeprecationsSource),
	"gpt-5-nano-2025-08-07": openAICatalog("gpt", "openai-gpt-efficient", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-12-11", ModelGPT5_4Nano, openAIDeprecationsSource),
	"o3-2025-04-16":         openAICatalog("o-series", "openai-o-reasoning", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-12-11", ModelGPT5_5, openAIDeprecationsSource),
	"o3-pro-2025-06-10":     openAICatalog("o-series", "openai-o-pro", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-12-11", ModelGPT5_5Pro, openAIDeprecationsSource),
	"gpt-4-turbo-preview":   openAICatalog("gpt", "openai-gpt-flagship", llm.ModelPositioningLegacy, llm.ModelLifecycleRetired, "2026-03-26", ModelGPT5_5, openAIDeprecationsSource),
}

// ModelCatalog returns recommendation and lifecycle metadata for a canonical,
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

// ModelCatalogOverrides returns exact deprecated or retired model IDs that
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
		return openAICatalog("gpt", "openai-gpt-flagship", llm.ModelPositioningFrontier, llm.ModelLifecycleActive, "", "", openAIModelsSource), true
	case ModelGPT5_6Terra:
		return openAICatalog("gpt", "openai-gpt-balanced", llm.ModelPositioningModern, llm.ModelLifecycleActive, "", "", openAIModelsSource), true
	case ModelGPT5_6Luna:
		return openAICatalog("gpt", "openai-gpt-efficient", llm.ModelPositioningModern, llm.ModelLifecycleActive, "", "", openAIModelsSource), true

	case ModelGPT5_5, ModelGPT5_4, ModelGPT5_2, ModelGPT5_1, ModelGPT41:
		return openAICatalog("gpt", "openai-gpt-flagship", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, "", ModelGPT5_6Sol, openAIModelsSource), true
	case ModelGPT5_5Pro:
		return openAICatalog("gpt", "openai-gpt-pro", llm.ModelPositioningModern, llm.ModelLifecycleActive, "", "", openAIModelsSource), true
	case ModelGPT4O:
		return openAICatalog("gpt", "openai-gpt-flagship", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "", ModelGPT5_6Sol, openAIModelsSource), true
	case ModelGPT5_4Mini, ModelGPT41Mini, ModelGPT4OMini:
		return openAICatalog("gpt", "openai-gpt-balanced", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, "", ModelGPT5_6Terra, openAIModelsSource), true
	case ModelGPT5_4Nano:
		return openAICatalog("gpt", "openai-gpt-efficient", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, "", ModelGPT5_6Luna, openAIModelsSource), true
	case ModelGPT5_2Pro:
		return openAICatalog("gpt", "openai-gpt-pro", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, "", ModelGPT5_5Pro, openAIModelsSource), true
	case ModelGPT5_3Codex:
		return openAICatalog("codex", "openai-codex", llm.ModelPositioningModern, llm.ModelLifecycleActive, "", "", openAIModelsSource), true

	case ModelGPT5:
		return openAICatalog("gpt", "openai-gpt-flagship", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, "", ModelGPT5_5, openAIModelsSource), true
	case ModelGPT5Mini:
		return openAICatalog("gpt", "openai-gpt-balanced", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, "", ModelGPT5_4Mini, openAIModelsSource), true
	case ModelGPT5Nano:
		return openAICatalog("gpt", "openai-gpt-efficient", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, "", ModelGPT5_4Nano, openAIModelsSource), true
	case ModelGPT5_2Instant, ModelGPT5_3ChatLatest:
		return openAICatalog("gpt", "openai-gpt-chat", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-08-10", ModelGPT5_5, openAIDeprecationsSource), true
	case ModelGPT4Turbo:
		return openAICatalog("gpt", "openai-gpt-flagship", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-10-23", ModelGPT5_5, openAIDeprecationsSource), true
	case ModelGPT35Turbo:
		return openAICatalog("gpt", "openai-gpt-balanced", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-10-23", ModelGPT5_4Mini, openAIDeprecationsSource), true

	case ModelO3:
		return openAICatalog("o-series", "openai-o-reasoning", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, "", ModelGPT5_5, openAIModelsSource), true
	case ModelO3Pro:
		return openAICatalog("o-series", "openai-o-pro", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, "", ModelGPT5_5Pro, openAIModelsSource), true
	case ModelO1Pro:
		return openAICatalog("o-series", "openai-o-pro", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-10-23", ModelGPT5_5Pro, openAIDeprecationsSource), true
	case ModelO4Mini:
		return openAICatalog("o-series", "openai-o-efficient", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, "2026-10-23", ModelGPT5_4Mini, openAIDeprecationsSource), true
	default:
		return llm.ModelCatalogMetadata{}, false
	}
}

func openAICatalog(
	familyKey string,
	recommendationGroup string,
	positioning llm.ModelPositioning,
	lifecycle llm.ModelLifecycle,
	endOfLifeDate string,
	replacement string,
	source string,
) llm.ModelCatalogMetadata {
	return llm.ModelCatalogMetadata{
		FamilyKey:           familyKey,
		RecommendationGroup: recommendationGroup,
		Positioning:         positioning,
		Lifecycle:           lifecycle,
		ReleaseStage:        llm.ModelReleaseStageStable,
		EndOfLifeDate:       endOfLifeDate,
		Replacement:         replacement,
		OfficialSourceURL:   source,
		VerifiedDate:        openAIVerifiedDate,
	}
}
