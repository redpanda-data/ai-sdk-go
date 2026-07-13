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

package google

import "github.com/redpanda-data/ai-sdk-go/llm"

const (
	googleModelsSource       = "https://ai.google.dev/gemini-api/docs/models"
	googleDeprecationsSource = "https://ai.google.dev/gemini-api/docs/deprecations"
	googleVerifiedDate       = "2026-07-13"
)

// ModelCatalog returns recommendation and lifecycle metadata for a canonical,
// aliased, or versioned Gemini model identifier.
func (*Provider) ModelCatalog(model string) (llm.ModelCatalogMetadata, bool) {
	return googleModelCatalog(resolveModelFamily(model))
}

func googleModelCatalog(name string) (llm.ModelCatalogMetadata, bool) {
	switch name {
	case ModelGemini31ProPreview:
		return googleCatalog("gemini-pro", llm.ModelPositioningFrontier, llm.ModelLifecycleActive, llm.ModelReleaseStagePreview, "", "", googleModelsSource), true
	case ModelGemini3ProPreview:
		return googleCatalog("gemini-pro", llm.ModelPositioningLegacy, llm.ModelLifecycleRetired, llm.ModelReleaseStagePreview, "2026-03-09", ModelGemini31ProPreview, googleDeprecationsSource), true
	case ModelGemini25Pro:
		return googleCatalog("gemini-pro", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, llm.ModelReleaseStageStable, "2026-10-16", ModelGemini31ProPreview, googleDeprecationsSource), true
	case ModelGemini35Flash:
		return googleCatalog("gemini-flash", llm.ModelPositioningModern, llm.ModelLifecycleActive, llm.ModelReleaseStageStable, "", "", googleModelsSource), true
	case ModelGemini3FlashPreview:
		return googleCatalog("gemini-flash", llm.ModelPositioningLegacy, llm.ModelLifecycleActive, llm.ModelReleaseStagePreview, "", ModelGemini35Flash, googleDeprecationsSource), true
	case ModelGemini25Flash:
		return googleCatalog("gemini-flash", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, llm.ModelReleaseStageStable, "2026-10-16", ModelGemini35Flash, googleDeprecationsSource), true
	case ModelGemini31FlashLite:
		return googleCatalog("gemini-flash-lite", llm.ModelPositioningModern, llm.ModelLifecycleDeprecated, llm.ModelReleaseStageStable, "2027-05-07", "", googleDeprecationsSource), true
	case ModelGemini25FlashLite:
		return googleCatalog("gemini-flash-lite", llm.ModelPositioningLegacy, llm.ModelLifecycleDeprecated, llm.ModelReleaseStageStable, "2026-10-16", ModelGemini31FlashLite, googleDeprecationsSource), true
	default:
		return llm.ModelCatalogMetadata{}, false
	}
}

func googleCatalog(
	familyKey string,
	positioning llm.ModelPositioning,
	lifecycle llm.ModelLifecycle,
	releaseStage llm.ModelReleaseStage,
	endOfLifeDate string,
	replacement string,
	source string,
) llm.ModelCatalogMetadata {
	return llm.ModelCatalogMetadata{
		FamilyKey:           familyKey,
		RecommendationGroup: "google-" + familyKey,
		Positioning:         positioning,
		Lifecycle:           lifecycle,
		ReleaseStage:        releaseStage,
		EndOfLifeDate:       endOfLifeDate,
		Replacement:         replacement,
		OfficialSourceURL:   source,
		VerifiedDate:        googleVerifiedDate,
	}
}
