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

import (
	"maps"
	"strings"
	"unicode"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	googleModelsSource       = "https://ai.google.dev/gemini-api/docs/models"
	googleDeprecationsSource = "https://ai.google.dev/gemini-api/docs/deprecations"
	googleVerifiedDate       = "2026-07-14"
)

var googleExactModelCatalogOverrides = map[string]llm.ModelCatalogMetadata{
	"gemini-3.1-flash-lite-preview": googleCatalog("gemini-flash-lite", "2026-03-03", "2026-05-25", true, true, ModelGemini31FlashLite, googleDeprecationsSource),
	"gemini-2.0-flash":              googleCatalog("gemini-flash", "2025-02-05", "2026-06-01", true, true, ModelGemini35Flash, googleDeprecationsSource),
	"gemini-2.5-pro-preview-03-25":  googleCatalog("gemini-pro", "2025-03-25", "2025-12-02", true, true, ModelGemini31ProPreview, googleDeprecationsSource),
	"gemini-2.5-pro-preview-05-06":  googleCatalog("gemini-pro", "2025-05-06", "2025-12-02", true, true, ModelGemini31ProPreview, googleDeprecationsSource),
	"gemini-2.5-pro-preview-06-05":  googleCatalog("gemini-pro", "2025-06-05", "2025-12-02", true, true, ModelGemini31ProPreview, googleDeprecationsSource),
}

// ModelCatalog returns factual catalog metadata for a canonical,
// aliased, or versioned Gemini model identifier.
func (*Provider) ModelCatalog(model string) (llm.ModelCatalogMetadata, bool) {
	model = strings.TrimPrefix(model, "models/")

	if model == ModelGeminiFlashLatest {
		return googleModelCatalog(ModelGemini35Flash)
	}

	if catalog, ok := googleRetiredModelCatalog(model); ok {
		return catalog, true
	}

	if _, ok := supportedModels[model]; ok {
		return googleModelCatalog(model)
	}

	if family, ok := googleVersionedModelFamily(model); ok {
		return googleModelCatalog(family)
	}

	return llm.ModelCatalogMetadata{}, false
}

// ModelCatalogOverrides returns exact EOL or retired model IDs that
// remain outside Models() so they are not offered for new selections.
func (*Provider) ModelCatalogOverrides() map[string]llm.ModelCatalogMetadata {
	return maps.Clone(googleExactModelCatalogOverrides)
}

func googleRetiredModelCatalog(model string) (llm.ModelCatalogMetadata, bool) {
	catalog, ok := googleExactModelCatalogOverrides[model]

	return catalog, ok
}

func googleVersionedModelFamily(model string) (string, bool) {
	best := ""

	for family := range supportedModels {
		prefix := family + "-"
		if !strings.HasPrefix(model, prefix) || len(family) <= len(best) {
			continue
		}

		version := strings.TrimPrefix(model, prefix)
		if len(version) == 3 && strings.IndexFunc(version, func(r rune) bool { return !unicode.IsDigit(r) }) == -1 {
			best = family
		}
	}

	return best, best != ""
}

func googleModelCatalog(name string) (llm.ModelCatalogMetadata, bool) {
	switch name {
	case ModelGemini31ProPreview:
		return googleCatalog("gemini-pro", "2026-02-19", "", false, false, "", googleModelsSource), true
	case ModelGemini3ProPreview:
		return googleCatalog("gemini-pro", "2025-11-18", "2026-03-09", true, true, ModelGemini31ProPreview, googleDeprecationsSource), true
	case ModelGemini25Pro:
		return googleCatalog("gemini-pro", "2025-06-17", "2026-10-16", true, false, ModelGemini31ProPreview, googleDeprecationsSource), true
	case ModelGemini35Flash:
		return googleCatalog("gemini-flash", "2026-05-19", "", false, false, "", googleModelsSource), true
	case ModelGemini3FlashPreview:
		return googleCatalog("gemini-flash", "2025-12-17", "", false, false, ModelGemini35Flash, googleModelsSource), true
	case ModelGemini25Flash:
		return googleCatalog("gemini-flash", "2025-06-17", "2026-10-16", true, false, ModelGemini35Flash, googleDeprecationsSource), true
	case ModelGemini31FlashLite:
		return googleCatalog("gemini-flash-lite", "2026-05-07", "2027-05-07", true, false, "", googleDeprecationsSource), true
	case ModelGemini25FlashLite:
		return googleCatalog("gemini-flash-lite", "2025-07-22", "2026-10-16", true, false, ModelGemini31FlashLite, googleDeprecationsSource), true
	default:
		return llm.ModelCatalogMetadata{}, false
	}
}

func googleCatalog(
	familyKey string,
	releaseDate string,
	endOfLifeDate string,
	deprecated bool,
	retired bool,
	replacement string,
	source string,
) llm.ModelCatalogMetadata {
	return llm.ModelCatalogMetadata{
		FamilyKey:           familyKey,
		RecommendationGroup: "google-" + familyKey,
		ReleaseDate:         releaseDate,
		EndOfLifeDate:       endOfLifeDate,
		Deprecated:          deprecated,
		Retired:             retired,
		Replacement:         replacement,
		OfficialSourceURL:   source,
		VerifiedDate:        googleVerifiedDate,
	}
}
