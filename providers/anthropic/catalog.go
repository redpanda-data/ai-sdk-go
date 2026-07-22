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

package anthropic

import (
	"strings"
	"time"

	"github.com/redpanda-data/ai-sdk-go/internal/catalogdate"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	anthropicModelsSource         = "https://platform.claude.com/docs/en/about-claude/models/overview"
	anthropicDeprecationsSource   = "https://platform.claude.com/docs/en/about-claude/model-deprecations"
	anthropicMetadataVerifiedDate = "2026-07-14"
)

// ModelCatalog returns factual catalog metadata for a canonical,
// aliased, or dated Anthropic model identifier.
func (*Provider) ModelCatalog(model string) (llm.ModelCatalogMetadata, bool) {
	model, ok := anthropicCatalogModel(model)
	if !ok {
		return llm.ModelCatalogMetadata{}, false
	}

	return anthropicModelCatalog(model)
}

// anthropicCatalogModel accepts exact canonical IDs for Claude 4.6 and newer.
// Older generations also accept the documented aliases and YYYYMMDD snapshots.
// Runtime custom-model resolution remains deliberately more permissive.
func anthropicCatalogModel(model string) (string, bool) {
	if _, ok := supportedModels[model]; ok {
		return model, true
	}

	for _, family := range []string{
		ModelClaudeOpus45,
		ModelClaudeOpus41,
		ModelClaudeSonnet45,
		ModelClaudeHaiku45,
	} {
		date := strings.TrimPrefix(model, family+"-")
		if date == model || len(date) != len("YYYYMMDD") {
			continue
		}

		if _, err := time.Parse("20060102", date); err == nil {
			return family, true
		}
	}

	return "", false
}

func anthropicModelCatalog(name string) (llm.ModelCatalogMetadata, bool) {
	switch name {
	case ModelClaudeFable5:
		return anthropicCatalog("claude-fable", "2026-06-09", "", llm.ModelLifecycleActive, "", anthropicModelsSource), true
	case ModelClaudeOpus48:
		return anthropicCatalog("claude-opus", "2026-05-28", "", llm.ModelLifecycleActive, "", anthropicModelsSource), true
	case ModelClaudeOpus47:
		return anthropicCatalog("claude-opus", "2026-04-16", "", llm.ModelLifecycleActive, ModelClaudeOpus48, anthropicModelsSource), true
	case ModelClaudeOpus46:
		return anthropicCatalog("claude-opus", "2026-02-05", "", llm.ModelLifecycleActive, ModelClaudeOpus48, anthropicModelsSource), true
	case ModelClaudeOpus45:
		return anthropicCatalog("claude-opus", "2025-11-24", "", llm.ModelLifecycleActive, ModelClaudeOpus48, anthropicModelsSource), true
	case ModelClaudeOpus41:
		return anthropicCatalog("claude-opus", "2025-08-05", "2026-08-05", llm.ModelLifecycleDeprecated, ModelClaudeOpus48, anthropicDeprecationsSource), true
	case ModelClaudeSonnet5:
		return anthropicCatalog("claude-sonnet", "2026-06-30", "", llm.ModelLifecycleActive, "", anthropicModelsSource), true
	case ModelClaudeSonnet46:
		return anthropicCatalog("claude-sonnet", "2026-02-17", "", llm.ModelLifecycleActive, ModelClaudeSonnet5, anthropicModelsSource), true
	case ModelClaudeSonnet45:
		return anthropicCatalog("claude-sonnet", "2025-09-29", "", llm.ModelLifecycleActive, ModelClaudeSonnet5, anthropicModelsSource), true
	case ModelClaudeHaiku45:
		return anthropicCatalog("claude-haiku", "2025-10-15", "", llm.ModelLifecycleActive, "", anthropicModelsSource), true
	default:
		return llm.ModelCatalogMetadata{}, false
	}
}

func anthropicCatalog(
	familyKey string,
	releaseDate string,
	endOfLifeDate string,
	lifecycle llm.ModelLifecycle,
	providerReplacement string,
	source string,
) llm.ModelCatalogMetadata {
	release, releaseErr := catalogdate.Parse(releaseDate)
	endOfLife, endOfLifeErr := catalogdate.Parse(endOfLifeDate)

	verified, verifiedErr := catalogdate.Parse(anthropicMetadataVerifiedDate)
	if releaseErr != nil || endOfLifeErr != nil || verifiedErr != nil {
		return llm.ModelCatalogMetadata{}
	}

	return llm.ModelCatalogMetadata{
		FamilyKey:            familyKey,
		UpgradeGroup:         "anthropic-" + familyKey,
		ReleaseDate:          release,
		EndOfLifeDate:        endOfLife,
		Lifecycle:            lifecycle,
		ProviderReplacement:  providerReplacement,
		OfficialSourceURL:    source,
		MetadataVerifiedDate: verified,
	}
}
