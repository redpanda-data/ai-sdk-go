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

import "github.com/redpanda-data/ai-sdk-go/llm"

const (
	anthropicModelsSource       = "https://platform.claude.com/docs/en/about-claude/models/overview"
	anthropicDeprecationsSource = "https://platform.claude.com/docs/en/about-claude/model-deprecations"
	anthropicVerifiedDate       = "2026-07-14"
)

// ModelCatalog returns factual catalog metadata for a canonical,
// aliased, or dated Anthropic model identifier.
func (*Provider) ModelCatalog(model string) (llm.ModelCatalogMetadata, bool) {
	return anthropicModelCatalog(resolveModelFamily(model))
}

func anthropicModelCatalog(name string) (llm.ModelCatalogMetadata, bool) {
	switch name {
	case ModelClaudeFable5:
		return anthropicCatalog("claude-fable", "2026-06-09", "", "", anthropicModelsSource), true
	case ModelClaudeOpus48:
		return anthropicCatalog("claude-opus", "2026-05-28", "", "", anthropicModelsSource), true
	case ModelClaudeOpus47:
		return anthropicCatalog("claude-opus", "2026-04-16", "", ModelClaudeOpus48, anthropicModelsSource), true
	case ModelClaudeOpus46:
		return anthropicCatalog("claude-opus", "2026-02-05", "", ModelClaudeOpus48, anthropicModelsSource), true
	case ModelClaudeOpus45:
		return anthropicCatalog("claude-opus", "2025-11-24", "", ModelClaudeOpus48, anthropicModelsSource), true
	case ModelClaudeOpus41:
		return anthropicCatalog("claude-opus", "2025-08-05", "2026-08-05", ModelClaudeOpus48, anthropicDeprecationsSource), true
	case ModelClaudeSonnet5:
		return anthropicCatalog("claude-sonnet", "2026-06-30", "", "", anthropicModelsSource), true
	case ModelClaudeSonnet46:
		return anthropicCatalog("claude-sonnet", "2026-02-17", "", ModelClaudeSonnet5, anthropicModelsSource), true
	case ModelClaudeSonnet45:
		return anthropicCatalog("claude-sonnet", "2025-09-29", "", ModelClaudeSonnet5, anthropicModelsSource), true
	case ModelClaudeHaiku45:
		return anthropicCatalog("claude-haiku", "2025-10-15", "", "", anthropicModelsSource), true
	default:
		return llm.ModelCatalogMetadata{}, false
	}
}

func anthropicCatalog(
	familyKey string,
	releaseDate string,
	endOfLifeDate string,
	replacement string,
	source string,
) llm.ModelCatalogMetadata {
	return llm.ModelCatalogMetadata{
		FamilyKey:           familyKey,
		RecommendationGroup: "anthropic-" + familyKey,
		ReleaseDate:         releaseDate,
		EndOfLifeDate:       endOfLifeDate,
		Replacement:         replacement,
		OfficialSourceURL:   source,
		VerifiedDate:        anthropicVerifiedDate,
	}
}
