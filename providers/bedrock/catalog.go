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

package bedrock

import (
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	bedrockVerifiedDate = "2026-07-13"
	bedrockGlobalRoute  = "global"

	bedrockFable5Source        = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5.html"
	bedrockOpus48Source        = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html"
	bedrockOpus47Source        = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-7.html"
	bedrockOpus46Source        = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-6.html"
	bedrockOpus45Source        = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-5.html"
	bedrockSonnet5Source       = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html"
	bedrockSonnet46Source      = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html"
	bedrockSonnet45Source      = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-5.html"
	bedrockHaiku45Source       = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html"
	bedrockNova2Source         = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html"
	bedrockMistralLarge3Source = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-mistral-large-3.html"
	bedrockGemma31BSource      = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-31b.html"
	bedrockGemma26BSource      = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-26b-a4b.html"
	bedrockGemmaE2BSource      = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-e2b.html"
)

// ModelCatalog returns recommendation and lifecycle metadata for a Bedrock
// model identifier from this provider's discovery catalog.
func (*Provider) ModelCatalog(model string) (llm.ModelCatalogMetadata, bool) {
	return bedrockModelCatalog(model)
}

func bedrockModelCatalog(name string) (llm.ModelCatalogMetadata, bool) {
	route := bedrockRoutingKey(name)

	switch name {
	case ModelClaudeFable5Global, ModelClaudeFable5US:
		return bedrockCatalog("claude-fable", "bedrock-claude-fable-"+route, llm.ModelPositioningFrontier, "", bedrockFable5Source), true
	case ModelClaudeFable5EU:
		return bedrockCatalog("claude-fable", "bedrock-claude-fable-"+route, llm.ModelPositioningLegacy, "", bedrockFable5Source), true

	case ModelClaudeOpus48Global, ModelClaudeOpus48US, ModelClaudeOpus48EU, ModelClaudeOpus48JP, ModelClaudeOpus48AU:
		return bedrockCatalog("claude-opus", "bedrock-claude-opus-"+route, llm.ModelPositioningModern, "", bedrockOpus48Source), true
	case ModelClaudeOpus47Global, ModelClaudeOpus47US, ModelClaudeOpus47EU, ModelClaudeOpus47JP, ModelClaudeOpus47AU:
		return bedrockCatalog("claude-opus", "bedrock-claude-opus-"+route, llm.ModelPositioningLegacy, bedrockRoutedModel(route, ModelClaudeOpus48), bedrockOpus47Source), true
	case ModelClaudeOpus46Global, ModelClaudeOpus46US, ModelClaudeOpus46EU, ModelClaudeOpus46AU:
		return bedrockCatalog("claude-opus", "bedrock-claude-opus-"+route, llm.ModelPositioningLegacy, bedrockRoutedModel(route, ModelClaudeOpus48), bedrockOpus46Source), true
	case ModelClaudeOpus45Global, ModelClaudeOpus45US, ModelClaudeOpus45EU:
		return bedrockCatalog("claude-opus", "bedrock-claude-opus-"+route, llm.ModelPositioningLegacy, bedrockRoutedModel(route, ModelClaudeOpus48), bedrockOpus45Source), true

	case ModelClaudeSonnet5Global, ModelClaudeSonnet5US:
		return bedrockCatalog("claude-sonnet", "bedrock-claude-sonnet-"+route, llm.ModelPositioningModern, "", bedrockSonnet5Source), true
	case ModelClaudeSonnet46EU, ModelClaudeSonnet46AU, ModelClaudeSonnet46JP:
		return bedrockCatalog("claude-sonnet", "bedrock-claude-sonnet-"+route, llm.ModelPositioningModern, "", bedrockSonnet46Source), true
	case ModelClaudeSonnet46Global, ModelClaudeSonnet46US:
		return bedrockCatalog("claude-sonnet", "bedrock-claude-sonnet-"+route, llm.ModelPositioningLegacy, bedrockRoutedModel(route, ModelClaudeSonnet5), bedrockSonnet46Source), true
	case ModelClaudeSonnet45Global, ModelClaudeSonnet45US:
		return bedrockCatalog("claude-sonnet", "bedrock-claude-sonnet-"+route, llm.ModelPositioningLegacy, bedrockRoutedModel(route, ModelClaudeSonnet5), bedrockSonnet45Source), true
	case ModelClaudeSonnet45EU, ModelClaudeSonnet45AU, ModelClaudeSonnet45JP:
		return bedrockCatalog("claude-sonnet", "bedrock-claude-sonnet-"+route, llm.ModelPositioningLegacy, bedrockRoutedModel(route, ModelClaudeSonnet46), bedrockSonnet45Source), true

	case ModelClaudeHaiku45Global, ModelClaudeHaiku45US, ModelClaudeHaiku45EU, ModelClaudeHaiku45AU, ModelClaudeHaiku45JP:
		return bedrockCatalog("claude-haiku", "bedrock-claude-haiku-"+route, llm.ModelPositioningModern, "", bedrockHaiku45Source), true
	case ModelNova2LiteGlobal, ModelNova2LiteUS, ModelNova2LiteEU, ModelNova2LiteJP:
		return bedrockCatalog("amazon-nova", "bedrock-amazon-nova-lite-"+route, llm.ModelPositioningModern, "", bedrockNova2Source), true
	case ModelMistralLarge3:
		return bedrockCatalog("mistral-large", "bedrock-mistral-large", llm.ModelPositioningModern, "", bedrockMistralLarge3Source), true
	case ModelGemma431B:
		return bedrockCatalog("google-gemma", "bedrock-google-gemma-31b", llm.ModelPositioningModern, "", bedrockGemma31BSource), true
	case ModelGemma426BA4B:
		return bedrockCatalog("google-gemma", "bedrock-google-gemma-26b-a4b", llm.ModelPositioningModern, "", bedrockGemma26BSource), true
	case ModelGemma4E2B:
		return bedrockCatalog("google-gemma", "bedrock-google-gemma-e2b", llm.ModelPositioningModern, "", bedrockGemmaE2BSource), true
	default:
		return llm.ModelCatalogMetadata{}, false
	}
}

func bedrockCatalog(
	familyKey string,
	recommendationGroup string,
	positioning llm.ModelPositioning,
	replacement string,
	source string,
) llm.ModelCatalogMetadata {
	return llm.ModelCatalogMetadata{
		FamilyKey:           familyKey,
		RecommendationGroup: recommendationGroup,
		Positioning:         positioning,
		Lifecycle:           llm.ModelLifecycleActive,
		ReleaseStage:        llm.ModelReleaseStageStable,
		Replacement:         replacement,
		OfficialSourceURL:   source,
		VerifiedDate:        bedrockVerifiedDate,
	}
}

func bedrockRoutingKey(name string) string {
	for _, route := range []string{bedrockGlobalRoute, "us", "eu", "jp", "au"} {
		if strings.HasPrefix(name, route+".") {
			return route
		}
	}

	return "regional"
}

func bedrockRoutedModel(route, bareModel string) string {
	if route == "regional" {
		return bareModel
	}

	return route + "." + bareModel
}
