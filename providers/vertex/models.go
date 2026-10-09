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

// Package vertex is the model catalog, pricing, and location-availability
// data for Google Vertex AI. It has no request transport (an llm.Model that
// builds Vertex requests).
package vertex

import (
	"sync"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// ProviderName is the catalog key Provider.Name() returns; see
// pricing.ProviderKey. It mirrors Bedrock's "aws.bedrock" — cloud, then surface.
const ProviderName = "gcp.vertex"

// Vertex model IDs are exactly the model segment of a Vertex resource path,
// publishers/{publisher}/models/{model}.
const (
	ModelGemini38Flash       = "gemini-3.8-flash"
	ModelGemini37Flash       = "gemini-3.7-flash"
	ModelGemini36Flash       = "gemini-3.6-flash"
	ModelGemini35Flash       = "gemini-3.5-flash"
	ModelGemini35FlashLite   = "gemini-3.5-flash-lite"
	ModelGemini31FlashLite   = "gemini-3.1-flash-lite"
	ModelGemini31ProPreview  = "gemini-3.1-pro-preview"
	ModelGemini3FlashPreview = "gemini-3-flash-preview"
	// Deprecated: Google retires Gemini 2.5 Pro on 2026-10-20. Use
	// [ModelGemini38Flash] or [ModelGemini35Flash].
	ModelGemini25Pro = "gemini-2.5-pro"
	// Deprecated: Google retires Gemini 2.5 Flash on 2026-10-20. Use
	// [ModelGemini38Flash], [ModelGemini35FlashLite] or
	// [ModelGemini31FlashLite].
	ModelGemini25Flash = "gemini-2.5-flash"
	// Deprecated: Google retires Gemini 2.5 Flash-Lite on 2026-10-20. Use
	// [ModelGemini38Flash] or [ModelGemini31FlashLite].
	ModelGemini25FlashLite = "gemini-2.5-flash-lite"
	ModelClaudeFable51     = "claude-fable-5-1"
	ModelClaudeFable5      = "claude-fable-5"
	ModelClaudeOpus55      = "claude-opus-5-5"
	ModelClaudeOpus5       = "claude-opus-5"
	ModelClaudeOpus48      = "claude-opus-4-8"
	ModelClaudeOpus47      = "claude-opus-4-7"
	ModelClaudeOpus46      = "claude-opus-4-6"
	ModelClaudeOpus45      = "claude-opus-4-5@20251101"
	ModelClaudeSonnet55    = "claude-sonnet-5-5"
	ModelClaudeSonnet5     = "claude-sonnet-5"
	ModelClaudeSonnet46    = "claude-sonnet-4-6"
	ModelClaudeSonnet45    = "claude-sonnet-4-5@20250929"
	ModelClaudeHaiku45     = "claude-haiku-4-5"
)

// Reasoning-effort values Vertex accepts. llm.ReasoningEffort is an open
// string type whose valid vocabulary is provider-owned, so the two
// publishers do not share one set: Gemini's thinking levels are
// minimal/low/medium/high, and Claude's are low/medium/high/xhigh/max
// (Haiku 4.5 omits xhigh). These mirror the Gemini-API and Anthropic-direct
// catalogs and are the source of truth for Vertex.
const (
	reasoningEffortMinimal llm.ReasoningEffort = "minimal"
	reasoningEffortLow     llm.ReasoningEffort = "low"
	reasoningEffortMedium  llm.ReasoningEffort = "medium"
	reasoningEffortHigh    llm.ReasoningEffort = "high"
	reasoningEffortXHigh   llm.ReasoningEffort = "xhigh"
	reasoningEffortMax     llm.ReasoningEffort = "max"
)

// geminiReasoningEfforts is the thinking-level vocabulary shared by
// catalogued Gemini models on Vertex, matching the Gemini-API provider's
// set (providers/google/models.go).
var geminiReasoningEfforts = []llm.ReasoningEffort{
	reasoningEffortMinimal, reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh,
}

// geminiNoMinimalReasoningEfforts drops minimal: the Gemini 3.7 and 3.8 Flash
// model pages say an explicit MINIMAL "will return an API validation error".
var geminiNoMinimalReasoningEfforts = []llm.ReasoningEffort{
	reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh,
}

// geminiCaps is the capability set shared by every catalogued Gemini
// model on Vertex, matching the Gemini-API provider's set.
var geminiCaps = llm.ModelCapabilities{
	Streaming:        true,
	Tools:            true,
	JSONMode:         true, // via response_mime_type
	StructuredOutput: true,
	Vision:           true,
	Audio:            true,
	MultiTurn:        true,
	SystemPrompts:    true,
	Reasoning:        true,
}

var geminiModalities = catalog.Modalities{
	Input: []catalog.Modality{
		catalog.ModalityText, catalog.ModalityImage, catalog.ModalityAudio,
		catalog.ModalityVideo, catalog.ModalityDocument,
	},
	Output: []catalog.Modality{catalog.ModalityText},
}

// geminiParams is the request-parameter surface shared by catalogued
// Gemini models.
var geminiParams = []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"}

// geminiNoPenaltyParams is for models that return an error on a custom
// frequency or presence penalty: Gemini 3.6 Flash and later, per the
// content-generation parameters page, and 3.5 Flash-Lite, per its model
// page (both read 2026-09-28).
var geminiNoPenaltyParams = []string{"temperature", "top_p", "top_k", "max_tokens", "stop"}

// claudeCaps is the capability set shared by catalogued Claude models on
// Vertex. JSONMode is false because Claude has no schemaless JSON mode;
// structured output is via output_config with a json_schema.
var claudeCaps = llm.ModelCapabilities{
	Streaming:        true,
	Tools:            true,
	StructuredOutput: true,
	Vision:           true,
	MultiTurn:        true,
	SystemPrompts:    true,
	Reasoning:        true,
}

// claudeCapsWithToolSearch is claudeCaps plus hosted tool search, which
// Vertex serves: rawPredict at global returned a tool_search_tool_result
// for both tool types on 2026-09-29.
var claudeCapsWithToolSearch = func() llm.ModelCapabilities {
	caps := claudeCaps
	caps.ToolSearch = true

	return caps
}()

var claudeAllEfforts = []llm.ReasoningEffort{
	reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh, reasoningEffortXHigh, reasoningEffortMax,
}

var claudeModalities = catalog.Modalities{
	Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityDocument},
	Output: []catalog.Modality{catalog.ModalityText},
}

var catalogOnce = sync.OnceValue(func() *catalog.Catalog {
	return catalog.MustNew(ProviderName, entries())
})

// Catalog returns the validated Vertex model catalog: every offering with
// its capabilities, constraints, modalities, reasoning controls, pricing,
// and lifecycle. It is shared and immutable, so reads return deep copies.
func Catalog() *catalog.Catalog {
	return catalogOnce()
}

// entries returns the authored Vertex catalog. Rates are
// USD-per-million-tokens from Google's pricing page, Claude rates included
// (not Anthropic's list prices):
//
//	https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
//
// Capabilities and constraints mirror the same models in the Gemini-API and
// Anthropic-direct catalogs, with four exceptions. No Claude entry offers
// speed; Vertex publishes no Claude fast-mode rate. Several Gemini entries
// cap output at 65536 where the Gemini-API catalog says 65535. Gemini 3.7
// and 3.8 Flash take no presence or frequency penalty, which the Gemini-API
// catalog lists. Claude Opus 4.5, Sonnet 4.5 and Haiku 4.5 take a manual
// thinking budget, which the Anthropic-direct catalog does not set.
//
// Lifecycle does not mirror them: each entry's Available is when Vertex
// began serving the model (its model page's Release date). Retires is set
// only where Google announced a shutdown date; a "not sooner than" floor
// leaves it unset.
func entries() []catalog.Entry {
	return []catalog.Entry{
		{
			ID:           ModelGemini38Flash,
			Model:        catalog.ModelGemini38Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiNoPenaltyParams,
			},
			Reasoning: catalog.ReasoningSupport{Efforts: geminiNoMinimalReasoningEfforts},
			// Gemini 3.8 Flash GA on the model page (docs.cloud.google.com/
			// gemini-enterprise-agent-platform/models/gemini/3-8-flash, read
			// 2026-09-22). No retirement published.
			Life: catalog.Lifecycle{Available: catalog.MustDate("2026-09-02")},
			// Same introductory rates and regional premium as 3.6 Flash.
			Pricing: geminiFlashPricing(),
		},
		{
			ID:           ModelGemini37Flash,
			Model:        catalog.ModelGemini37Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiNoPenaltyParams,
			},
			Reasoning: catalog.ReasoningSupport{Efforts: geminiNoMinimalReasoningEfforts},
			// Gemini 3.7 Flash GA, "No retirement date announced" on the model page
			// (docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/
			// 3-7-flash, read 2026-09-28).
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2026-08-13"),
				ReplacedBy: ModelGemini38Flash,
			},
			Pricing: geminiFlashPricing(),
		},
		{
			ID:           ModelGemini36Flash,
			Model:        catalog.ModelGemini36Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiNoPenaltyParams,
			},
			Reasoning: catalog.ReasoningSupport{Efforts: geminiReasoningEfforts},
			// Gemini 3.6 Flash GA on the model page (docs.cloud.google.com/
			// gemini-enterprise-agent-platform/models/gemini/3-6-flash, read
			// 2026-09-10). No retirement published.
			// model-versions names gemini-3.8-flash as the replacement.
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2026-07-21"),
				ReplacedBy: ModelGemini38Flash,
			},
			Pricing: geminiFlashPricing(),
		},
		{
			ID:           ModelGemini35Flash,
			Model:        catalog.ModelGemini35Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			Reasoning: catalog.ReasoningSupport{Efforts: geminiReasoningEfforts},
			// Gemini 3.5 Flash GA, retirement "May 19, 2027 or later" (a floor) on
			// the model page (docs.cloud.google.com/gemini-enterprise-agent-platform/
			// models/gemini/3-5-flash, read 2026-09-28).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-05-19")},
			Pricing: gemini35FlashPricing(),
		},
		{
			ID:           ModelGemini35FlashLite,
			Model:        catalog.ModelGemini35FlashLite,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiNoPenaltyParams,
			},
			Reasoning: catalog.ReasoningSupport{Efforts: geminiReasoningEfforts},
			// Gemini 3.5 Flash-Lite GA, retirement "July 21, 2027 or later" (a floor)
			// on the model page (docs.cloud.google.com/gemini-enterprise-agent-
			// platform/models/gemini/3-5-flash-lite, read 2026-09-28).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-07-21")},
			Pricing: gemini35FlashLitePricing(),
		},
		{
			ID:           ModelGemini31FlashLite,
			Model:        catalog.ModelGemini31FlashLite,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			Reasoning: catalog.ReasoningSupport{Efforts: geminiReasoningEfforts},
			// Gemini 3.1 Flash-Lite GA, retirement "May 7, 2027 or later" (a floor)
			// on the model page (docs.cloud.google.com/gemini-enterprise-agent-
			// platform/models/gemini/3-1-flash-lite, read 2026-09-22).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-05-07")},
			Pricing: geminiFlashLitePricing(),
		},
		{
			ID:           ModelGemini31ProPreview,
			Model:        catalog.ModelGemini31ProPreview,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			// Thinking cannot be turned off; high is the default.
			Reasoning: catalog.ReasoningSupport{Efforts: []llm.ReasoningEffort{
				reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh,
			}},
			// Gemini 3.1 Pro public preview on the model page (docs.cloud.google.com/
			// gemini-enterprise-agent-platform/models/gemini/3-1-pro, read
			// 2026-09-28). No retirement published.
			Life: catalog.Lifecycle{
				Stage:     catalog.StagePreview,
				Available: catalog.MustDate("2026-02-19"),
			},
			// Served at global only. Standard rates from the pricing page, read
			// 2026-09-28.
			Pricing: pricing.TieredInfo(
				pricing.NewRates(2.00, 12.00, 0.20),
				pricing.Bracket{
					MinContextTokens: 200_001,
					Rates:            pricing.NewRates(4.00, 18.00, 0.40),
				},
			),
		},
		{
			ID:           ModelGemini3FlashPreview,
			Model:        catalog.ModelGemini3FlashPreview,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			Reasoning: catalog.ReasoningSupport{Efforts: geminiReasoningEfforts},
			// Gemini 3 Flash public preview on the model page (docs.cloud.google.com/
			// gemini-enterprise-agent-platform/models/gemini/3-flash, read
			// 2026-09-28). No retirement published.
			Life: catalog.Lifecycle{
				Stage:     catalog.StagePreview,
				Available: catalog.MustDate("2025-12-17"),
			},
			// Served at global only. The pricing page's Standard table omits this
			// model's output row (read 2026-09-28); the Cloud Billing Catalog
			// publishes all three rates for Vertex AI (service C7E2-9256-1C43,
			// region global, read 2026-09-28): input SKU 7EBE-3B46-F75C,
			// output SKU 0127-F0B7-365E, cache read SKU E5C2-A033-7712.
			Pricing: pricing.FlatInfo(0.50, 3.00, 0.05),
		},
		{
			ID:           ModelGemini25Pro,
			Model:        catalog.ModelGemini25Pro,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			// The thinking budget on 2.5 Pro is 128 to 32,768 tokens.
			Reasoning: catalog.ReasoningSupport{Adaptive: true, Budget: true},
			// Gemini 2.5 Pro GA on the model page
			// (docs.cloud.google.com/gemini-enterprise-agent-platform/models/
			// gemini/2-5-pro, read 2026-09-28). Retires is the bare retirement
			// date on that page and on model-versions, both last updated
			// 2026-09-25. Google may extend it but won't move it earlier
			// (model-versions). ReplacedBy is the first replacement
			// model-versions lists, "gemini-3.8-flash or gemini-3.5-flash"
			// (last updated 2026-10-07).
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-06-17"),
				Retires:    catalog.MustDate("2026-10-20"),
				ReplacedBy: ModelGemini38Flash,
			},
			// Flat across locations.
			Pricing: pricing.TieredInfo(
				pricing.NewRates(1.25, 10.00, 0.125),
				pricing.Bracket{
					MinContextTokens: 200_001,
					Rates:            pricing.NewRates(2.50, 15.00, 0.25),
				},
			),
		},
		{
			ID:           ModelGemini25Flash,
			Model:        catalog.ModelGemini25Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			Reasoning: catalog.ReasoningSupport{Adaptive: true, Budget: true},
			// Gemini 2.5 Flash GA on the model page
			// (docs.cloud.google.com/gemini-enterprise-agent-platform/models/
			// gemini/2-5-flash, read 2026-09-28). Retires is the bare
			// retirement date on the model page and model-versions.
			// ReplacedBy is the first replacement model-versions lists,
			// "gemini-3.8-flash or gemini-3.5-flash-lite or
			// gemini-3.1-flash-lite" (last updated 2026-10-07).
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-06-17"),
				Retires:    catalog.MustDate("2026-10-20"),
				ReplacedBy: ModelGemini38Flash,
			},
			// Text/image/video rate only; audio input is $1.00.
			Pricing: pricing.FlatInfo(0.30, 2.50, 0.03),
		},
		{
			ID:           ModelGemini25FlashLite,
			Model:        catalog.ModelGemini25FlashLite,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			Reasoning: catalog.ReasoningSupport{Adaptive: true, Budget: true},
			// Gemini 2.5 Flash-Lite GA on the model page
			// (docs.cloud.google.com/gemini-enterprise-agent-platform/models/
			// gemini/2-5-flash-lite, read 2026-09-28). Retires is the bare
			// retirement date on the model page and model-versions.
			// ReplacedBy is the first replacement model-versions lists,
			// "gemini-3.8-flash or gemini-3.1-flash-lite or Gemma 4" (last
			// updated 2026-10-07).
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-07-22"),
				Retires:    catalog.MustDate("2026-10-20"),
				ReplacedBy: ModelGemini38Flash,
			},
			// Text/image/video rate only; audio input is $0.30.
			Pricing: pricing.FlatInfo(0.10, 0.40, 0.01),
		},
		{
			ID:           ModelClaudeFable51,
			Model:        catalog.ModelClaudeFable51,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000,
				MaxOutputTokens: 128000,
				// Thinking is always on; non-default sampling parameters
				// return 400.
				SupportedParams: []string{"max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{Efforts: claudeAllEfforts, Adaptive: true},
			// Claude Fable 5.1 GA, retirement floor "No sooner than March 1, 2027"
			// on the model page (docs.cloud.google.com/gemini-enterprise-agent-
			// platform/models/partner-models/claude/fable-5-1, read 2026-09-28).
			Life: catalog.Lifecycle{Available: catalog.MustDate("2026-09-01")},
			// Cache hits are 0.025x input; the pricing page prints it in every
			// tab (read 2026-09-28).
			Pricing: regionalPricing(
				pricing.NewRates(10.00, 50.00, 0.25).WithCacheCreation(12.50, 20.00, 0),
				pricing.NewRates(11.00, 55.00, 0.275).WithCacheCreation(13.75, 22.00, 0),
				"us", "eu",
			),
		},
		{
			ID:           ModelClaudeFable5,
			Model:        catalog.ModelClaudeFable5,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000,
				MaxOutputTokens: 128000,
				SupportedParams: []string{"max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{Efforts: claudeAllEfforts, Adaptive: true},
			// Claude Fable 5 GA, retirement floor "Not sooner than June 8, 2027"
			// on the model page (docs.cloud.google.com/gemini-enterprise-agent-
			// platform/models/partner-models/claude/fable-5, read 2026-09-28).
			Life: catalog.Lifecycle{Available: catalog.MustDate("2026-06-09")},
			// asia-southeast1 carries the 10% premium that Anthropic's pricing
			// page (platform.claude.com/docs/en/about-claude/pricing, read
			// 2026-09-28) puts on every Google Cloud regional and multi-region
			// endpoint for Claude 4.5 and later models.
			Pricing: regionalPricing(
				pricing.NewRates(10.00, 50.00, 1.00).WithCacheCreation(12.50, 20.00, 0),
				pricing.NewRates(11.00, 55.00, 1.10).WithCacheCreation(13.75, 22.00, 0),
				"us", "eu", "asia-southeast1",
			),
		},
		{
			ID:           ModelClaudeOpus55,
			Model:        catalog.ModelClaudeOpus55,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000,
				MaxOutputTokens: 128000,
				// Adaptive thinking is always on; non-default sampling
				// parameters return 400 (Anthropic's model-deprecations page).
				// Vertex publishes no fast-mode rate, so speed is not offered.
				SupportedParams: []string{"max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []llm.ReasoningEffort{reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh, reasoningEffortXHigh, reasoningEffortMax},
				Adaptive: true,
			},
			// Claude Opus 5.5 GA, retirement floor "not sooner than 2027-09-22" on
			// the model page (docs.cloud.google.com/gemini-enterprise-agent-platform/
			// models/partner-models/claude/opus-5-5, read 2026-09-22).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-09-22")},
			Pricing: claudeOpus55Pricing(),
		},
		{
			ID:           ModelClaudeOpus5,
			Model:        catalog.ModelClaudeOpus5,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000,
				MaxOutputTokens: 128000,
				SupportedParams: []string{"max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{Efforts: claudeAllEfforts, Adaptive: true},
			// Claude Opus 5 GA, retirement floor "Not sooner than January 24,
			// 2027" on the model page (docs.cloud.google.com/gemini-enterprise-
			// agent-platform/models/partner-models/claude/opus-5, read 2026-09-28).
			Life: catalog.Lifecycle{Available: catalog.MustDate("2026-07-24")},
			// asia-southeast1 has no pricing tab for this model. Anthropic's
			// pricing page (platform.claude.com/docs/en/about-claude/pricing,
			// read 2026-09-28) puts a 10% premium on every Google Cloud regional
			// and multi-region endpoint for Claude 4.5 and later models.
			Pricing: claudeOpusPricing("us", "eu", "asia-southeast1"),
		},
		{
			ID:           ModelClaudeOpus48,
			Model:        catalog.ModelClaudeOpus48,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000,
				MaxOutputTokens: 128000,
				SupportedParams: []string{"max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{Efforts: claudeAllEfforts, Adaptive: true},
			// Claude Opus 4.8 GA, retirement floor "Not sooner than May 28, 2027"
			// on the model page (docs.cloud.google.com/gemini-enterprise-agent-
			// platform/models/partner-models/claude/opus-4-8, read 2026-09-28).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-05-28")},
			Pricing: claudeOpusPricing("us", "eu"),
		},
		{
			ID:           ModelClaudeOpus47,
			Model:        catalog.ModelClaudeOpus47,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000,
				MaxOutputTokens: 128000,
				SupportedParams: []string{"max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{Efforts: claudeAllEfforts, Adaptive: true},
			// Claude Opus 4.7 GA, retirement floor "Not sooner than April 16,
			// 2027" on the model page (docs.cloud.google.com/gemini-enterprise-
			// agent-platform/models/partner-models/claude/opus-4-7, read 2026-09-28).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-04-15")},
			Pricing: claudeOpusPricing("us", "eu"),
		},
		{
			ID:           ModelClaudeOpus46,
			Model:        catalog.ModelClaudeOpus46,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   1000000,
				MaxOutputTokens:  128000,
				SupportedParams:  []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "thinking_budget"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []llm.ReasoningEffort{reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh, reasoningEffortMax},
				Adaptive: true,
				Budget:   true,
			},
			// Claude Opus 4.6 GA, retirement floor "Not sooner than February 5,
			// 2027" on the model page (docs.cloud.google.com/gemini-enterprise-
			// agent-platform/models/partner-models/claude/opus-4-6, read 2026-09-28).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-02-05")},
			Pricing: claudeOpusPricing("us-east5", "europe-west1", "asia-southeast1"),
		},
		{
			ID:    ModelClaudeOpus45,
			Model: catalog.ModelClaudeOpus45,
			// Vertex serves the bare ID as the same model version (rawPredict
			// at global and us-east5, 2026-09-28).
			Aliases:      []string{"claude-opus-4-5"},
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   200000,
				MaxOutputTokens:  64000,
				SupportedParams:  []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "thinking_budget"},
			},
			// thinking.type adaptive returns 400; effort is set alongside
			// budget_tokens.
			Reasoning: catalog.ReasoningSupport{
				Efforts: []llm.ReasoningEffort{reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh},
				Budget:  true,
			},
			// Claude Opus 4.5 GA on the model page (docs.cloud.google.com/
			// gemini-enterprise-agent-platform/models/partner-models/claude/
			// opus-4-5, read 2026-09-28); retirement floor "not sooner than:
			// November 24, 2026".
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2025-11-24")},
			Pricing: claudeOpusPricing("us-east5", "europe-west1", "asia-southeast1"),
		},
		{
			ID:           ModelClaudeSonnet55,
			Model:        catalog.ModelClaudeSonnet55,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000,
				MaxOutputTokens: 128000,
				// Adaptive thinking only; sampling parameters return 400.
				SupportedParams: []string{"max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []llm.ReasoningEffort{reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh, reasoningEffortXHigh, reasoningEffortMax},
				Adaptive: true,
			},
			// GA 2026-09-28; retirement floor 2027-09-28 is not an exact date.
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-09-28")},
			Pricing: claudeSonnet55Pricing(),
		},
		{
			ID:           ModelClaudeSonnet5,
			Model:        catalog.ModelClaudeSonnet5,
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				MaxInputTokens:  1000000,
				MaxOutputTokens: 128000,
				// Claude 4.7 and later return 400 for non-default sampling
				// parameters (Anthropic's model-deprecations page).
				SupportedParams: []string{"max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []llm.ReasoningEffort{reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh, reasoningEffortXHigh, reasoningEffortMax},
				Adaptive: true,
			},
			// Claude Sonnet 5 GA, retirement floor "not sooner than 2026-12-24" on
			// the model page (docs.cloud.google.com/gemini-enterprise-agent-platform/
			// models/partner-models/claude/sonnet-5, read 2026-09-10).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-06-30")},
			Pricing: claudeSonnet5Pricing(),
		},
		{
			ID:           ModelClaudeSonnet46,
			Model:        catalog.ModelClaudeSonnet46,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   1000000,
				MaxOutputTokens:  128000,
				SupportedParams:  []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "thinking_budget"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []llm.ReasoningEffort{reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh, reasoningEffortMax},
				Adaptive: true,
				Budget:   true,
			},
			// Claude Sonnet 4.6 GA on the model page (docs.cloud.google.com/
			// gemini-enterprise-agent-platform/models/partner-models/claude/
			// sonnet-4-6, read 2026-09-28), which shows no retirement date.
			Life: catalog.Lifecycle{Available: catalog.MustDate("2026-02-17")},
			Pricing: regionalPricing(
				pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
				pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
				"us-east5", "europe-west1", "asia-southeast1",
			),
		},
		{
			ID:    ModelClaudeSonnet45,
			Model: catalog.ModelClaudeSonnet45,
			// Vertex serves the bare ID as the same model version (rawPredict
			// at global and us-east5, 2026-09-28).
			Aliases:      []string{"claude-sonnet-4-5"},
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				// 200K is the GA window; the 1M window is a preview.
				MaxInputTokens:  200000,
				MaxOutputTokens: 64000,
				SupportedParams: []string{"temperature", "top_p", "top_k", "max_tokens", "thinking_budget"},
			},
			// A thinking.type of adaptive returns 400.
			Reasoning: catalog.ReasoningSupport{Budget: true},
			// Claude Sonnet 4.5 GA, retirement floor "not sooner than: September
			// 29, 2026" on the model page (docs.cloud.google.com/
			// gemini-enterprise-agent-platform/models/partner-models/claude/
			// sonnet-4-5, read 2026-09-28).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2025-09-29")},
			Pricing: claudeSonnet45Pricing(),
		},
		{
			ID:           ModelClaudeHaiku45,
			Model:        catalog.ModelClaudeHaiku45,
			Capabilities: claudeCapsWithToolSearch,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   200000,
				MaxOutputTokens:  64000,
				SupportedParams:  []string{"temperature", "top_p", "top_k", "max_tokens", "thinking_budget"},
			},
			// A thinking.type of adaptive returns 400 (Anthropic's
			// thinking-troubleshooting page, and rawPredict at global, 2026-09-28).
			Reasoning: catalog.ReasoningSupport{Budget: true},
			// Claude Haiku 4.5 GA, retirement floor "not sooner than 2026-10-15" on
			// the model page (docs.cloud.google.com/gemini-enterprise-agent-platform/
			// models/partner-models/claude/haiku-4-5, read 2026-09-10).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2025-10-15")},
			Pricing: claudeHaiku45Pricing(),
		},
	}
}

// geminiFlashLitePricing returns the Gemini 3.1 Flash-Lite rate card, same
// shape as [geminiFlashPricing]. Standard rates from the pricing page, read
// 2026-09-22; no introductory rate. Text/image/video rates only: audio input
// ($0.50 global, $0.55 non-global) has no per-modality bucket.
func geminiFlashLitePricing() pricing.Info {
	global := pricing.NewRates(0.25, 1.50, 0.025)
	nonGlobal := pricing.NewRates(0.275, 1.65, 0.0275)

	info := pricing.FlatInfoFromRates(global)
	for _, region := range []string{"us", "eu"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}

// claudeOpus55Pricing returns the Opus 5.5 rate card, same shape as
// [claudeSonnet5Pricing]. Global, and non-global = global x 1.10, from the
// pricing page's region tabs, read 2026-09-22; flat across the =< 200K and
// > 200K input tiers. Cache hits are 0.05x input, as on Anthropic direct.
// Only the us and eu tabs list Opus 5.5.
func claudeOpus55Pricing() pricing.Info {
	global := pricing.NewRates(4.00, 20.00, 0.20).WithCacheCreation(5.00, 8.00, 0)
	nonGlobal := pricing.NewRates(4.40, 22.00, 0.22).WithCacheCreation(5.50, 8.80, 0)

	info := pricing.FlatInfoFromRates(global)
	for _, region := range []string{"us", "eu"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}

// claudeSonnet55Pricing returns the Sonnet 5.5 rate card: global, plus
// global x 1.10 for us and eu.
func claudeSonnet55Pricing() pricing.Info {
	global := pricing.NewRates(2.00, 10.00, 0.20).WithCacheCreation(2.50, 4.00, 0)
	nonGlobal := pricing.NewRates(2.20, 11.00, 0.22).WithCacheCreation(2.75, 4.40, 0)

	info := pricing.FlatInfoFromRates(global)
	for _, region := range []string{"us", "eu"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}

// geminiFlashPricing returns the Gemini 3.6, 3.7 and 3.8 Flash rate card.
func geminiFlashPricing() pricing.Info {
	// Promotional net rate through 2026-12-31 (pricing page footnote, read
	// 2026-09-28): the billing SKUs stay at the standard $1.50/$7.50/$0.15
	// global and $1.65/$8.25 non-global, and Google returns 50% as credits
	// back on net spend. The standard rate applies from 2027-01-01.
	global := pricing.NewRates(0.75, 3.75, 0.075)
	// Non-global = global x 1.10 on input, output, and cache alike, from
	// Google's pricing page, verified live 2026-09-08.
	nonGlobal := pricing.NewRates(0.825, 4.125, 0.0825)

	info := pricing.FlatInfoFromRates(global)
	// An empty-Region selector is a wildcard that also matches global.
	for _, region := range []string{"us", "eu"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}

// claudeSonnet5Pricing returns the Sonnet 5 rate card: the global rate as
// default, and Google's flat 10% non-global premium as one override per
// served non-global region. (Source rationale in entries().)
func claudeSonnet5Pricing() pricing.Info {
	// Global, and non-global = global x 1.10 on input, output, cache write,
	// and cache read alike, from the pricing page's region tabs, read
	// 2026-09-08. Flat across the page's =< 200K and > 200K input tiers.
	global := pricing.NewRates(2.00, 10.00, 0.20).WithCacheCreation(2.50, 4.00, 0)
	nonGlobal := pricing.NewRates(2.20, 11.00, 0.22).WithCacheCreation(2.75, 4.40, 0)

	info := pricing.FlatInfoFromRates(global)
	// asia-southeast1 has no pricing tab for Sonnet 5; its premium is from
	// Anthropic's pricing page (platform.claude.com/docs/en/about-claude/
	// pricing, read 2026-09-28), which puts it on every Google Cloud regional
	// and multi-region endpoint for Claude 4.5 and later models.
	for _, region := range []string{"us", "eu", "asia-southeast1"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}

// claudeHaiku45Pricing returns the Haiku 4.5 rate card, same shape as
// [claudeSonnet5Pricing].
func claudeHaiku45Pricing() pricing.Info {
	// Global, and non-global = global x 1.10, from the pricing page's region
	// tabs, read 2026-09-08. Flat across the =< 200K and > 200K input tiers.
	global := pricing.NewRates(1.00, 5.00, 0.10).WithCacheCreation(1.25, 2.00, 0)
	nonGlobal := pricing.NewRates(1.10, 5.50, 0.11).WithCacheCreation(1.375, 2.20, 0)

	info := pricing.FlatInfoFromRates(global)
	for _, region := range []string{"us-east5", "europe-west1"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}

// regionalPricing returns a rate card with the global rate as default and
// the non-global rate (global x 1.10) as one override per priced location.
// Claude rates added with it are from the pricing page's region tabs, read
// 2026-09-28, and flat across the =< 200K and > 200K input tiers, except
// Fable 5 and Opus 5 at asia-southeast1, which has no tab for them.
func regionalPricing(global, nonGlobal pricing.Rates, regions ...string) pricing.Info {
	info := pricing.FlatInfoFromRates(global)
	for _, region := range regions {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}

// gemini35FlashPricing returns the Gemini 3.5 Flash rate card. Standard
// rates from the pricing page, read 2026-09-28; its one "Non-global" row
// applies at every served non-global location, the named regions included.
func gemini35FlashPricing() pricing.Info {
	return regionalPricing(
		pricing.NewRates(1.50, 9.00, 0.15),
		pricing.NewRates(1.65, 9.90, 0.165),
		"us", "eu",
		"northamerica-northeast1", "europe-west2", "europe-west3",
		"asia-northeast1", "asia-south1", "asia-southeast1", "australia-southeast1",
	)
}

// gemini35FlashLitePricing returns the Gemini 3.5 Flash-Lite rate card.
// Standard rates from the pricing page, read 2026-09-28.
func gemini35FlashLitePricing() pricing.Info {
	return regionalPricing(
		pricing.NewRates(0.30, 2.50, 0.03),
		pricing.NewRates(0.33, 2.75, 0.033),
		"us", "eu",
	)
}

// claudeOpusPricing returns the rate card Opus 4.5 through Opus 5 share on
// Vertex, priced at the given locations.
func claudeOpusPricing(regions ...string) pricing.Info {
	return regionalPricing(
		pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		regions...,
	)
}

// claudeSonnet45Pricing returns the Sonnet 4.5 rate card, the one Claude
// model Vertex bills higher above 200K input tokens. Rates from the pricing
// page's region tabs, read 2026-09-28.
func claudeSonnet45Pricing() pricing.Info {
	info := pricing.TieredInfo(
		pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
		pricing.Bracket{
			MinContextTokens: 200_001,
			Rates:            pricing.NewRates(6.00, 22.50, 0.60).WithCacheCreation(7.50, 12.00, 0),
		},
	)
	regional := pricing.RateCard{
		Base: pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
		Brackets: []pricing.Bracket{{
			MinContextTokens: 200_001,
			Rates:            pricing.NewRates(6.60, 24.75, 0.66).WithCacheCreation(8.25, 13.20, 0),
		}},
	}

	for _, region := range []string{"us-east5", "europe-west1", "asia-southeast1"} {
		info = info.WithOverride(pricing.Selector{Region: region}, regional)
	}

	return info
}
