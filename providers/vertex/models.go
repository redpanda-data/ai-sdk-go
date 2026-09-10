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
// data for Google Vertex AI, where Google Cloud hosts first-party Gemini
// alongside partner models such as Anthropic's Claude. One GCP project
// reaches all of them, billed on that project's own account.
//
// This package is the catalog half of the provider (RFC-0014 milestone
// M1): the day-one Gemini + Claude catalog, ModelPricing through each
// offering's [catalog.Entry].Pricing, and the location-availability
// helper in locations.go. The request/response transport (an llm.Model
// that builds Vertex requests) lands with the managed-agent milestone and
// is intentionally not part of this package yet.
//
// Catalog keys are namespaced vertex.<model>. On Vertex a model keeps its
// publisher's bare ID, so "claude-sonnet-5" is byte-identical to the ID
// the Anthropic-direct catalog already uses; a shared pricing catalog
// rejects that collision. The bare wire model and its publisher travel in
// each entry's Attributes instead of as an alias, because an alias would
// re-introduce the same collision in a merged catalog.
package vertex

import (
	"strings"
	"sync"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// providerName is the provider identifier used in offerings and
// telemetry. It matches the catalog key prefix's stem and mirrors
// Bedrock's "aws.bedrock": the cloud, then the surface.
const providerName = "gcp.vertex"

// catalogKeyPrefix namespaces every Vertex offering ID. See the package
// doc for why the prefix is required rather than optional.
const catalogKeyPrefix = "vertex."

// Bare Vertex model IDs - exactly the model segment of a Vertex resource
// path, publishers/{publisher}/models/{model}. The day-one catalog is
// Gemini + Claude, enumerated in RFC-0014's Model catalog and pricing
// section (the F5 scope call). The open-weight families on the
// OpenAI-compatible route are deferred until a customer asks.
const (
	ModelGemini36Flash = "gemini-3.6-flash"
	ModelClaudeSonnet5 = "claude-sonnet-5"
	ModelClaudeHaiku45 = "claude-haiku-4-5"
)

// Offering IDs are the namespaced catalog keys, the vertex. prefix plus the
// bare model. These are what catalog lookups and the pricing map are keyed
// on, so a caller uses Offering* for Catalog().Lookup and the pricing map,
// and the bare Model* for the request path. Composing the key from the same
// prefix keeps one catalog key per offering, so this adds no alias and does
// not re-open the collision the package doc argues against.
const (
	OfferingGemini36Flash = catalogKeyPrefix + ModelGemini36Flash
	OfferingClaudeSonnet5 = catalogKeyPrefix + ModelClaudeSonnet5
	OfferingClaudeHaiku45 = catalogKeyPrefix + ModelClaudeHaiku45
)

// Publishers own the model on Vertex and name the segment before the
// model in a Vertex resource path. Stored per offering in Attributes so
// the runtime provider can build the path without re-deriving it.
const (
	publisherGoogle    = "google"
	publisherAnthropic = "anthropic"
)

// Attribute keys carried on every Vertex offering. Keys are snake_case
// and values are strings so the committed snapshot stays stable.
const (
	// ModelMetadataPublisher is the Vertex publisher segment ("google",
	// "anthropic").
	ModelMetadataPublisher = "publisher"
	// ModelMetadataVertexModel is the bare wire model ID, i.e. the offering
	// ID with the vertex. prefix stripped. The offering ID is the pricing
	// key; this is what goes in the request path.
	ModelMetadataVertexModel = "vertex_model"
)

// catalogID returns the namespaced offering ID for a bare Vertex model.
func catalogID(bareModel string) string {
	return catalogKeyPrefix + bareModel
}

// bareModelID strips the vertex. catalog-key prefix when present, so a
// caller may pass either a bare publisher model ID ("claude-sonnet-5")
// or a namespaced offering ID ("vertex.claude-sonnet-5") and reach the
// same entry. A string without the prefix is returned unchanged.
func bareModelID(model string) string {
	return strings.TrimPrefix(model, catalogKeyPrefix)
}

// OfferingForModel returns the Vertex offering for a bare publisher model
// ID, adding the vertex. catalog-key prefix and looking it up. It is the
// bridge for callers that hold a bare model name rather than a namespaced
// offering ID, keeping the prefix an internal catalog detail. A model ID
// that already carries the prefix is accepted as-is. ok is false for a
// model the catalog does not offer.
func OfferingForModel(model string) (catalog.Offering, bool) {
	return Catalog().Resolve(catalogID(bareModelID(model)))
}

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

var claudeModalities = catalog.Modalities{
	Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityDocument},
	Output: []catalog.Modality{catalog.ModalityText},
}

var catalogOnce = sync.OnceValue(func() *catalog.Catalog {
	return catalog.MustNew(providerName, entries())
})

// Catalog returns the validated Vertex model catalog: every offering with
// its capabilities, constraints, modalities, reasoning controls, pricing,
// and lifecycle. The catalog is immutable and shared; all reads return
// deep copies.
func Catalog() *catalog.Catalog {
	return catalogOnce()
}

// entries returns the authored Vertex catalog.
//
// Rates are USD-per-million-tokens, from Google's single pricing page for
// the platform (Vertex AI was renamed the Gemini Enterprise Agent Platform,
// so older /vertex-ai/ links redirect here):
//
//	https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
//
// Re-reading each rate against the live page is an M1 exit condition. Claude
// rates come from that page, not Anthropic's list prices: Vertex sets its own
// Claude rates (a ~10% non-global premium included), and Anthropic has no SKUs
// in Google's Billing Catalog, so this page is authoritative.
//
// Capabilities and constraints mirror the same models in the Gemini-API and
// Anthropic-direct catalogs; only the host differs. Lifecycle does not: each
// entry's Available is when Vertex began serving the model (its model page's
// Release date, cited at the entry), and Retires stays unset because Google
// publishes only "not sooner than" floors, not shutdown dates.
func entries() []catalog.Entry {
	return []catalog.Entry{
		{
			ID:           OfferingGemini36Flash,
			Model:        catalog.ModelGemini36Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			Reasoning: catalog.ReasoningSupport{Efforts: geminiReasoningEfforts},
			// Vertex began serving Gemini 3.6 Flash at its GA, release date
			// 2026-07-21 on the model page (docs.cloud.google.com/
			// gemini-enterprise-agent-platform/models/gemini/3-6-flash, read
			// 2026-09-10). No retirement published, so Retires stays unset.
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-07-21")},
			Pricing: geminiFlashPricing(),
			Attributes: map[string]string{
				ModelMetadataPublisher:   publisherGoogle,
				ModelMetadataVertexModel: ModelGemini36Flash,
			},
		},
		{
			ID:           OfferingClaudeSonnet5,
			Model:        catalog.ModelClaudeSonnet5,
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   1000000,
				MaxOutputTokens:  128000,
				SupportedParams:  []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []llm.ReasoningEffort{reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh, reasoningEffortXHigh, reasoningEffortMax},
				Adaptive: true,
			},
			// Vertex began serving Claude Sonnet 5 at its GA, release date
			// 2026-06-30 on the model page; the retirement floor ("not sooner
			// than 2026-12-24") is a lower bound, not a shutdown date, so
			// Retires stays unset (docs.cloud.google.com/
			// gemini-enterprise-agent-platform/models/partner-models/claude/sonnet-5,
			// read 2026-09-10).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-06-30")},
			Pricing: claudeSonnet5Pricing(),
			Attributes: map[string]string{
				ModelMetadataPublisher:   publisherAnthropic,
				ModelMetadataVertexModel: ModelClaudeSonnet5,
			},
		},
		{
			ID:           OfferingClaudeHaiku45,
			Model:        catalog.ModelClaudeHaiku45,
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   200000,
				MaxOutputTokens:  64000,
				SupportedParams:  []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []llm.ReasoningEffort{reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh, reasoningEffortMax},
				Adaptive: true,
			},
			// Claude Haiku 4.5 GA, release date 2025-10-15 on the model page;
			// the retirement floor ("not sooner than 2026-10-15") is a lower
			// bound, not a shutdown date, so Retires stays unset
			// (docs.cloud.google.com/gemini-enterprise-agent-platform/models/
			// partner-models/claude/haiku-4-5, read 2026-09-10).
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2025-10-15")},
			Pricing: claudeHaiku45Pricing(),
			Attributes: map[string]string{
				ModelMetadataPublisher:   publisherAnthropic,
				ModelMetadataVertexModel: ModelClaudeHaiku45,
			},
		},
	}
}

// geminiFlashPricing returns the Gemini 3.6 Flash rate card: the global
// rate as default, and the 10% higher non-global rate as one override per
// priced multi-region.
func geminiFlashPricing() pricing.Info {
	// Introductory per-token rate through 2026-12-31 (standard from
	// 2027-01-01: $1.50/$7.50 global, $1.65/$8.25 non-global), tracked like
	// the Gemini-API provider (providers/google/models.go). Not the separate
	// 50% Provisioned Throughput credit the page lists, which is post-hoc
	// spend accounting the pricing package excludes (pricing/doc.go).
	global := pricing.NewRates(0.75, 3.75, 0.075)
	// Non-global = global x 1.10 on input, output, and cache alike, from
	// Google's pricing page, verified live 2026-09-08.
	nonGlobal := pricing.NewRates(0.825, 4.125, 0.0825)

	info := pricing.FlatInfoFromRates(global)
	// Priced regions are a literal, not derived from availability
	// (locations.go): a region may be served at the global rate, so served
	// does not imply priced. TestGeminiRegionalOverride guards the reverse.
	// The override carries the non-global rate because an empty-Region
	// selector is a wildcard that would also match global.
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
	// Sonnet's served non-global regions (locations.go); asia-southeast1
	// carries the same premium, not a special APAC rate.
	// TestClaudeRegionalOverride guards that every priced region is served.
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
	// Served named regions (locations.go); asia-east1 carries the same
	// premium, not a special rate.
	for _, region := range []string{"us-east5", "europe-west1", "asia-east1"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}
