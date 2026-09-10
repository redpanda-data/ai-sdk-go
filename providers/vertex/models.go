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
// Rates are USD-per-million-tokens. Every rate here was transcribed from
// Google's published Vertex pricing; re-reading each one against the live
// pages is an M1 exit condition, because a catalog PR is the last moment a
// wrong rate costs nothing.
//
// Every rate, Gemini and Claude alike, comes from Google's one pricing
// page for the platform (Google renamed Vertex AI to the Gemini Enterprise
// Agent Platform, so older /vertex-ai/ links redirect here):
//
//	https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
//
// Claude rates come from that page, not Anthropic's list prices: Vertex
// sets its own Claude rates (including a ~10% non-global premium
// Anthropic-direct does not have), and Anthropic is the one publisher with
// no SKUs in Google's Billing Catalog, so this page, with its per-region
// tabs, is the authoritative source.
//
// Capabilities and constraints mirror the same models in the Gemini-API
// and Anthropic-direct catalogs: the model is the same, only the host
// differs. Lifecycle does not mirror them - Available is when Vertex began
// serving the model, a partner host's own schedule. The Agent Platform's
// per-model pages publish that as the version's Release date, so each entry
// sets Available from its own model page (cited at the entry) and leaves
// Retires unset, because Google publishes only "not sooner than" retirement
// floors, which are lower bounds rather than shutdown dates.
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

// geminiFlashPricing builds the Gemini 3.6 Flash rate card: the global
// rate as the default, and the ~10% higher non-global rate as one Region
// override per multi-region that carries the premium.
//
// The default is Gemini 3.6 Flash's introductory per-token rate, tracked
// like the Gemini-API provider (providers/google/models.go). It is a real
// per-token price, not the separate 50% Provisioned Throughput credit the
// page also lists; that credit is post-hoc spend accounting the pricing
// package excludes (pricing/doc.go).
//
// The priced regions are their own literal, not derived from the
// availability matrix (locations.go). Pricing and availability are
// separate facts: a region can be served at the plain global rate, so a
// served location is not automatically a priced one. The direction that
// must hold is the reverse - a region priced here must be one the model
// is served at - and TestGeminiRegionalOverride guards exactly that, so
// the two never contradict without coupling the definitions.
//
// The override carries the non-global rate (default = global) because an
// empty-Region override cannot mean "every non-global region" - an empty
// selector field is a wildcard that would also match global.
func geminiFlashPricing() pricing.Info {
	global := pricing.NewRates(0.75, 3.75, 0.075)
	// The non-global rates ($0.825/$4.125 input/output, $0.0825 cache) are
	// the regional Gemini rates Google's pricing page publishes, verified
	// against the live page on 2026-09-08: a uniform ~10% markup over the
	// global rate on input, output, and cache alike. These are the
	// introductory rates in effect through 2026-12-31; the standard rates
	// from 2027-01-01 are $1.50/$7.50 global, $1.65/$8.25 non-global.
	nonGlobal := pricing.NewRates(0.825, 4.125, 0.0825)

	info := pricing.FlatInfoFromRates(global)
	for _, region := range []string{"us", "eu"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}

// claudeSonnet5Pricing builds the Sonnet 5 rate card: the global rate as
// the default, and the ~10% higher non-global rate as one Region override
// per served non-global region.
//
// Claude is not flat. Google's Agent Platform pricing page groups Sonnet 5
// under "Models with regional pricing" and publishes every non-global rate
// at exactly global x 1.10 - input, output, cache write, and cache read
// alike. The us and eu rates below read from the page's region tabs on
// 2026-09-08.
//
// asia-southeast1 is a served non-global region (locations.go) with its
// own tab on the pricing page. That tab carries the standard non-global
// rate - global x 1.10, the same premium as the us and eu multi-regions,
// flat across the page's =< 200K and > 200K input tiers - confirmed on the
// tab on 2026-09-08. It is not a special APAC rate, so a customer calling
// there is billed the same non-global premium as one calling us or eu.
//
// The override regions are Sonnet's served non-global regions (locations.go),
// so TestClaudeRegionalOverride can guard that every priced region is served,
// the same invariant Gemini carries. Anthropic is absent from Google's
// Billing Catalog, so this page is the authoritative source, not the SKU
// API.
func claudeSonnet5Pricing() pricing.Info {
	global := pricing.NewRates(2.00, 10.00, 0.20).WithCacheCreation(2.50, 4.00, 0)
	nonGlobal := pricing.NewRates(2.20, 11.00, 0.22).WithCacheCreation(2.75, 4.40, 0)

	info := pricing.FlatInfoFromRates(global)
	for _, region := range []string{"us", "eu", "asia-southeast1"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}

// claudeHaiku45Pricing builds the Haiku 4.5 rate card: the global rate as
// the default, and the ~10% non-global premium as one Region override per
// served named region.
//
// Same shape as Sonnet. The us-east5 and europe-west1 rates below read from
// the Agent Platform pricing page's region tabs on 2026-09-08.
//
// asia-east1 is a served named region (locations.go) with its own tab on
// the pricing page. Haiku 4.5 is a line item on that tab at input $1.10,
// output $5.50, cache hit $0.11, and cache write $1.375 (5m) / $2.20 (1h) -
// exactly global x 1.10 and flat across the page's =< 200K and > 200K input
// tiers, matching the nonGlobal rate below, read from the tab on 2026-09-08.
// It is the same non-global premium the us-east5 and europe-west1 tabs carry.
func claudeHaiku45Pricing() pricing.Info {
	global := pricing.NewRates(1.00, 5.00, 0.10).WithCacheCreation(1.25, 2.00, 0)
	nonGlobal := pricing.NewRates(1.10, 5.50, 0.11).WithCacheCreation(1.375, 2.20, 0)

	info := pricing.FlatInfoFromRates(global)
	for _, region := range []string{"us-east5", "europe-west1", "asia-east1"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}
