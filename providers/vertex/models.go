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
	// AttrPublisher is the Vertex publisher segment ("google",
	// "anthropic").
	AttrPublisher = "publisher"
	// AttrVertexModel is the bare wire model ID, i.e. the offering ID
	// with the vertex. prefix stripped. The offering ID is the pricing
	// key; this is what goes in the request path.
	AttrVertexModel = "vertex_model"
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
	return Catalog().Lookup(catalogID(bareModelID(model)))
}

// Claude reasoning-effort values Vertex accepts, mirroring the
// Anthropic-direct catalog. llm.ReasoningEffort is an open string type
// whose valid vocabulary is provider-owned; these are the source of truth
// for Vertex Claude.
const (
	reasoningEffortLow    llm.ReasoningEffort = "low"
	reasoningEffortMedium llm.ReasoningEffort = "medium"
	reasoningEffortHigh   llm.ReasoningEffort = "high"
	reasoningEffortXHigh  llm.ReasoningEffort = "xhigh"
	reasoningEffortMax    llm.ReasoningEffort = "max"
)

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

// geminiModalities: text, image, audio, video, and PDF inputs; text
// output.
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

// claudeModalities: text, image, and PDF inputs; text output.
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
// Google's published Vertex pricing (Gemini) and Anthropic's list prices
// (Claude); re-reading each one against the live pages is an M1 exit
// condition, because a catalog PR is the last moment a wrong rate costs
// nothing.
//
//   - Google's Vertex Gemini pricing:
//     https://cloud.google.com/vertex-ai/generative-ai/pricing
//   - Anthropic pricing: https://platform.claude.com/docs/en/about-claude/pricing
//
// Capabilities, constraints, and lifecycle mirror the same models in the
// Gemini-API and Anthropic-direct catalogs: the model is the same, only
// the host differs.
func entries() []catalog.Entry {
	return []catalog.Entry{
		{
			ID:           catalogID(ModelGemini36Flash),
			Model:        catalog.ModelGemini36Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65536,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-07-21"),
			},
			// On Vertex, Gemini 3.6 Flash bills at the full SKU rate
			// ($1.50/$7.50, cache $0.15); the introductory discount the
			// Gemini-API provider tracks arrives on Vertex as an
			// account-level credit, not a lower price, and ends
			// 2026-12-31. That credit is post-hoc spend accounting, which
			// the pricing package puts out of scope (pricing/doc.go), so
			// the catalog tracks the SKU rate. A non-global endpoint bills
			// ~10% above global across the us/eu multi-regions, expressed
			// as Region overrides rather than separate model entries.
			Pricing: geminiFlashPricing(),
			Attributes: map[string]string{
				AttrPublisher:   publisherGoogle,
				AttrVertexModel: ModelGemini36Flash,
			},
		},
		{
			ID:           catalogID(ModelClaudeSonnet5),
			Model:        catalog.ModelClaudeSonnet5,
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   1000000, // 1M context window
				MaxOutputTokens:  128000,  // 128K output tokens
				SupportedParams:  []string{"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort"},
			},
			Reasoning: catalog.ReasoningSupport{
				Efforts:  []llm.ReasoningEffort{reasoningEffortLow, reasoningEffortMedium, reasoningEffortHigh, reasoningEffortXHigh, reasoningEffortMax},
				Adaptive: true,
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-06-29"),
			},
			// $2/$10 is Sonnet 5's standard list price, cache reads at the
			// 0.10x multiplier and writes at 1.25x (5m) / 2x (1h). Anthropic
			// is the one publisher absent from Google's Billing Catalog, so
			// Claude rates rest on Anthropic's page. There is no Claude
			// regional premium: the non-global markup is Gemini-only.
			Pricing: pricing.FlatInfoFromRates(
				pricing.NewRates(2.00, 10.00, 0.20).WithCacheCreation(2.50, 4.00, 0),
			),
			Attributes: map[string]string{
				AttrPublisher:   publisherAnthropic,
				AttrVertexModel: ModelClaudeSonnet5,
			},
		},
		{
			ID:           catalogID(ModelClaudeHaiku45),
			Model:        catalog.ModelClaudeHaiku45,
			Capabilities: claudeCaps,
			Modalities:   claudeModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 1.0},
				MaxInputTokens:   200000, // 200K context window
				MaxOutputTokens:  64000,  // 64K output tokens
				SupportedParams:  []string{"temperature", "top_p", "top_k", "max_tokens"},
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2025-10-15"),
			},
			// $1/$5 standard list price, cache reads 0.10x, writes 1.25x
			// (5m) / 2x (1h).
			Pricing: pricing.FlatInfoFromRates(
				pricing.NewRates(1.00, 5.00, 0.10).WithCacheCreation(1.25, 2.00, 0),
			),
			Attributes: map[string]string{
				AttrPublisher:   publisherAnthropic,
				AttrVertexModel: ModelClaudeHaiku45,
			},
		},
	}
}

// geminiFlashPricing builds the Gemini 3.6 Flash rate card: the global
// rate as the default, and the ~10% higher non-global rate as one Region
// override per multi-region that carries the premium.
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
	global := pricing.NewRates(1.50, 7.50, 0.15)
	// The non-global rates ($1.65/$8.25 input/output, $0.165 cache) are the
	// regional Gemini SKUs Google's Vertex pricing page publishes, verified
	// against the live page on 2026-09-07: a uniform ~10% markup over the
	// global rate on input, output, and cache alike.
	nonGlobal := pricing.NewRates(1.65, 8.25, 0.165)

	info := pricing.FlatInfoFromRates(global)
	for _, region := range []string{"us", "eu"} {
		info = info.WithOverride(
			pricing.Selector{Region: region},
			pricing.RateCard{Base: nonGlobal},
		)
	}

	return info
}
