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
	"sync"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Model ID constants for Claude models on Bedrock.
//
// Each logical Claude model is exposed as multiple Bedrock model IDs, one per
// inference profile. The catalog registers each variant as its own entry
// because AWS prices them differently:
//
//   - bare ID (when invokable) and geo profiles (us./eu./au./jp.) use the
//     "Geo and In-region Cross-region Inference" rate.
//   - global. profile uses the "Global Cross-region Inference" rate, which
//     AWS publishes ~10% cheaper than the geo rate. See
//     https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
//     and the two side-by-side Anthropic tables on
//     https://aws.amazon.com/bedrock/pricing/.
//
// This catalog registers every published inference-profile variant and any
// bare ID that Bedrock documents as in-region invokable. For earlier 4.5+
// models, invoking the bare ID via bedrock-runtime returned ValidationException
// "Invocation of model ID … with on-demand throughput isn't supported" in
// empirical checks (2026-04); those bare consts remain building blocks for the
// prefixed variants and are not registered in supportedModels.
const (
	// ModelClaudeFable5 is the bare Bedrock ID for Claude Fable 5
	// (inference-profile-only — invoke via one of the prefixed variants).
	ModelClaudeFable5       = "anthropic.claude-fable-5"
	ModelClaudeFable5Global = "global." + ModelClaudeFable5
	ModelClaudeFable5US     = "us." + ModelClaudeFable5
	ModelClaudeFable5EU     = "eu." + ModelClaudeFable5

	// ModelClaudeSonnet5 is the bare Bedrock ID for Claude Sonnet 5
	// (inference-profile-only — invoke via one of the prefixed variants).
	// Only us. and global. profiles are published so far (verified by
	// invocation 2026-06: us. succeeds, global. is IAM-gated, and eu./au./jp.
	// return ValidationException "model identifier is invalid" even from their
	// home regions). Add geo profiles here once AWS publishes them.
	ModelClaudeSonnet5       = "anthropic.claude-sonnet-5"
	ModelClaudeSonnet5Global = "global." + ModelClaudeSonnet5
	ModelClaudeSonnet5US     = "us." + ModelClaudeSonnet5

	// ModelClaudeSonnet46 is the bare Bedrock ID for Claude Sonnet 4.6
	// (inference-profile-only — invoke via one of the prefixed variants).
	ModelClaudeSonnet46       = "anthropic.claude-sonnet-4-6"
	ModelClaudeSonnet46Global = "global." + ModelClaudeSonnet46
	ModelClaudeSonnet46US     = "us." + ModelClaudeSonnet46
	ModelClaudeSonnet46EU     = "eu." + ModelClaudeSonnet46
	ModelClaudeSonnet46AU     = "au." + ModelClaudeSonnet46

	// ModelClaudeSonnet45 is the bare Bedrock ID for Claude Sonnet 4.5
	// (inference-profile-only — invoke via one of the prefixed variants).
	ModelClaudeSonnet45       = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	ModelClaudeSonnet45Global = "global." + ModelClaudeSonnet45
	ModelClaudeSonnet45US     = "us." + ModelClaudeSonnet45
	ModelClaudeSonnet45EU     = "eu." + ModelClaudeSonnet45
	ModelClaudeSonnet45AU     = "au." + ModelClaudeSonnet45
	ModelClaudeSonnet45JP     = "jp." + ModelClaudeSonnet45

	// ModelClaudeHaiku45 is the bare Bedrock ID for Claude Haiku 4.5
	// (inference-profile-only — invoke via one of the prefixed variants).
	ModelClaudeHaiku45       = "anthropic.claude-haiku-4-5-20251001-v1:0"
	ModelClaudeHaiku45Global = "global." + ModelClaudeHaiku45
	ModelClaudeHaiku45US     = "us." + ModelClaudeHaiku45
	ModelClaudeHaiku45EU     = "eu." + ModelClaudeHaiku45
	ModelClaudeHaiku45AU     = "au." + ModelClaudeHaiku45

	// ModelClaudeOpus5 is the bare building block for Claude Opus 5 profile IDs.
	// bedrock-runtime publishes only global, US, EU, and AU profiles. The bare
	// ID is invokable through bedrock-mantle's Anthropic Messages surface, but
	// this provider's mantle transport implements only the OpenAI-compatible
	// Responses surface.
	ModelClaudeOpus5       = "anthropic.claude-opus-5"
	ModelClaudeOpus5Global = "global." + ModelClaudeOpus5
	ModelClaudeOpus5US     = "us." + ModelClaudeOpus5
	ModelClaudeOpus5EU     = "eu." + ModelClaudeOpus5
	ModelClaudeOpus5AU     = "au." + ModelClaudeOpus5

	// ModelClaudeOpus48 is the bare Bedrock ID for Claude Opus 4.8
	// (inference-profile-only — invoke via one of the prefixed variants).
	ModelClaudeOpus48       = "anthropic.claude-opus-4-8"
	ModelClaudeOpus48Global = "global." + ModelClaudeOpus48
	ModelClaudeOpus48US     = "us." + ModelClaudeOpus48
	ModelClaudeOpus48EU     = "eu." + ModelClaudeOpus48
	ModelClaudeOpus48JP     = "jp." + ModelClaudeOpus48

	// ModelClaudeOpus47 is the bare Bedrock ID for Claude Opus 4.7
	// (inference-profile-only — invoke via one of the prefixed variants).
	ModelClaudeOpus47       = "anthropic.claude-opus-4-7"
	ModelClaudeOpus47Global = "global." + ModelClaudeOpus47
	ModelClaudeOpus47US     = "us." + ModelClaudeOpus47
	ModelClaudeOpus47EU     = "eu." + ModelClaudeOpus47
	ModelClaudeOpus47JP     = "jp." + ModelClaudeOpus47

	// ModelClaudeOpus46 is the bare Bedrock ID for Claude Opus 4.6
	// (inference-profile-only — invoke via one of the prefixed variants).
	ModelClaudeOpus46       = "anthropic.claude-opus-4-6-v1"
	ModelClaudeOpus46Global = "global." + ModelClaudeOpus46
	ModelClaudeOpus46US     = "us." + ModelClaudeOpus46
	ModelClaudeOpus46EU     = "eu." + ModelClaudeOpus46
	ModelClaudeOpus46AU     = "au." + ModelClaudeOpus46

	// ModelClaudeOpus45 is the bare Bedrock ID for Claude Opus 4.5
	// (inference-profile-only — invoke via one of the prefixed variants).
	ModelClaudeOpus45       = "anthropic.claude-opus-4-5-20251101-v1:0"
	ModelClaudeOpus45Global = "global." + ModelClaudeOpus45
	ModelClaudeOpus45US     = "us." + ModelClaudeOpus45
	ModelClaudeOpus45EU     = "eu." + ModelClaudeOpus45
)

// Model ID constants for Amazon Nova models on Bedrock.
//
// Like the Claude 4.5+ models, Nova 2 Lite is inference-profile-only: its
// regional-availability table lists In-Region = No in every region, so the bare
// "amazon.nova-2-lite-v1:0" ID is not on-demand invocable and is registered only
// as a building block for the prefixed variants. AWS publishes us./eu./jp. geo
// profiles and a global. profile (no apac./au.). See
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html
const (
	// ModelNova2Lite is the bare Bedrock ID for Amazon Nova 2 Lite
	// (inference-profile-only — invoke via one of the prefixed variants).
	ModelNova2Lite       = "amazon.nova-2-lite-v1:0"
	ModelNova2LiteGlobal = "global." + ModelNova2Lite
	ModelNova2LiteUS     = "us." + ModelNova2Lite
	ModelNova2LiteEU     = "eu." + ModelNova2Lite
	ModelNova2LiteJP     = "jp." + ModelNova2Lite
)

// Model ID constants for Mistral AI models on Bedrock.
//
// Unlike the Claude and Nova entries — which are inference-profile-only and
// return ValidationException for the bare ID — Mistral Large 3 is an
// on-demand / in-region model invoked by its BARE ID
// "mistral.mistral-large-3-675b-instruct". It has no us./eu./global.
// cross-region inference profile: the AWS Price List publishes rates only under
// the single us-east-1 usagetype (mistral.mistral-large-3-675b-instruct-mantle-*)
// with service tiers (standard/flex/priority/batch), and invoking the
// "us.mistral..." profile returns "ValidationException: the provided model
// identifier is invalid". So we register the bare ID and NewModel resolves it
// as-is (no geo-prefix). See
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
const (
	// ModelMistralLarge3 is the on-demand / in-region Bedrock ID for
	// Mistral Large 3 (invoked bare, with no inference-profile prefix).
	ModelMistralLarge3 = "mistral.mistral-large-3-675b-instruct"
)

// Model ID constants for Google Gemma 4 models on Bedrock.
//
// The Gemma 4 family is served ONLY on the bedrock-mantle endpoint (the
// OpenAI-compatible Responses / Chat Completions API), NOT the standard
// bedrock-runtime Converse API — so every entry sets Mantle: true and NewModel
// routes it through the SigV4-signed Responses transport in mantle.go.
//
// Like Mistral Large 3, these are invoked by their BARE ID with no us./eu./
// global. cross-region inference profile: the model cards list Geo = Not
// supported and Global = Not supported, with only In-Region availability
// (us-east-1, us-east-2, us-west-2, eu-central-1). So the bare ID is registered
// and NewModel resolves it as-is (no geo-prefix). See
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-31b.html
const (
	// ModelGemma431B is Google Gemma 4 31B — a 30.7B dense model with
	// built-in reasoning, native function calling, and text+image+video input
	// (text output), 256K context. mantle-only.
	ModelGemma431B = "google.gemma-4-31b"

	// ModelGemma426BA4B is Google Gemma 4 26B-A4B — a 25.2B-total /
	// 3.8B-active mixture-of-experts model tuned for cost/latency-sensitive
	// workloads, 256K context. mantle-only.
	ModelGemma426BA4B = "google.gemma-4-26b-a4b"

	// ModelGemma4E2B is Google Gemma 4 E2B — the smallest variant (5.1B
	// total / 2.3B effective) for low-latency interactive use, 128K context.
	// mantle-only.
	ModelGemma4E2B = "google.gemma-4-e2b"
)

// ReasoningEffort controls how much work a model spends on reasoning. On
// Claude models it is sent as the "effort" field of output_config together
// with adaptive thinking; on mantle-served models it maps to the OpenAI
// reasoning-effort parameter. It is an alias of [llm.ReasoningEffort] so
// effort values are portable across provider packages; the constants below
// declare the values Bedrock-hosted models accept. Which subset a specific
// model supports is validated against the model catalog in NewModel.
type ReasoningEffort = llm.ReasoningEffort

const (
	// ReasoningEffortLow biases the model toward fast, shallow reasoning.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium is the balanced middle setting.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh biases the model toward deep reasoning.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh spends very high effort (frontier models, Opus 4.7+).
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
	// ReasoningEffortMax removes all effort ceilings (frontier models, Opus 4.6+).
	ReasoningEffortMax ReasoningEffort = "max"
)

// ThinkingSupport describes which provider-native thinking controls a model
// accepts. It is catalog metadata: each supportedModels entry carries the
// shape for its generation, and NewModel validates the requested options
// against it.
type ThinkingSupport struct {
	// ReasoningEfforts are the effort values the model accepts, in ascending
	// order. Empty means the model has no reasoning-effort control.
	ReasoningEfforts []ReasoningEffort

	// Adaptive reports whether the model accepts adaptive thinking
	// (thinking.type=adaptive), where the model decides how long to think.
	Adaptive bool

	// Budget reports whether the model accepts a manual thinking token
	// budget (thinking.type=enabled with budget_tokens).
	Budget bool
}

// Model ID constants for OpenAI GPT-5.6 models on Bedrock.
//
// All three models are served only through the bedrock-mantle Responses API
// and use bare, in-region IDs. AWS does not publish Geo or Global inference
// IDs for them. See
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-openai.html
const (
	// ModelGPT56Sol is OpenAI's flagship GPT-5.6 reasoning model.
	ModelGPT56Sol = "openai.gpt-5.6-sol"

	// ModelGPT56Terra balances reasoning capability and cost.
	ModelGPT56Terra = "openai.gpt-5.6-terra"

	// ModelGPT56Luna is optimized for fast, cost-efficient inference.
	ModelGPT56Luna = "openai.gpt-5.6-luna"
)

// ModelMetadataRequiresProviderDataSharing is set to "true" on discovery
// metadata for Bedrock models that require provider data sharing.
const ModelMetadataRequiresProviderDataSharing = "requires_provider_data_sharing"

// ModelMetadataInferenceGeo is set on inference-profile offerings to the
// geography the profile pins inference to: "us", "eu", "jp", "au", or
// "global" (routes to any commercial region, no residency boundary).
// Bare on-demand IDs carry no value: they run in the calling region.
//
// The key is deliberately provider-agnostic vocabulary (it matches
// Anthropic's inference_geo request parameter) so a future provider whose
// offerings also encode a geography can reuse it, but where geography is
// not part of an offering's identity (Vertex endpoints, OpenAI project
// residency) it stays out of the catalog.
const ModelMetadataInferenceGeo = "inference_geo"

// InferenceProfileRegion maps an AWS region to the Bedrock cross-region
// inference profile geographic prefix.
//
// AWS uses these prefixes:
//   - "us"   for US regions        (us-east-1, us-west-2, …)
//   - "eu"   for European regions  (eu-west-1, eu-central-1, …)
//   - "apac" for Asia Pacific      (ap-southeast-1, ap-northeast-1, …)
//
// See https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference-support.html
func InferenceProfileRegion(region string) string {
	idx := strings.IndexByte(region, '-')
	if idx <= 0 {
		return "us"
	}

	prefix := region[:idx]
	if prefix == "ap" {
		return "apac"
	}

	return prefix
}

// geoProfilePrefixes are the Bedrock cross-region and global inference-profile
// prefixes that can lead a model ID — e.g. the "us" in
// "us.anthropic.claude-sonnet-4-6" or the "global" in
// "global.anthropic.claude-opus-4-6-v1".
var geoProfilePrefixes = map[string]bool{
	"us":     true,
	"eu":     true,
	"apac":   true,
	"jp":     true,
	"au":     true,
	"global": true,
}

// profileRegionResolvers opts model families into exact, model-specific geo
// routing; profileRegionResolverRegions exposes each resolver's complete
// source-region domain so catalog invariants can verify every returned
// profile is registered. Both are single-sourced from the family
// declarations (family.ProfileRegions).
var profileRegionResolvers, profileRegionResolverRegions = buildProfileRegionResolvers(bedrockFamilies)

// hasRegionPrefix reports whether a model ID already begins with a Bedrock
// region/global inference-profile prefix (e.g. "us.anthropic.claude-sonnet-4-6"
// or "global.anthropic.claude-opus-4-6-v1").
//
// It matches the leading dot-separated segment against the known geo-profile
// prefixes rather than counting dots. Dot-counting misclassifies
// vendor-namespaced IDs whose model component itself contains a dot — notably
// "openai.gpt-5.5" (the "5.5" version) — as region-prefixed. That mistake both
// blocks such models from every region in IsModelAllowedFromRegion (no geo
// matches the "openai" pseudo-prefix) and mis-prepends a profile in NewModel.
func hasRegionPrefix(modelID string) bool {
	prefix, _, ok := strings.Cut(modelID, ".")
	if !ok {
		return false
	}

	return geoProfilePrefixes[prefix]
}

func profileRegionResolverFor(modelID string) (func(string) (string, bool), bool) {
	prefix, family, ok := strings.Cut(modelID, ".")
	if ok && geoProfilePrefixes[prefix] {
		modelID = family
	}

	resolver, ok := profileRegionResolvers[modelID]

	return resolver, ok
}

// lookupModel finds an offering by exact model ID. Each inference profile
// variant is registered as its own entry (with its own pricing), so
// callers must pass the full prefixed ID to get the correct rate card.
func lookupModel(modelName string) (catalog.Offering, bool) {
	return Catalog().Lookup(modelName)
}

// Reasoning-control shapes shared by Claude generations on Bedrock. Like
// the capability/constraint shapes below, these are fixed per model
// generation and identical across inference-profile variants. Claude 4.6
// supports both adaptive thinking and deprecated manual budgets; newer
// frontier models are adaptive-only:
// https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
var (
	frontierClaudeThinking = catalog.ReasoningSupport{
		Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
		Adaptive: true,
	}

	claudeOpus46Thinking = catalog.ReasoningSupport{
		Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortMax},
		Adaptive: true,
		Budget:   true,
	}

	claudeSonnet46Thinking = catalog.ReasoningSupport{
		Efforts:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		Adaptive: true,
		Budget:   true,
	}

	claude45Thinking = catalog.ReasoningSupport{Budget: true}
)

// Capability and constraint shapes shared by Claude variants on Bedrock.
// These are genuinely fixed per model generation (a Sonnet 4.5 has the same
// context window in us-east-1 as in eu-west-1), so reusing them across
// variants is safe and avoids duplicating well-known invariants.
//
// Pricing intentionally is NOT shared across models even when two generations
// happen to publish the same numbers today — Anthropic's intermediate
// releases (e.g. Opus 4.1 vs 4.5) have priced differently in the past, so
// each model's rates are spelled out next to its catalog entry.
var (
	// JSONMode is false throughout this file for Converse-path models:
	// outputConfig.textFormat accepts only a JSON schema, so there is no
	// schemaless JSON mode to advertise.
	claudeStandardCaps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         false,
		StructuredOutput: true,
		Vision:           true,
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        true,
	}

	// claudeModalities is shared by every Claude family on Bedrock: text,
	// image and PDF document input, text output.
	claudeModalities = catalog.Modalities{
		Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityDocument},
		Output: []catalog.Modality{catalog.ModalityText},
	}

	claudeContext1MConstraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 1.0},
		MaxInputTokens:   1000000,
		MaxOutputTokens:  128000,
		SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
	}

	claudeNoSampling1MConstraints = llm.ModelConstraints{
		MaxInputTokens:  1000000,
		MaxOutputTokens: 128000,
		SupportedParams: []string{"max_tokens", "stop"},
	}

	claudeContext200kConstraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 1.0},
		MaxInputTokens:   200000,
		MaxOutputTokens:  64000,
		SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
	}

	// Capability/constraint shapes for Amazon Nova 2 Lite: multimodal
	// text + image + video + document input, text output, Bedrock
	// constrained-decoding structured outputs, and extended thinking.
	//
	// top_k is omitted from SupportedParams because the Converse request
	// mapper only emits temperature/top_p/max_tokens/stop.
	nova2LiteCaps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         false,
		StructuredOutput: true,
		Vision:           true,
		Audio:            false,
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        true,
	}

	nova2LiteModalities = catalog.Modalities{
		Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityVideo, catalog.ModalityDocument},
		Output: []catalog.Modality{catalog.ModalityText},
	}

	nova2LiteConstraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 1.0},
		MaxInputTokens:   1000000,
		MaxOutputTokens:  64000,
		SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
	}

	// Capability/constraint shapes for Mistral Large 3. Vision is false:
	// Mistral Large 3 is text-only on Bedrock (Pixtral Large is Mistral's
	// multimodal entry). Reasoning is false — it is not a reasoning model
	// (Magistral is Mistral's reasoning line), and the only reasoning lever in
	// this SDK (WithThinking) emits an Anthropic-shaped thinking document that
	// non-Anthropic Bedrock models reject.
	//
	// Context window: the Bedrock-hosted variant lists a 128K (131072)
	// input window and an 8,192 max-output cap — smaller than Mistral's
	// native API, so re-confirm the exact output cap when validating live
	// (it may be raised). top_k is omitted from SupportedParams because the Converse
	// request mapper only emits temperature/top_p/max_tokens/stop.
	mistralLarge3Caps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         false,
		StructuredOutput: true,
		Vision:           false,
		Audio:            false,
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        false,
	}

	mistralLarge3Constraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 1.0},
		MaxInputTokens:   128000,
		MaxOutputTokens:  8192,
		SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
	}

	// Capability/constraint shapes for the Google Gemma 4 family on the
	// bedrock-mantle endpoint. All variants share the same capability surface:
	// streaming, native function calling (Tools), built-in Reasoning, image
	// and video input, and text output.
	//
	// The context windows differ by variant (256K for 31B / 26B-A4B, 128K for
	// E2B), so each gets its own constraints var. AWS publishes no separate
	// max-output cap for Gemma on Bedrock — output is bounded only by the
	// context window (other hosts report the full context as the output
	// ceiling) — so MaxOutputTokens mirrors MaxInputTokens rather than a guessed
	// smaller cap that would make WithMaxTokens spuriously reject large but
	// valid requests. TemperatureRange is the OpenAI API's 0..2 (the mantle
	// Responses endpoint honors OpenAI sampling semantics). SupportedParams
	// lists only temperature and max_tokens: those are the only sampling
	// controls the request mapper serializes, so advertising top_p / stop /
	// penalties would let callers set knobs that silently do nothing.
	gemma4Caps = llm.ModelCapabilities{
		Streaming:     true,
		Tools:         true,
		Vision:        true,
		Audio:         false,
		MultiTurn:     true,
		SystemPrompts: true,
		Reasoning:     true,
	}

	gemma4Modalities = catalog.Modalities{
		Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage, catalog.ModalityVideo},
		Output: []catalog.Modality{catalog.ModalityText},
	}

	gemma4Context256kConstraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 2.0},
		MaxInputTokens:   256000,
		MaxOutputTokens:  256000,
		SupportedParams:  []string{"temperature", "max_tokens"},
	}

	gemma4Context128kConstraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 2.0},
		MaxInputTokens:   128000,
		MaxOutputTokens:  128000,
		SupportedParams:  []string{"temperature", "max_tokens"},
	}

	// GPT-5.6 is available through the same bedrock-mantle Responses transport
	// as Gemma 4. AWS advertises a 272K context window for the Bedrock-hosted
	// variants, rather than the larger first-party OpenAI window.
	// SupportedParams contains only options the mantle adapter serializes.
	gpt56Caps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true,
		StructuredOutput: true,
		Vision:           true,
		Audio:            false,
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        true,
	}

	gpt56Modalities = catalog.Modalities{
		Input:  []catalog.Modality{catalog.ModalityText, catalog.ModalityImage},
		Output: []catalog.Modality{catalog.ModalityText},
	}

	gpt56Constraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 2.0},
		MaxInputTokens:   272000,
		MaxOutputTokens:  128000,
		SupportedParams:  []string{"temperature", "max_tokens"},
	}
)

// gpt56Reasoning: the Bedrock-hosted GPT-5.6 models expose reasoning but
// no effort control through the mantle adapter today.
var gpt56Reasoning = catalog.ReasoningSupport{}

var catalogOnce = sync.OnceValue(func() *catalog.Catalog {
	entries, _ := expandFamilies(bedrockFamilies)
	return catalog.MustNew("aws.bedrock", entries)
})

// mantleModelIDs is the set of bare model IDs served exclusively on the
// bedrock-mantle endpoint, built from the family declarations.
var mantleModelIDs = func() map[string]bool {
	_, mantle := expandFamilies(bedrockFamilies)
	return mantle
}()

// Catalog returns the validated Bedrock model catalog: every inference-
// profile variant with its capabilities, constraints, reasoning controls,
// pricing, and lifecycle. The catalog is immutable and shared; all reads
// return deep copies.
func Catalog() *catalog.Catalog {
	return catalogOnce()
}

// bedrockFamilies is the per-family Bedrock catalog. Each family expands
// to one entry per published inference profile; every rate is authored in
// place, never derived.
//
// Pricing rule (per AWS https://aws.amazon.com/bedrock/pricing/, 2026-08):
//   - bare ID (when invokable) and geo profiles (us./eu./au./jp.) use the
//     "Geo and In-region Cross-region Inference" rate.
//   - global. profiles use the "Global Cross-region Inference" rate, which
//     is exactly 10% cheaper than the geo rate for the same model. That
//     relationship is pinned by TestGeoGlobalRatio as a tripwire.
//
// Lifecycle: Bedrock sets its own availability and retirement schedules
// (surfaced by ListFoundationModels' modelLifecycle); no catalogued model
// has an announced deprecation, and per-profile availability dates are not
// published, so Life stays empty rather than inventing dates.
var bedrockFamilies = []family{
	{
		// Claude Fable 5 — inference-profile-only, no bare entry. Geo
		// profiles cover us and eu (jp/au are not published).
		BareID:       ModelClaudeFable5,
		Model:        catalog.ModelClaudeFable5,
		DisplayName:  "Claude Fable 5",
		Profiles:     []string{"global", "us", "eu"},
		DataSharing:  true,
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeNoSampling1MConstraints,
		Reasoning:    frontierClaudeThinking,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(10.00, 50.00, 1.00).WithCacheCreation(12.50, 20.00, 0)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(11.00, 55.00, 1.10).WithCacheCreation(13.75, 22.00, 0)},
	},
	{
		// Claude Opus 5 — inference-profile-only on bedrock-runtime. AWS
		// publishes global, US, EU, and AU profiles:
		// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html
		BareID:       ModelClaudeOpus5,
		Model:        catalog.ModelClaudeOpus5,
		DisplayName:  "Claude Opus 5",
		Profiles:     []string{"global", "us", "eu", "au"},
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeNoSampling1MConstraints,
		Reasoning:    frontierClaudeThinking,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0)},
		// Opus 5 opts into exact geo routing (AU is Sydney/Melbourne only).
		ProfileRegions: claudeOpus5ProfileRegions,
	},
	{
		// Claude Opus 4.8 — inference-profile-only. Geo profiles cover
		// us, eu, jp (au is not published).
		BareID:       ModelClaudeOpus48,
		Model:        catalog.ModelClaudeOpus48,
		DisplayName:  "Claude Opus 4.8",
		Profiles:     []string{"global", "us", "eu", "jp"},
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeContext1MConstraints,
		Reasoning:    frontierClaudeThinking,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0)},
	},
	{
		// Claude Opus 4.7 — inference-profile-only. Geo profiles cover
		// us, eu, jp (au is not published).
		BareID:       ModelClaudeOpus47,
		Model:        catalog.ModelClaudeOpus47,
		DisplayName:  "Claude Opus 4.7",
		Profiles:     []string{"global", "us", "eu", "jp"},
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeContext1MConstraints,
		Reasoning:    frontierClaudeThinking,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0)},
	},
	{
		// Claude Opus 4.6 — inference-profile-only.
		BareID:       ModelClaudeOpus46,
		Model:        catalog.ModelClaudeOpus46,
		DisplayName:  "Claude Opus 4.6",
		Profiles:     []string{"global", "us", "eu", "au"},
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeContext1MConstraints,
		Reasoning:    claudeOpus46Thinking,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0)},
	},
	{
		// Claude Opus 4.5 — inference-profile-only.
		BareID:       ModelClaudeOpus45,
		Model:        catalog.ModelClaudeOpus45,
		DisplayName:  "Claude Opus 4.5",
		Profiles:     []string{"global", "us", "eu"},
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeContext200kConstraints,
		Reasoning:    claude45Thinking,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0)},
	},
	{
		// Claude Sonnet 5 — inference-profile-only; global and us are
		// published so far.
		BareID:       ModelClaudeSonnet5,
		Model:        catalog.ModelClaudeSonnet5,
		DisplayName:  "Claude Sonnet 5",
		Profiles:     []string{"global", "us"},
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeContext1MConstraints,
		Reasoning:    frontierClaudeThinking,
		// $2/$10 is Sonnet 5's standard price: the increase to $3/$15 once
		// scheduled for 2026-09-01 was cancelled.
		GlobalRates: &pricing.RateCard{Base: pricing.NewRates(2.00, 10.00, 0.20).WithCacheCreation(2.50, 4.00, 0)},
		Rates:       pricing.RateCard{Base: pricing.NewRates(2.20, 11.00, 0.22).WithCacheCreation(2.75, 4.40, 0)},
	},
	{
		// Claude Sonnet 4.6 — inference-profile-only. 1M context on
		// Bedrock as well as the first-party API.
		BareID:       ModelClaudeSonnet46,
		Model:        catalog.ModelClaudeSonnet46,
		DisplayName:  "Claude Sonnet 4.6",
		Profiles:     []string{"global", "us", "eu", "au"},
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeContext1MConstraints,
		Reasoning:    claudeSonnet46Thinking,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0)},
	},
	{
		// Claude Sonnet 4.5 — inference-profile-only; the widest geo
		// coverage of the Claude 4.x line.
		BareID:       ModelClaudeSonnet45,
		Model:        catalog.ModelClaudeSonnet45,
		DisplayName:  "Claude Sonnet 4.5",
		Profiles:     []string{"global", "us", "eu", "au", "jp"},
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeContext200kConstraints,
		Reasoning:    claude45Thinking,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0)},
	},
	{
		// Claude Haiku 4.5 — inference-profile-only.
		BareID:       ModelClaudeHaiku45,
		Model:        catalog.ModelClaudeHaiku45,
		DisplayName:  "Claude Haiku 4.5",
		Profiles:     []string{"global", "us", "eu", "au"},
		Capabilities: claudeStandardCaps,
		Modalities:   claudeModalities,
		Constraints:  claudeContext200kConstraints,
		Reasoning:    claude45Thinking,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(1.00, 5.00, 0.10).WithCacheCreation(1.25, 2.00, 0)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(1.10, 5.50, 0.11).WithCacheCreation(1.375, 2.20, 0)},
	},
	{
		// Amazon Nova 2 Lite. Cache WRITE is free (the AWS price list
		// carries an explicit $0.00 cache-write usagetype), so no
		// WithCacheCreation is set — cache-write buckets stay zero, which
		// the pricing shape tests explicitly allow for free-cache-write
		// models.
		BareID:       ModelNova2Lite,
		Model:        catalog.ModelNova2Lite,
		DisplayName:  "Amazon Nova 2 Lite",
		Profiles:     []string{"global", "us", "eu", "jp"},
		Capabilities: nova2LiteCaps,
		Modalities:   nova2LiteModalities,
		Constraints:  nova2LiteConstraints,
		GlobalRates:  &pricing.RateCard{Base: pricing.NewRates(0.30, 2.50, 0.075)},
		Rates:        pricing.RateCard{Base: pricing.NewRates(0.33, 2.75, 0.0825)},
	},
	{
		// Mistral Large 3 — on-demand / in-region, BARE ID only (no
		// inference profile exists). Does NOT follow the geo/global split:
		// AWS prices it by service tier under a single usagetype; the
		// on-demand standard tier is registered here. Rates verified
		// against the AWS Price List bulk API (mistral...-mantle-
		// {input,output}-tokens-standard). Prompt caching is not billed
		// (no cache usagetype published), so cache rates stay zero.
		BareID:        ModelMistralLarge3,
		Model:         catalog.ModelMistralLarge3,
		DisplayName:   "Mistral Large 3",
		BareInvokable: true,
		Capabilities:  mistralLarge3Caps,
		Constraints:   mistralLarge3Constraints,
		Rates:         pricing.RateCard{Base: pricing.NewRates(0.50, 1.50, 0)},
	},
	{
		// OpenAI GPT-5.6 family — mantle-only, bare in-region IDs via the
		// Responses API; AWS publishes no Geo or Global inference IDs.
		// In-region STANDARD-tier rates from the AWS pricing page (OpenAI
		// section, 2026-08). Cache writes have a 30-minute TTL and the
		// Responses usage payload reports an aggregate cache_write_tokens
		// count, so the write price sits in the unknown-TTL bucket.
		BareID:        ModelGPT56Sol,
		Model:         catalog.ModelGPT5_6Sol,
		DisplayName:   "OpenAI GPT-5.6 Sol",
		BareInvokable: true,
		Mantle:        true,
		Capabilities:  gpt56Caps,
		Modalities:    gpt56Modalities,
		Constraints:   gpt56Constraints,
		Reasoning:     gpt56Reasoning,
		Rates:         pricing.RateCard{Base: pricing.NewRates(5.50, 33.00, 0.55).WithCacheCreation(0, 0, 6.875)},
	},
	{
		BareID:        ModelGPT56Terra,
		Model:         catalog.ModelGPT5_6Terra,
		DisplayName:   "OpenAI GPT-5.6 Terra",
		BareInvokable: true,
		Mantle:        true,
		Capabilities:  gpt56Caps,
		Modalities:    gpt56Modalities,
		Constraints:   gpt56Constraints,
		Reasoning:     gpt56Reasoning,
		Rates:         pricing.RateCard{Base: pricing.NewRates(2.75, 16.50, 0.275).WithCacheCreation(0, 0, 3.4375)},
	},
	{
		BareID:        ModelGPT56Luna,
		Model:         catalog.ModelGPT5_6Luna,
		DisplayName:   "OpenAI GPT-5.6 Luna",
		BareInvokable: true,
		Mantle:        true,
		Capabilities:  gpt56Caps,
		Modalities:    gpt56Modalities,
		Constraints:   gpt56Constraints,
		Reasoning:     gpt56Reasoning,
		Rates:         pricing.RateCard{Base: pricing.NewRates(1.10, 6.60, 0.11).WithCacheCreation(0, 0, 1.375)},
	},
	{
		// Google Gemma 4 family — mantle-only, bare IDs via the
		// bedrock-mantle Responses endpoint (model cards list Geo/Global =
		// Not supported). On-demand STANDARD-tier US rates from the AWS
		// pricing page (Google section, 2026-06); the ~15-20% higher
		// eu-central-1 rate is not separately modeled (revisit with a
		// pricing region Override if EU-accurate billing is required). The
		// mantle endpoint bills only input/output for Gemma — no cache
		// usagetype is published — so cache rates stay zero.
		BareID:        ModelGemma431B,
		Model:         catalog.ModelGemma431B,
		DisplayName:   "Google Gemma 4 31B",
		BareInvokable: true,
		Mantle:        true,
		Capabilities:  gemma4Caps,
		Modalities:    gemma4Modalities,
		Constraints:   gemma4Context256kConstraints,
		Rates:         pricing.RateCard{Base: pricing.NewRates(0.14, 0.40, 0)},
	},
	{
		BareID:        ModelGemma426BA4B,
		Model:         catalog.ModelGemma426BA4B,
		DisplayName:   "Google Gemma 4 26B-A4B",
		BareInvokable: true,
		Mantle:        true,
		Capabilities:  gemma4Caps,
		Modalities:    gemma4Modalities,
		Constraints:   gemma4Context256kConstraints,
		Rates:         pricing.RateCard{Base: pricing.NewRates(0.13, 0.40, 0)},
	},
	{
		BareID:        ModelGemma4E2B,
		Model:         catalog.ModelGemma4E2B,
		DisplayName:   "Google Gemma 4 E2B",
		BareInvokable: true,
		Mantle:        true,
		Capabilities:  gemma4Caps,
		Modalities:    gemma4Modalities,
		Constraints:   gemma4Context128kConstraints,
		Rates:         pricing.RateCard{Base: pricing.NewRates(0.04, 0.08, 0)},
	},
}
