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
// This catalog registers inference-profile variants instead of bare Claude
// IDs so each entry carries the correct routing and pricing metadata. For
// earlier 4.5+ models, invoking the bare ID via bedrock-runtime returned
// ValidationException "Invocation of model ID … with on-demand throughput
// isn't supported" in empirical checks (2026-04); the bare consts below exist
// only as building blocks for the prefixed variants and are NOT registered in
// supportedModels.
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

// Effort controls reasoning depth for Bedrock models with adaptive thinking.
type Effort string

const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
	EffortXHigh  Effort = "xhigh"
	EffortMax    Effort = "max"
)

// ModelDefinition defines a model with its capabilities and constraints.
type ModelDefinition struct {
	Name                        string // Real Bedrock model ID (e.g. "us.anthropic.claude-sonnet-4-6")
	Label                       string
	Capabilities                llm.ModelCapabilities
	Constraints                 llm.ModelConstraints
	Pricing                     pricing.Info
	RequiresProviderDataSharing bool

	// Mantle marks a model that is served ONLY on the bedrock-mantle
	// endpoint (the OpenAI-compatible Responses / Chat Completions API at
	// bedrock-mantle.{region}.api.aws), not the standard bedrock-runtime
	// Converse API. NewModel routes these models through a SigV4-signed
	// OpenAI Responses transport (see mantle.go) instead of Converse. The
	// Google Gemma 4 family and OpenAI's gpt-5.x frontier models on Bedrock
	// are mantle-only. See
	// https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
	Mantle bool
}

// ModelMetadataRequiresProviderDataSharing is set to "true" on discovery
// metadata for Bedrock models that require provider data sharing.
const ModelMetadataRequiresProviderDataSharing = "requires_provider_data_sharing"

func (d ModelDefinition) discoveryMetadata() map[string]string {
	if !d.RequiresProviderDataSharing {
		return nil
	}

	return map[string]string{
		ModelMetadataRequiresProviderDataSharing: "true",
	}
}

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

// lookupModel finds a ModelDefinition by exact model ID. Each inference
// profile variant is registered as its own entry (with its own pricing), so
// callers must pass the full prefixed ID to get the correct rate card.
func lookupModel(modelName string) (ModelDefinition, bool) {
	def, ok := supportedModels[modelName]
	return def, ok
}

type thinkingCapabilities struct {
	supportedEfforts []Effort
	adaptive         bool
	budget           bool
}

var (
	// Claude 4.6 supports both adaptive thinking and deprecated manual
	// budgets. Newer frontier models are adaptive-only:
	// https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
	adaptiveThinkingEfforts         = []Effort{EffortLow, EffortMedium, EffortHigh}
	frontierAdaptiveThinkingEfforts = []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}
	opus46ThinkingEfforts           = []Effort{EffortLow, EffortMedium, EffortHigh, EffortMax}
)

func modelThinkingCapabilities(modelID string) thinkingCapabilities {
	prefix, unprefixed, ok := strings.Cut(modelID, ".")
	if ok && geoProfilePrefixes[prefix] {
		modelID = unprefixed
	}

	switch modelID {
	case ModelClaudeFable5, ModelClaudeSonnet5, ModelClaudeOpus48, ModelClaudeOpus47:
		return thinkingCapabilities{
			supportedEfforts: frontierAdaptiveThinkingEfforts,
			adaptive:         true,
		}
	case ModelClaudeOpus46:
		return thinkingCapabilities{
			supportedEfforts: opus46ThinkingEfforts,
			adaptive:         true,
			budget:           true,
		}
	case ModelClaudeSonnet46:
		return thinkingCapabilities{
			supportedEfforts: adaptiveThinkingEfforts,
			adaptive:         true,
			budget:           true,
		}
	case ModelClaudeOpus45, ModelClaudeSonnet45, ModelClaudeHaiku45:
		return thinkingCapabilities{budget: true}
	default:
		return thinkingCapabilities{}
	}
}

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
	claudeStandardCaps = llm.ModelCapabilities{
		Streaming:     true,
		Tools:         true,
		Vision:        false,
		MultiTurn:     true,
		SystemPrompts: true,
		Reasoning:     true,
	}

	claudeContext1MConstraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 1.0},
		MaxInputTokens:   1000000,
		MaxOutputTokens:  128000,
		SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
	}

	claudeFable5Constraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 1.0},
		MaxInputTokens:   1000000,
		MaxOutputTokens:  128000,
		SupportedParams:  []string{"max_tokens", "stop"},
	}

	claudeContext200kConstraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 1.0},
		MaxInputTokens:   200000,
		MaxOutputTokens:  64000,
		SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
	}

	// Capability/constraint shapes for Amazon Nova 2 Lite. Unlike the Claude
	// entries, JSONMode and StructuredOutput are set: Nova supports Bedrock
	// constrained-decoding structured outputs (response_format / Converse
	// outputConfig.textFormat), which is the primary reason to reach for it on
	// extraction workloads. Multimodal image + video input, text output.
	//
	// Reasoning is left false even though Nova 2 Lite supports extended
	// thinking at the model level: the only lever in this SDK — WithThinking —
	// emits the Anthropic-shaped {"thinking":{"type":"enabled",...}} document
	// via AdditionalModelRequestFields (request_mapper.go), which Nova does not
	// accept, so reasoning is not reachable for Nova through the current API.
	// Flip to true once Nova's reasoning schema is wired through a Bedrock
	// option. top_k is omitted from SupportedParams for the same reason: the
	// Converse request mapper only emits temperature/top_p/max_tokens/stop.
	nova2LiteCaps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true,
		StructuredOutput: true,
		Vision:           true,
		Audio:            false,
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        false,
	}

	nova2LiteConstraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 1.0},
		MaxInputTokens:   1000000,
		MaxOutputTokens:  64000,
		SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
	}

	// Capability/constraint shapes for Mistral Large 3. Tools and native
	// structured output (Converse outputConfig.textFormat / response_format)
	// are set from LiteLLM's bedrock_converse catalog entry, which marks
	// mistral.mistral-large-3-675b-instruct with supports_function_calling and
	// supports_native_structured_output; a live Converse call with
	// outputConfig.textFormat should still confirm constrained decoding before
	// this leaves draft, since no unit test here exercises it. Vision is false:
	// Mistral Large 3 is text-only on Bedrock (Pixtral Large is Mistral's
	// multimodal entry). Reasoning is false — it is not a reasoning model
	// (Magistral is Mistral's reasoning line), and the only reasoning lever in
	// this SDK (WithThinking) emits an Anthropic-shaped thinking document that
	// non-Anthropic Bedrock models reject.
	//
	// Context window: LiteLLM's bedrock_converse entry lists a 128K
	// (max_input_tokens 131072 rounded) input window and an 8,192 max-output
	// cap for the Bedrock-hosted variant — smaller than Mistral's native API,
	// so re-confirm the exact output cap when validating live (it may be
	// raised). top_k is omitted from SupportedParams because the Converse
	// request mapper only emits temperature/top_p/max_tokens/stop.
	mistralLarge3Caps = llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true,
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
	// streaming, native function calling (Tools), built-in Reasoning, and text
	// output. Audio is false, and although Gemma accepts image/video input at
	// the model level, Vision is left FALSE: the SDK has no image Part and the
	// reused OpenAI Responses request mapper only serializes text/tool/reasoning
	// parts, so there is no wired path to send an image today — advertising
	// Vision would promise input the mapper silently drops. Flip to true once
	// image input is threaded through the Responses request mapping. JSONMode /
	// StructuredOutput are false for the same reason: unverified on this path.
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
		Vision:        false,
		Audio:         false,
		MultiTurn:     true,
		SystemPrompts: true,
		Reasoning:     true,
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
)

// supportedModels is the per-variant Bedrock catalog. Every model ID the SDK
// accepts is a literal key in this map and every rate is spelled out in
// place — adding, removing, or repricing a variant is a single visible diff,
// and there is no shared rate constant whose name might survive a price
// change for a single model.
//
// Pricing rule (per AWS https://aws.amazon.com/bedrock/pricing/, 2026-04):
//   - bare ID (when invokable) and geo profiles (us./eu./au./jp.) use the
//     "Geo and In-region Cross-region Inference" rate.
//   - global. profiles use the "Global Cross-region Inference" rate, which
//     is exactly 10% cheaper than the geo rate for the same model.
var supportedModels = map[string]ModelDefinition{
	// ----------------------------------------------------------------
	// Claude Fable 5 — inference-profile-only, no bare entry. Geo
	// profiles cover us and eu (jp/au are not published).
	// ----------------------------------------------------------------
	ModelClaudeFable5Global: {
		Name:                        ModelClaudeFable5Global,
		Label:                       "Claude Fable 5 (Global)",
		Capabilities:                claudeStandardCaps,
		Constraints:                 claudeFable5Constraints,
		RequiresProviderDataSharing: true,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(10.00, 50.00, 1.00).WithCacheCreation(12.50, 20.00, 0),
		),
	},
	ModelClaudeFable5US: {
		Name:                        ModelClaudeFable5US,
		Label:                       "Claude Fable 5 (US)",
		Capabilities:                claudeStandardCaps,
		Constraints:                 claudeFable5Constraints,
		RequiresProviderDataSharing: true,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(11.00, 55.00, 1.10).WithCacheCreation(13.75, 22.00, 0),
		),
	},
	ModelClaudeFable5EU: {
		Name:                        ModelClaudeFable5EU,
		Label:                       "Claude Fable 5 (EU)",
		Capabilities:                claudeStandardCaps,
		Constraints:                 claudeFable5Constraints,
		RequiresProviderDataSharing: true,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(11.00, 55.00, 1.10).WithCacheCreation(13.75, 22.00, 0),
		),
	},

	// ----------------------------------------------------------------
	// Claude Opus 4.8 — inference-profile-only, no bare entry. Geo
	// profiles cover us, eu, jp (au is not published).
	// ----------------------------------------------------------------
	ModelClaudeOpus48Global: {
		Name:         ModelClaudeOpus48Global,
		Label:        "Claude Opus 4.8 (Global)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		),
	},
	ModelClaudeOpus48US: {
		Name:         ModelClaudeOpus48US,
		Label:        "Claude Opus 4.8 (US)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},
	ModelClaudeOpus48EU: {
		Name:         ModelClaudeOpus48EU,
		Label:        "Claude Opus 4.8 (EU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},
	ModelClaudeOpus48JP: {
		Name:         ModelClaudeOpus48JP,
		Label:        "Claude Opus 4.8 (JP)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},

	// ----------------------------------------------------------------
	// Claude Opus 4.7 — inference-profile-only, no bare entry. Geo
	// profiles cover us, eu, jp (au is not published).
	// ----------------------------------------------------------------
	ModelClaudeOpus47Global: {
		Name:         ModelClaudeOpus47Global,
		Label:        "Claude Opus 4.7 (Global)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		),
	},
	ModelClaudeOpus47US: {
		Name:         ModelClaudeOpus47US,
		Label:        "Claude Opus 4.7 (US)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},
	ModelClaudeOpus47EU: {
		Name:         ModelClaudeOpus47EU,
		Label:        "Claude Opus 4.7 (EU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},
	ModelClaudeOpus47JP: {
		Name:         ModelClaudeOpus47JP,
		Label:        "Claude Opus 4.7 (JP)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},

	// ----------------------------------------------------------------
	// Claude Opus 4.6 — inference-profile-only, no bare entry.
	// ----------------------------------------------------------------
	ModelClaudeOpus46Global: {
		Name:         ModelClaudeOpus46Global,
		Label:        "Claude Opus 4.6 (Global)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		),
	},
	ModelClaudeOpus46US: {
		Name:         ModelClaudeOpus46US,
		Label:        "Claude Opus 4.6 (US)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},
	ModelClaudeOpus46EU: {
		Name:         ModelClaudeOpus46EU,
		Label:        "Claude Opus 4.6 (EU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},
	ModelClaudeOpus46AU: {
		Name:         ModelClaudeOpus46AU,
		Label:        "Claude Opus 4.6 (AU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},

	// ----------------------------------------------------------------
	// Claude Opus 4.5 — inference-profile-only, no bare entry.
	// ----------------------------------------------------------------
	ModelClaudeOpus45Global: {
		Name:         ModelClaudeOpus45Global,
		Label:        "Claude Opus 4.5 (Global)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.00, 25.00, 0.50).WithCacheCreation(6.25, 10.00, 0),
		),
	},
	ModelClaudeOpus45US: {
		Name:         ModelClaudeOpus45US,
		Label:        "Claude Opus 4.5 (US)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},
	ModelClaudeOpus45EU: {
		Name:         ModelClaudeOpus45EU,
		Label:        "Claude Opus 4.5 (EU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(5.50, 27.50, 0.55).WithCacheCreation(6.875, 11.00, 0),
		),
	},

	// ----------------------------------------------------------------
	// Claude Sonnet 5 — inference-profile-only, no bare entry. Only us
	// and global are published so far (see const block); 1M context.
	// ----------------------------------------------------------------
	ModelClaudeSonnet5Global: {
		Name:         ModelClaudeSonnet5Global,
		Label:        "Claude Sonnet 5 (Global)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
		),
	},
	ModelClaudeSonnet5US: {
		Name:         ModelClaudeSonnet5US,
		Label:        "Claude Sonnet 5 (US)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext1MConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
		),
	},

	// ----------------------------------------------------------------
	// Claude Sonnet 4.6 — inference-profile-only, no bare entry.
	// ----------------------------------------------------------------
	ModelClaudeSonnet46Global: {
		Name:         ModelClaudeSonnet46Global,
		Label:        "Claude Sonnet 4.6 (Global)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
		),
	},
	ModelClaudeSonnet46US: {
		Name:         ModelClaudeSonnet46US,
		Label:        "Claude Sonnet 4.6 (US)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
		),
	},
	ModelClaudeSonnet46EU: {
		Name:         ModelClaudeSonnet46EU,
		Label:        "Claude Sonnet 4.6 (EU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
		),
	},
	ModelClaudeSonnet46AU: {
		Name:         ModelClaudeSonnet46AU,
		Label:        "Claude Sonnet 4.6 (AU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
		),
	},

	// ----------------------------------------------------------------
	// Claude Sonnet 4.5 — inference-profile-only, widest geo coverage.
	// ----------------------------------------------------------------
	ModelClaudeSonnet45Global: {
		Name:         ModelClaudeSonnet45Global,
		Label:        "Claude Sonnet 4.5 (Global)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.00, 15.00, 0.30).WithCacheCreation(3.75, 6.00, 0),
		),
	},
	ModelClaudeSonnet45US: {
		Name:         ModelClaudeSonnet45US,
		Label:        "Claude Sonnet 4.5 (US)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
		),
	},
	ModelClaudeSonnet45EU: {
		Name:         ModelClaudeSonnet45EU,
		Label:        "Claude Sonnet 4.5 (EU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
		),
	},
	ModelClaudeSonnet45AU: {
		Name:         ModelClaudeSonnet45AU,
		Label:        "Claude Sonnet 4.5 (AU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
		),
	},
	ModelClaudeSonnet45JP: {
		Name:         ModelClaudeSonnet45JP,
		Label:        "Claude Sonnet 4.5 (JP)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(3.30, 16.50, 0.33).WithCacheCreation(4.125, 6.60, 0),
		),
	},

	// ----------------------------------------------------------------
	// Claude Haiku 4.5 — inference-profile-only, no bare entry.
	// ----------------------------------------------------------------
	ModelClaudeHaiku45Global: {
		Name:         ModelClaudeHaiku45Global,
		Label:        "Claude Haiku 4.5 (Global)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(1.00, 5.00, 0.10).WithCacheCreation(1.25, 2.00, 0),
		),
	},
	ModelClaudeHaiku45US: {
		Name:         ModelClaudeHaiku45US,
		Label:        "Claude Haiku 4.5 (US)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(1.10, 5.50, 0.11).WithCacheCreation(1.375, 2.20, 0),
		),
	},
	ModelClaudeHaiku45EU: {
		Name:         ModelClaudeHaiku45EU,
		Label:        "Claude Haiku 4.5 (EU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(1.10, 5.50, 0.11).WithCacheCreation(1.375, 2.20, 0),
		),
	},
	ModelClaudeHaiku45AU: {
		Name:         ModelClaudeHaiku45AU,
		Label:        "Claude Haiku 4.5 (AU)",
		Capabilities: claudeStandardCaps,
		Constraints:  claudeContext200kConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(1.10, 5.50, 0.11).WithCacheCreation(1.375, 2.20, 0),
		),
	},

	// ----------------------------------------------------------------
	// Amazon Nova 2 Lite — inference-profile-only, no bare entry. Geo
	// profiles cover us, eu, jp; global. is also published.
	//
	// All rates verified against the AWS Price List bulk API (authoritative,
	// machine-readable — the pricing web page is JS-rendered and unusable):
	// offers/v1.0/aws/AmazonBedrock/<ts>/us-east-1/index.json, published
	// 2026-07-07. Pricing follows the same geo/global split as the Claude
	// entries above — the global. tier is the round base rate and each geo
	// profile is exactly 1.10x it (enforced by TestGeoGlobalRatio):
	//   global (…-cross-region-global): $0.30 in / $2.50 out / $0.075 cache-read
	//   geo/in-region:                  $0.33 in / $2.75 out / $0.0825 cache-read
	// Cache reads are billed at 25% of the input rate (75% discount).
	//
	// Cache WRITE is FREE: the AWS price list carries an explicit
	// USE1-Nova2.0Lite-cache-write-input-token-count usagetype priced at
	// $0.00 in both tiers (Nova charges nothing to populate the cache; only
	// cache reads are billed, at the discounted rate above). So no
	// WithCacheCreation is set — the cache-write buckets stay at zero, which
	// TestAllModelsHavePricing explicitly allows for free-cache-write models.
	// ----------------------------------------------------------------
	ModelNova2LiteGlobal: {
		Name:         ModelNova2LiteGlobal,
		Label:        "Amazon Nova 2 Lite (Global)",
		Capabilities: nova2LiteCaps,
		Constraints:  nova2LiteConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(0.30, 2.50, 0.075),
		),
	},
	ModelNova2LiteUS: {
		Name:         ModelNova2LiteUS,
		Label:        "Amazon Nova 2 Lite (US)",
		Capabilities: nova2LiteCaps,
		Constraints:  nova2LiteConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(0.33, 2.75, 0.0825),
		),
	},
	ModelNova2LiteEU: {
		Name:         ModelNova2LiteEU,
		Label:        "Amazon Nova 2 Lite (EU)",
		Capabilities: nova2LiteCaps,
		Constraints:  nova2LiteConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(0.33, 2.75, 0.0825),
		),
	},
	ModelNova2LiteJP: {
		Name:         ModelNova2LiteJP,
		Label:        "Amazon Nova 2 Lite (JP)",
		Capabilities: nova2LiteCaps,
		Constraints:  nova2LiteConstraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(0.33, 2.75, 0.0825),
		),
	},

	// ----------------------------------------------------------------
	// Mistral Large 3 — on-demand / in-region, registered under the BARE ID
	// (no inference profile; the "us.mistral..." profile does not exist —
	// invoking it returns ValidationException "invalid model identifier").
	// Unlike the Claude/Nova entries it does NOT follow the geo/global 1.10x
	// split: AWS prices it by service tier (standard/flex/priority/batch) under
	// a single us-east-1 usagetype, with no global. rate — so there is no
	// global. sibling and TestGeoGlobalRatio does not apply. We register the
	// on-demand *standard* tier here (the tier the Converse proxy uses);
	// flex/priority/batch are not modeled.
	//
	// Rates verified against the AWS Price List bulk API (authoritative,
	// machine-readable — the pricing web page is JS-rendered and unusable):
	// offers/v1.0/aws/AmazonBedrock/20260707080509/us-east-1/index.json,
	// usagetype mistral.mistral-large-3-675b-instruct-mantle-{input,output}-
	// tokens-standard:
	//   standard: $0.50 / 1M input, $1.50 / 1M output
	// Cross-checked against LiteLLM's bedrock_converse catalog entry
	// (input 5e-07, output 1.5e-06 per token; 128K context; supports function
	// calling + native structured output).
	//
	// Prompt caching is NOT billed for Mistral Large 3 (the price list carries
	// no cache-read/cache-write usagetype), so all cache rates stay zero — the
	// documented shape for a non-caching provider (see pricing.Rates). This is
	// why ModelMistralLarge3 is listed in noCacheModels in pricing_test.go.
	//
	// Because the bare ID has no geo prefix, this entry is excluded from the
	// per-region conformance sweep (BedrockFixture.Models filters to the
	// source region's geo profile); the aigw external_llm integration test
	// covers live invocation and skips cleanly when access is not granted.
	// ----------------------------------------------------------------
	ModelMistralLarge3: {
		Name:         ModelMistralLarge3,
		Label:        "Mistral Large 3",
		Capabilities: mistralLarge3Caps,
		Constraints:  mistralLarge3Constraints,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(0.50, 1.50, 0),
		),
	},

	// ----------------------------------------------------------------
	// Google Gemma 4 family — mantle-only (Mantle: true), invoked by the
	// bare ID via the bedrock-mantle Responses endpoint (no inference
	// profile; the model cards list Geo/Global = Not supported). Because the
	// bare IDs have no geo prefix, these are excluded from the per-region
	// geo-profile conformance sweep and from TestGeoGlobalRatio (no global.
	// sibling), and they are listed in noCacheModels in pricing_test.go.
	//
	// Pricing is the on-demand STANDARD-tier rate for the US regions
	// (us-east-1/2, us-west-2), taken from https://aws.amazon.com/bedrock/pricing/
	// (Google section, 2026-06). AWS also publishes a ~15-20% higher
	// eu-central-1 (Frankfurt) rate that is NOT separately modeled here (the
	// catalog keys on model ID, not region, and no existing Bedrock entry
	// carries a region override): 31B EU $0.17/$0.48, 26B-A4B EU $0.16/$0.48,
	// E2B EU $0.05/$0.10 per 1M in/out. Revisit with a pricing.Info region
	// Override if EU-accurate billing is required. The mantle Responses
	// endpoint bills only input/output (no cache-read/cache-write usagetype
	// is published for Gemma), so all cache rates stay zero — the documented
	// shape for a non-caching model (see pricing.Rates / noCacheModels).
	// ----------------------------------------------------------------
	ModelGemma431B: {
		Name:         ModelGemma431B,
		Label:        "Google Gemma 4 31B",
		Capabilities: gemma4Caps,
		Constraints:  gemma4Context256kConstraints,
		Mantle:       true,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(0.14, 0.40, 0),
		),
	},
	ModelGemma426BA4B: {
		Name:         ModelGemma426BA4B,
		Label:        "Google Gemma 4 26B-A4B",
		Capabilities: gemma4Caps,
		Constraints:  gemma4Context256kConstraints,
		Mantle:       true,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(0.13, 0.40, 0),
		),
	},
	ModelGemma4E2B: {
		Name:         ModelGemma4E2B,
		Label:        "Google Gemma 4 E2B",
		Capabilities: gemma4Caps,
		Constraints:  gemma4Context128kConstraints,
		Mantle:       true,
		Pricing: pricing.FlatInfoFromRates(
			pricing.NewRates(0.04, 0.08, 0),
		),
	},
}
