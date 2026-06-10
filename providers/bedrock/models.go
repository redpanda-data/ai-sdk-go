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

// ModelDefinition defines a model with its capabilities and constraints.
type ModelDefinition struct {
	Name                        string // Real Bedrock model ID (e.g. "us.anthropic.claude-sonnet-4-6")
	Label                       string
	Capabilities                llm.ModelCapabilities
	Constraints                 llm.ModelConstraints
	Pricing                     pricing.Info
	RequiresProviderDataSharing bool
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

// hasRegionPrefix reports whether a model ID already contains a region
// inference-profile prefix (e.g. "us.anthropic.claude-sonnet-4-6").
// It checks for the "{region}.{provider}." pattern by counting dot-separated
// segments: bare model IDs like "anthropic.claude-sonnet-4-6" have one dot,
// while prefixed IDs have two or more.
func hasRegionPrefix(modelID string) bool {
	_, after, ok := strings.Cut(modelID, ".")
	if !ok {
		return false
	}

	return strings.ContainsRune(after, '.')
}

// lookupModel finds a ModelDefinition by exact model ID. Each inference
// profile variant is registered as its own entry (with its own pricing), so
// callers must pass the full prefixed ID to get the correct rate card.
func lookupModel(modelName string) (ModelDefinition, bool) {
	def, ok := supportedModels[modelName]
	return def, ok
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
}
