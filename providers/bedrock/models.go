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
// inference profile. AWS bills these at different rates: the bare ID and the
// "global." profile use the headline price, while geo profiles
// ("us." / "eu." / "au." / "apac.") carry a ~10% cross-region premium.
//
// New 4.6+ models (Sonnet 4.6, Opus 4.6/4.7) are inference-profile-only and
// can only be invoked via a prefixed ID. Older 4.5 models can be invoked with
// the bare ID for in-region access.
const (
	// Claude Sonnet 4.6 (inference-profile-only).
	ModelClaudeSonnet46       = "anthropic.claude-sonnet-4-6"
	ModelClaudeSonnet46Global = "global." + ModelClaudeSonnet46
	ModelClaudeSonnet46US     = "us." + ModelClaudeSonnet46
	ModelClaudeSonnet46EU     = "eu." + ModelClaudeSonnet46
	ModelClaudeSonnet46AU     = "au." + ModelClaudeSonnet46

	// Claude Sonnet 4.5.
	ModelClaudeSonnet45       = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	ModelClaudeSonnet45Global = "global." + ModelClaudeSonnet45
	ModelClaudeSonnet45US     = "us." + ModelClaudeSonnet45
	ModelClaudeSonnet45EU     = "eu." + ModelClaudeSonnet45
	ModelClaudeSonnet45AU     = "au." + ModelClaudeSonnet45

	// Claude Haiku 4.5.
	ModelClaudeHaiku45       = "anthropic.claude-haiku-4-5-20251001-v1:0"
	ModelClaudeHaiku45Global = "global." + ModelClaudeHaiku45
	ModelClaudeHaiku45US     = "us." + ModelClaudeHaiku45
	ModelClaudeHaiku45EU     = "eu." + ModelClaudeHaiku45
	ModelClaudeHaiku45AU     = "au." + ModelClaudeHaiku45
	ModelClaudeHaiku45APAC   = "apac." + ModelClaudeHaiku45

	// Claude Opus 4.7 (inference-profile-only).
	ModelClaudeOpus47       = "anthropic.claude-opus-4-7"
	ModelClaudeOpus47Global = "global." + ModelClaudeOpus47
	ModelClaudeOpus47US     = "us." + ModelClaudeOpus47
	ModelClaudeOpus47EU     = "eu." + ModelClaudeOpus47
	ModelClaudeOpus47AU     = "au." + ModelClaudeOpus47

	// Claude Opus 4.6 (inference-profile-only).
	ModelClaudeOpus46       = "anthropic.claude-opus-4-6-v1"
	ModelClaudeOpus46Global = "global." + ModelClaudeOpus46
	ModelClaudeOpus46US     = "us." + ModelClaudeOpus46
	ModelClaudeOpus46EU     = "eu." + ModelClaudeOpus46
	ModelClaudeOpus46AU     = "au." + ModelClaudeOpus46

	// Claude Opus 4.5.
	ModelClaudeOpus45       = "anthropic.claude-opus-4-5-20251101-v1:0"
	ModelClaudeOpus45Global = "global." + ModelClaudeOpus45
	ModelClaudeOpus45EU     = "eu." + ModelClaudeOpus45
)

// ModelDefinition defines a model with its capabilities and constraints.
type ModelDefinition struct {
	Name         string // Real Bedrock model ID (e.g. "us.anthropic.claude-sonnet-4-6")
	Label        string
	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints
	Pricing      pricing.Info
}

// inferenceProfileRegion maps an AWS region to the Bedrock cross-region
// inference profile geographic prefix.
//
// AWS uses these prefixes:
//   - "us"   for US regions        (us-east-1, us-west-2, …)
//   - "eu"   for European regions  (eu-west-1, eu-central-1, …)
//   - "apac" for Asia Pacific      (ap-southeast-1, ap-northeast-1, …)
//
// See https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference-support.html
func inferenceProfileRegion(region string) string {
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

// claudeModel describes a single logical Claude model on Bedrock and the
// inference profile variants under which it is exposed.
type claudeModel struct {
	baseID       string
	label        string
	capabilities llm.ModelCapabilities
	constraints  llm.ModelConstraints
	// baseRates apply to the bare ID and the "global." profile.
	baseRates pricing.Rates
	// geoRates apply to the geo profiles (us./eu./au./apac.) — the AWS
	// cross-region inference premium (~10% over baseRates).
	geoRates pricing.Rates
	// geoProfiles lists the geo prefixes (without trailing dot) that AWS
	// publishes for this model.
	geoProfiles []string
}

// expand returns one ModelDefinition per inference profile variant.
func (c claudeModel) expand() []ModelDefinition {
	defs := make([]ModelDefinition, 0, 2+len(c.geoProfiles))

	defs = append(defs,
		ModelDefinition{
			Name:         c.baseID,
			Label:        c.label,
			Capabilities: c.capabilities,
			Constraints:  c.constraints,
			Pricing:      pricing.FlatInfoFromRates(c.baseRates),
		},
		ModelDefinition{
			Name:         "global." + c.baseID,
			Label:        c.label + " (Global)",
			Capabilities: c.capabilities,
			Constraints:  c.constraints,
			Pricing:      pricing.FlatInfoFromRates(c.baseRates),
		},
	)

	for _, geo := range c.geoProfiles {
		defs = append(defs, ModelDefinition{
			Name:         geo + "." + c.baseID,
			Label:        c.label + " (" + strings.ToUpper(geo) + ")",
			Capabilities: c.capabilities,
			Constraints:  c.constraints,
			Pricing:      pricing.FlatInfoFromRates(c.geoRates),
		})
	}

	return defs
}

// claudeMillionTokenCaps and claudeStandardCaps capture the two capability
// shapes shared by all Claude models on Bedrock.
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

	claudeContext200kConstraints = llm.ModelConstraints{
		TemperatureRange: [2]float64{0.0, 1.0},
		MaxInputTokens:   200000,
		MaxOutputTokens:  64000,
		SupportedParams:  []string{"temperature", "top_p", "max_tokens", "stop"},
	}
)

// claudeModels enumerates every logical Claude model on Bedrock. Pricing
// rates source: https://aws.amazon.com/bedrock/pricing/ (as of 2026-04).
//
// Per-profile pricing notes:
//   - Bare ID and "global.": headline rate ("baseRates").
//   - Geo profiles (us./eu./au./apac.): ~10% cross-region premium ("geoRates").
//
// Non-Anthropic Bedrock models (Gemma, DeepSeek, Amazon Nova) can have
// larger regional pricing differences (up to ~57% premium); those should
// model regional rates with pricing.Info overrides instead of this helper.
var claudeModels = []claudeModel{
	{
		baseID:       ModelClaudeOpus47,
		label:        "Claude Opus 4.7",
		capabilities: claudeStandardCaps,
		constraints:  claudeContext1MConstraints,
		baseRates: pricing.NewRates(5.00, 25.00, 0.50).
			WithCacheCreation(6.25, 10.00, 0),
		geoRates: pricing.NewRates(5.50, 27.50, 0.55).
			WithCacheCreation(6.875, 11.00, 0),
		geoProfiles: []string{"us", "eu", "au"},
	},
	{
		baseID:       ModelClaudeOpus46,
		label:        "Claude Opus 4.6",
		capabilities: claudeStandardCaps,
		constraints:  claudeContext1MConstraints,
		baseRates: pricing.NewRates(5.00, 25.00, 0.50).
			WithCacheCreation(6.25, 10.00, 0),
		geoRates: pricing.NewRates(5.50, 27.50, 0.55).
			WithCacheCreation(6.875, 11.00, 0),
		geoProfiles: []string{"us", "eu", "au"},
	},
	{
		baseID:       ModelClaudeOpus45,
		label:        "Claude Opus 4.5",
		capabilities: claudeStandardCaps,
		constraints:  claudeContext200kConstraints,
		baseRates: pricing.NewRates(5.00, 25.00, 0.50).
			WithCacheCreation(6.25, 10.00, 0),
		geoRates: pricing.NewRates(5.50, 27.50, 0.55).
			WithCacheCreation(6.875, 11.00, 0),
		geoProfiles: []string{"eu"},
	},
	{
		baseID:       ModelClaudeSonnet46,
		label:        "Claude Sonnet 4.6",
		capabilities: claudeStandardCaps,
		constraints:  claudeContext200kConstraints,
		baseRates: pricing.NewRates(3.00, 15.00, 0.30).
			WithCacheCreation(3.75, 6.00, 0),
		geoRates: pricing.NewRates(3.30, 16.50, 0.33).
			WithCacheCreation(4.125, 6.60, 0),
		geoProfiles: []string{"us", "eu", "au"},
	},
	{
		baseID:       ModelClaudeSonnet45,
		label:        "Claude Sonnet 4.5",
		capabilities: claudeStandardCaps,
		constraints:  claudeContext200kConstraints,
		baseRates: pricing.NewRates(3.00, 15.00, 0.30).
			WithCacheCreation(3.75, 6.00, 0),
		geoRates: pricing.NewRates(3.30, 16.50, 0.33).
			WithCacheCreation(4.125, 6.60, 0),
		geoProfiles: []string{"us", "eu", "au"},
	},
	{
		baseID:       ModelClaudeHaiku45,
		label:        "Claude Haiku 4.5",
		capabilities: claudeStandardCaps,
		constraints:  claudeContext200kConstraints,
		baseRates: pricing.NewRates(1.00, 5.00, 0.10).
			WithCacheCreation(1.25, 2.00, 0),
		geoRates: pricing.NewRates(1.10, 5.50, 0.11).
			WithCacheCreation(1.375, 2.20, 0),
		geoProfiles: []string{"us", "eu", "au", "apac"},
	},
}

// supportedModels is the per-variant catalog. Every Bedrock model ID that the
// SDK accepts — including each inference profile prefix — is a key in this map.
var supportedModels = buildSupportedModels()

func buildSupportedModels() map[string]ModelDefinition {
	m := make(map[string]ModelDefinition)
	for _, model := range claudeModels {
		for _, def := range model.expand() {
			m[def.Name] = def
		}
	}
	return m
}
