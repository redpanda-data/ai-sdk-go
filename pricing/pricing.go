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

// Package pricing provides an in-memory pricing catalog for LLM model cost
// calculation. Prices are stored in microcents per million tokens
// (1 cent = 1,000,000 microcents).
package pricing

// Info holds pricing for a model. Embed this in provider ModelDefinition
// structs so pricing is defined alongside capabilities.
// All prices are in microcents per million tokens.
//
// For tiered models, set only Tiers — the flat fields (InputPerMillion etc.)
// are automatically populated from the first (lowest) tier by ToModelPricing.
// For flat-rate models, set the flat fields and leave Tiers nil.
type Info struct {
	InputPerMillion       int64
	OutputPerMillion      int64
	CachedInputPerMillion int64
	Tiers                 []Tier // optional context-length tiers
}

// ToModelPricing converts Info into a ModelPricing entry for the given model ID.
//
// When Tiers is non-empty, the flat fields are auto-populated from the
// first tier to prevent duplication drift.
func (info Info) ToModelPricing(modelID string) ModelPricing {
	resolved := info

	if len(info.Tiers) > 0 {
		// Auto-populate flat fields from the first (lowest) tier so they
		// can never drift out of sync with the tiered values.
		first := info.Tiers[0]
		resolved.InputPerMillion = first.InputPerMillion
		resolved.OutputPerMillion = first.OutputPerMillion
		resolved.CachedInputPerMillion = first.CachedInputPerMillion
	}

	return ModelPricing{ModelID: modelID, Info: resolved}
}

// Tier represents pricing for a specific context-length range.
// Some providers (e.g. Google Gemini Pro) charge different rates based on
// the total input context size. All prices are in microcents per million tokens.
type Tier struct {
	// MaxInputTokens is the upper bound (inclusive) of input context tokens
	// for this tier. A value of 0 means unlimited (catch-all tier).
	MaxInputTokens        int64
	InputPerMillion       int64
	OutputPerMillion      int64
	CachedInputPerMillion int64
}

// ModelPricing holds the pricing for a single model, identified by ModelID.
type ModelPricing struct {
	Info

	ModelID string
}

// Cost represents the calculated cost breakdown for a request.
// All values are in microcents.
type Cost struct {
	InputCostMicrocents  int64
	OutputCostMicrocents int64
	CachedCostMicrocents int64
	TotalCostMicrocents  int64
}

// CalculateCost computes the cost for a given pricing info and token counts.
// Cost = tokens * pricePerMillion / 1_000_000.
//
// When the info has Tiers, the tier is selected based on total context size
// (inputTokens + cachedTokens). Otherwise the flat fields are used.
//
// Note: integer division truncates toward zero, under-counting by up to
// ~1 microcent per component per request. This is acceptable for cost
// reporting; if this is ever used for billing, switch to rounding:
// (tokens * rate + 500_000) / 1_000_000.
func CalculateCost(info *Info, inputTokens, outputTokens, cachedTokens int) Cost {
	inputRate, outputRate, cachedRate := info.InputPerMillion, info.OutputPerMillion, info.CachedInputPerMillion

	if len(info.Tiers) > 0 {
		contextSize := int64(inputTokens + cachedTokens)
		for _, tier := range info.Tiers {
			if tier.MaxInputTokens == 0 || contextSize <= tier.MaxInputTokens {
				inputRate = tier.InputPerMillion
				outputRate = tier.OutputPerMillion
				cachedRate = tier.CachedInputPerMillion

				break
			}
		}
	}

	input := int64(inputTokens) * inputRate / 1_000_000
	output := int64(outputTokens) * outputRate / 1_000_000
	cached := int64(cachedTokens) * cachedRate / 1_000_000

	return Cost{
		InputCostMicrocents:  input,
		OutputCostMicrocents: output,
		CachedCostMicrocents: cached,
		TotalCostMicrocents:  input + output + cached,
	}
}

// Catalog is an in-memory lookup table of model pricing.
type Catalog struct {
	models map[string]*ModelPricing
}

// NewCatalog creates a Catalog from a slice of ModelPricing.
func NewCatalog(models []ModelPricing) *Catalog {
	m := make(map[string]*ModelPricing, len(models))
	for i := range models {
		m[models[i].ModelID] = &models[i]
	}

	return &Catalog{models: m}
}

// Lookup returns the ModelPricing for the given model ID.
func (c *Catalog) Lookup(modelID string) (*ModelPricing, bool) {
	mp, ok := c.models[modelID]

	return mp, ok
}
