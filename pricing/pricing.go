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

import "time"

// Info holds the current pricing for a model. Embed this in provider
// ModelDefinition structs so pricing is defined alongside capabilities.
// All prices are in microcents per million tokens.
type Info struct {
	InputPerMillion       int64
	OutputPerMillion      int64
	CachedInputPerMillion int64
	Tiers                 []Tier // optional context-length tiers
}

// ToModelPricing converts Info into a ModelPricing entry for the given model ID.
func (info Info) ToModelPricing(modelID string) ModelPricing {
	rate := Rate{
		EffectiveFrom:         Epoch,
		InputPerMillion:       info.InputPerMillion,
		OutputPerMillion:      info.OutputPerMillion,
		CachedInputPerMillion: info.CachedInputPerMillion,
		Tiers:                 info.Tiers,
	}

	return ModelPricing{ModelID: modelID, Rates: []Rate{rate}}
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

// Rate represents pricing for a model at a point in time.
// All prices are in microcents per million tokens.
//
// The top-level InputPerMillion/OutputPerMillion/CachedInputPerMillion fields
// are the default (lowest) tier and are always populated. When Tiers is
// non-empty, CalculateCost selects the matching tier based on context size;
// otherwise it uses the default fields.
type Rate struct {
	EffectiveFrom         time.Time
	InputPerMillion       int64
	OutputPerMillion      int64
	CachedInputPerMillion int64

	// Tiers holds context-length-based pricing tiers, sorted ascending by
	// MaxInputTokens. The last entry should have MaxInputTokens=0 (unlimited).
	// When empty, the flat Rate fields are used for all requests.
	Tiers []Tier
}

// ModelPricing holds the pricing history for a single model.
// Rates are sorted by EffectiveFrom descending (newest first).
type ModelPricing struct {
	ModelID string
	Rates   []Rate
}

// RateAt returns the effective rate for the given timestamp.
// It returns nil if the timestamp is before all known rates.
func (mp *ModelPricing) RateAt(t time.Time) *Rate {
	for i := range mp.Rates {
		if !t.Before(mp.Rates[i].EffectiveFrom) {
			return &mp.Rates[i]
		}
	}

	return nil
}

// CurrentRate returns the most recent rate (Rates[0]), or nil if empty.
func (mp *ModelPricing) CurrentRate() *Rate {
	if len(mp.Rates) == 0 {
		return nil
	}

	return &mp.Rates[0]
}

// Cost represents the calculated cost breakdown for a request.
// All values are in microcents.
type Cost struct {
	InputCostMicrocents  int64
	OutputCostMicrocents int64
	CachedCostMicrocents int64
	TotalCostMicrocents  int64
}

// CalculateCost computes the cost for a given rate and token counts.
// Cost = tokens * pricePerMillion / 1_000_000.
//
// When the rate has Tiers, the tier is selected based on total context size
// (inputTokens + cachedTokens). Otherwise the flat rate fields are used.
//
// Note: integer division truncates toward zero, under-counting by up to
// ~1 microcent per component per request. This is acceptable for cost
// reporting; if this is ever used for billing, switch to rounding:
// (tokens * rate + 500_000) / 1_000_000.
func CalculateCost(rate *Rate, inputTokens, outputTokens, cachedTokens int) Cost {
	inputRate, outputRate, cachedRate := rate.InputPerMillion, rate.OutputPerMillion, rate.CachedInputPerMillion

	if len(rate.Tiers) > 0 {
		contextSize := int64(inputTokens + cachedTokens)
		for _, tier := range rate.Tiers {
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
