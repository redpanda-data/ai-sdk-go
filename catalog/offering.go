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

package catalog

import "github.com/redpanda-data/ai-sdk-go/pricing"

// Offering is one resolved, validated catalog row. It is produced by New,
// never authored directly: the embedded Entry has been normalized
// (default Label, explicit Modalities, StageGA default) and the facts
// reference has been resolved against the Registry.
type Offering struct {
	Entry

	provider string
	facts    Facts
}

// Provider returns the name of the provider serving this offering.
func (o Offering) Provider() string {
	return o.provider
}

// Facts returns the host-independent facts of the underlying model.
func (o Offering) Facts() Facts {
	return o.facts
}

// PriceTier is a coarse cost class for catalog UIs, derived from the
// offering's base rate card.
type PriceTier string

const (
	// PriceTierUnknown means the offering has no usable base rate: an
	// input or output rate authored as 0 means "unpriced", NOT "free" —
	// bucketing an unpriced model as cheap would be the most damaging
	// thing this derivation could do.
	PriceTierUnknown PriceTier = ""

	// PriceTierFree means both the input and output base rates are
	// explicitly pricing.RateFree.
	PriceTierFree PriceTier = "free"

	PriceTierLow    PriceTier = "low"
	PriceTierMedium PriceTier = "medium"
	PriceTierHigh   PriceTier = "high"
)

// Blended price thresholds in microcents per million tokens. Absolute by
// design: percentile bucketing would silently re-tier the whole catalog
// the moment one expensive model landed, violating the rule that adding a
// model never changes existing entries.
const (
	priceTierLowMax    = 250_000_000   // blended < $2.50/M ⇒ low
	priceTierMediumMax = 1_000_000_000 // blended < $10.00/M ⇒ medium
)

// PriceTier derives the offering's cost class from Pricing.Default.Base
// ONLY. Selector overrides (speed, service tier, region) and context
// brackets are ignored, so an offering's tier does not depend on which
// request you ask about.
//
// The blend is (3*input + output) / 4, reflecting a roughly 3:1
// input:output token ratio for agent workloads.
func (o Offering) PriceTier() PriceTier {
	in := o.Pricing.Default.Base.InputPerMillion
	out := o.Pricing.Default.Base.OutputPerMillion

	if in == pricing.RateFree && out == pricing.RateFree {
		return PriceTierFree
	}

	if in == 0 || out == 0 {
		return PriceTierUnknown
	}

	// A single free bucket alongside a priced one still blends; RateFree
	// contributes zero cost.
	blended := (3*max(in, 0) + max(out, 0)) / 4

	switch {
	case blended < priceTierLowMax:
		return PriceTierLow
	case blended < priceTierMediumMax:
		return PriceTierMedium
	default:
		return PriceTierHigh
	}
}
