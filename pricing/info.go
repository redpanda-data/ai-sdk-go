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

package pricing

import (
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Info is the pricing entry for one model.
//
// Default always applies. Overrides refine it for request metadata such
// as speed, service tier, or region. Each Override carries a full
// RateCard rather than a Rates delta, so "fast + above_200k" style
// pricing is expressible without schema changes.
type Info struct {
	Default   RateCard
	Overrides []Override
}

// Clone returns a deep copy of the Info. Info embeds slices (bracket
// lists and overrides), so callers that hand Info values across an
// immutability boundary — the model catalog's deep-copying accessors —
// clone rather than share.
func (info Info) Clone() Info {
	return cloneInfo(info)
}

// WithOverride returns a copy of the Info with a selector-specific
// rate card override appended. The builder catches duplicates and
// ambiguous overlaps at Build time.
func (info Info) WithOverride(selector Selector, card RateCard) Info {
	info.Overrides = append(cloneOverrides(info.Overrides), Override{
		Match:    normalizeSelector(selector),
		RateCard: cloneRateCard(card),
	})

	return info
}

// SelectorFromResponse extracts the pricing-relevant selector fields
// from a model response.
func SelectorFromResponse(resp *llm.Response) Selector {
	if resp == nil {
		return Selector{}
	}

	return normalizeSelector(Selector{
		ServiceTier: resp.ServiceTier,
		Speed:       resp.Speed,
		Region:      resp.InferenceRegion,
	})
}

// UsageField identifies one billable usage bucket in Cost.Breakdown.
//
// Adding a new UsageField is the extensibility seam for new billable
// buckets (e.g. audio input, image input). Renaming or removing an
// existing value is a breaking change because consumers key breakdown
// maps on these constants.
type UsageField string

const (
	UsageFieldInput                   UsageField = "input"
	UsageFieldCachedInput             UsageField = "cached_input"
	UsageFieldCacheCreation5m         UsageField = "cache_creation_5m"
	UsageFieldCacheCreation1h         UsageField = "cache_creation_1h"
	UsageFieldCacheCreationUnknownTTL UsageField = "cache_creation_unknown_ttl"
	UsageFieldOutput                  UsageField = "output"
	UsageFieldReasoning               UsageField = "reasoning"
	UsageFieldToolUseInput            UsageField = "tool_use_input"
)

// CalcRequest carries the metadata required to resolve a rate card and
// price the usage.
//
// ContextTokens is explicit because provider context-tier pricing keys
// off the size of the request context, not the billed-token sum. When
// left at zero the calculator falls back to the total input-side token
// sum from llm.TokenUsage; callers that know the exact context size
// (e.g. Google's grounding-adjusted size) should set this explicitly.
type CalcRequest struct {
	Selector      Selector
	ContextTokens int64
}

// Fallback records a case where the requested selector did not match
// an override exactly, so resolution fell back to a less specific
// match or to the model's Default rate card.
type Fallback struct {
	// Requested is the selector the caller presented.
	Requested Selector
	// Resolved is the override selector that ultimately matched. Its
	// zero value means resolution fell all the way back to
	// Info.Default.
	Resolved Selector
}

// String renders the fallback in a human-readable form.
func (f Fallback) String() string {
	if f.Resolved.IsZero() {
		return fmt.Sprintf("selector %q -> default", selectorString(f.Requested))
	}

	return fmt.Sprintf("selector %q -> %q", selectorString(f.Requested), selectorString(f.Resolved))
}

// Cost is the result of applying a rate card to usage.
//
// Breakdown carries per-UsageField microcents; Total is the sum. The
// map shape (rather than a fixed set of fields) makes additive
// UsageFields a non-breaking change and lets "known-zero" rates be
// represented as a key with value 0 distinct from an unpriced bucket.
//
// Provenance fields:
//   - CatalogVersion identifies the catalog snapshot that produced the
//     cost.
//   - AppliedSelector is the override selector that matched, or the
//     zero value if Default applied.
//   - AppliedBracketMinContextTokens is the bracket threshold that was
//     applied (0 for RateCard.Base).
//   - Unpriced lists non-zero usage buckets that had no corresponding
//     rate.
//   - Fallbacks explains any selector fallback decisions.
type Cost struct {
	// Breakdown holds the per-UsageField microcent amounts for priced
	// buckets. Unpriced buckets are reported in Unpriced instead.
	Breakdown map[UsageField]int64
	// Total equals sum(Breakdown) and is set only by Calculate.
	// Hand-constructed Cost values must populate it explicitly.
	Total int64

	CatalogVersion                 string
	AppliedSelector                Selector
	AppliedBracketMinContextTokens int64
	Unpriced                       []UsageField
	Fallbacks                      []Fallback
}
