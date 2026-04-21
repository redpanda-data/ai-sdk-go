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
	"errors"
	"fmt"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Catalog is an in-memory lookup table of model pricing. The zero
// value is not usable; construct catalogs through NewCatalog().Build().
//
// Catalogs key on bare model IDs. Provider namespacing is the
// responsibility of each provider package (Bedrock IDs are prefixed
// "anthropic.claude-...", OpenAI IDs are "gpt-...", etc.), and the
// builder errors out if two providers try to register the exact same
// ID so runtime lookups are always unambiguous.
type Catalog struct {
	models  map[string]Info
	version string
}

// Lookup returns a deep copy of the pricing Info for the given model
// ID. The copy isolates callers from mutation and keeps the catalog's
// stored Info values authoritative for CatalogVersion.
func (c *Catalog) Lookup(modelID string) (Info, bool) {
	if c == nil {
		return Info{}, false
	}

	info, ok := c.models[modelID]
	if !ok {
		return Info{}, false
	}

	return cloneInfo(info), true
}

// Version returns a content-derived hash identifying the pricing data
// in this catalog. Store it alongside spending events so costs remain
// reproducible across deployments.
func (c *Catalog) Version() string {
	if c == nil {
		return ""
	}

	return c.version
}

// ErrUnknownModel is returned from Calculate when the requested model
// ID is not registered in the catalog. Surfacing this as an error
// (rather than silently pricing as zero) is deliberate: in a
// billing/logging path a miswired model ID must not be
// indistinguishable from a free call. Callers that want fail-open cost
// estimation may ignore the error and still use the returned Cost
// (which carries CatalogVersion but no breakdown).
var ErrUnknownModel = errors.New("pricing: unknown model")

// Calculate prices one model call using the catalog-registered Info
// for modelID. Resolution runs against the catalog's internal
// (immutable) Info so callers cannot forge a CatalogVersion by passing
// a mutated copy.
//
// Resolution is:
//  1. Match the most specific override whose non-empty fields all
//     match the request selector. Empty override fields are wildcards.
//  2. Resolve the highest matching context bracket of that rate card.
//  3. Multiply each disjoint llm.TokenUsage bucket by the resolved
//     rate; record per-bucket microcents in Cost.Breakdown and the sum
//     in Total.
//
// Buckets with non-zero tokens but no corresponding rate land in
// Unpriced rather than being silently folded into another bucket.
//
// Returns ErrUnknownModel when modelID is not registered. A nil usage
// is treated as an empty TokenUsage (returns zero Cost with provenance
// stamped) and is not an error.
func (c *Catalog) Calculate(modelID string, usage *llm.TokenUsage, req CalcRequest) (Cost, error) {
	cost := Cost{Breakdown: map[UsageField]int64{}}
	if c != nil {
		cost.CatalogVersion = c.version
	}

	if c == nil {
		return cost, ErrUnknownModel
	}

	info, ok := c.models[modelID]
	if !ok {
		return cost, fmt.Errorf("%w: %q", ErrUnknownModel, modelID)
	}

	if usage == nil {
		usage = &llm.TokenUsage{}
	}

	return c.calculate(info, usage, req), nil
}

func (c *Catalog) calculate(info Info, usage *llm.TokenUsage, req CalcRequest) Cost {
	cost := Cost{
		Breakdown:      make(map[UsageField]int64, 8),
		CatalogVersion: c.version,
	}

	selector := normalizeSelector(req.Selector)
	card, appliedSelector, fallbacks := resolveRateCard(info, selector)
	cost.AppliedSelector = appliedSelector
	cost.Fallbacks = fallbacks

	contextTokens := req.ContextTokens
	if contextTokens == 0 {
		// Bracket thresholds gate on context size, not billing rate: a
		// 200k prompt with 190k cached still lands in the large-
		// context bracket. BilledInputTokens sums every disjoint
		// input-side counter.
		contextTokens = int64(usage.BilledInputTokens())
	}

	rates, appliedMin := resolveBracket(card, contextTokens)
	cost.AppliedBracketMinContextTokens = appliedMin

	// Unpriced is a small ordered slice (<=8 fields, each paired at
	// most once) so a map+sort would just add allocations.
	unpriced := make([]UsageField, 0, 8)
	total := int64(0)

	pair := func(field UsageField, tokens, rate int64) {
		priced, ok := priceBucket(tokens, rate)
		if !ok {
			unpriced = append(unpriced, field)
			return
		}

		cost.Breakdown[field] = priced
		total += priced
	}

	pair(UsageFieldInput, max(int64(usage.InputTokens), 0), rates.InputPerMillion)
	pair(UsageFieldCachedInput, max(int64(usage.CachedInputTokens), 0), rates.CachedInputPerMillion)
	pair(UsageFieldCacheCreation5m, max(int64(usage.CacheCreation5mTokens), 0), rates.CacheCreation5mPerMillion)
	pair(UsageFieldCacheCreation1h, max(int64(usage.CacheCreation1hTokens), 0), rates.CacheCreation1hPerMillion)
	pair(UsageFieldCacheCreationUnknownTTL, max(int64(usage.CacheCreationUnknownTTLTokens), 0), rates.CacheCreationUnknownTTLPerMillion)
	pair(UsageFieldOutput, max(int64(usage.OutputTokens), 0), rates.OutputPerMillion)
	pair(UsageFieldReasoning, max(int64(usage.ReasoningTokens), 0), rates.OutputPerMillion)
	pair(UsageFieldToolUseInput, max(int64(usage.ToolUseInputTokens), 0), rates.InputPerMillion)

	cost.Total = total

	if len(unpriced) > 0 {
		slices.Sort(unpriced)
		cost.Unpriced = unpriced
	}

	return cost
}

// resolveRateCard reads directly from the catalog-immutable Info. No
// cloning: the returned RateCard is only consumed by resolveBracket,
// which reads Base and iterates Brackets without mutating either.
func resolveRateCard(info Info, selector Selector) (RateCard, Selector, []Fallback) {
	if len(info.Overrides) == 0 {
		return info.Default, Selector{}, nil
	}

	var (
		matched         *RateCard
		matchedSelector Selector
		bestSpecificity = -1
	)

	for i := range info.Overrides {
		override := info.Overrides[i]
		if !selectorMatches(override.Match, selector) {
			continue
		}

		specificity := selectorSpecificity(override.Match)
		if specificity > bestSpecificity {
			bestSpecificity = specificity
			matchedSelector = override.Match
			matched = &info.Overrides[i].RateCard
		}
	}

	if matched == nil {
		if selector.IsZero() {
			return info.Default, Selector{}, nil
		}

		return info.Default, Selector{}, []Fallback{{Requested: selector}}
	}

	if matchedSelector == selector {
		return *matched, matchedSelector, nil
	}

	return *matched, matchedSelector, []Fallback{{Requested: selector, Resolved: matchedSelector}}
}

func resolveBracket(card RateCard, contextTokens int64) (Rates, int64) {
	rates := card.Base
	appliedMin := int64(0)

	for _, bracket := range card.Brackets {
		if contextTokens >= bracket.MinContextTokens {
			rates = bracket.Rates
			appliedMin = bracket.MinContextTokens
		}
	}

	return rates, appliedMin
}

// priceBucket returns the microcent amount for a (tokens, rate) pair
// and whether the bucket was priced. When tokens == 0 the bucket is
// treated as priced-at-zero: no unpriced signal is useful when nothing
// was used.
func priceBucket(tokens, rate int64) (int64, bool) {
	if tokens == 0 {
		return 0, true
	}

	if rate == RateFree {
		return 0, true
	}

	if rate == 0 {
		return 0, false
	}

	return tokens * rate / 1_000_000, true
}

func cloneInfo(info Info) Info {
	return Info{
		Default:   cloneRateCard(info.Default),
		Overrides: cloneOverrides(info.Overrides),
	}
}

func cloneRateCard(card RateCard) RateCard {
	return RateCard{
		Base:     card.Base,
		Brackets: slices.Clone(card.Brackets),
	}
}

func cloneOverrides(src []Override) []Override {
	if len(src) == 0 {
		return nil
	}

	out := make([]Override, len(src))
	for i, override := range src {
		out[i] = Override{
			Match:    override.Match,
			RateCard: cloneRateCard(override.RateCard),
		}
	}

	return out
}
