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
	"math"
	"slices"
)

// RateFree is a sentinel indicating that a rate is intentionally zero —
// e.g. a provider promotion that makes cached reads free for a period.
// A literal 0 in Rates means "no rate configured" and routes any
// non-zero tokens into Cost.Unpriced; RateFree prices at exactly zero
// without the Unpriced signal. Use it sparingly and only when the
// provider has actually published a $0 rate.
//
// RateFree is untyped so it can be passed as either int64 (to struct
// literals) or float64 (to dollar-valued constructors) without casting.
const RateFree = -1

// microcentsPerDollar is the multiplier between USD-per-million-tokens
// (the rate a provider publishes) and the int64 microcent-per-million
// representation we store.
const microcentsPerDollar = 100_000_000

// usdToMicrocents converts a USD-per-million-tokens rate to the
// int64 microcent-per-million form used internally. Rounding to the
// nearest microcent means any published rate with up to 8 decimal
// places converts exactly. The RateFree sentinel passes through
// unchanged.
func usdToMicrocents(usd float64) int64 {
	if usd == RateFree {
		return RateFree
	}

	return int64(math.Round(usd * microcentsPerDollar))
}

// Rates is the leaf rate card for one resolved pricing context: the
// per-million microcent multiplier for each billable UsageField.
//
// Fields are grouped by how widely providers support them. There is no
// dedicated reasoning or tool-use rate: reasoning is billed at output
// rates and server-side tool prompt expansion is billed at input rates,
// so those buckets reuse the base rates. Adding new buckets (e.g.
// per-modality rates) should flow through UsageField additions rather
// than by growing this struct indefinitely.
//
// A field left at literal 0 is "unpriced": any non-zero tokens in that
// bucket surface in Cost.Unpriced. To model a published $0 rate, use
// the RateFree sentinel.
type Rates struct {
	// Core inference rates. Every provider charges at these rates;
	// leaving either at zero on a catalog entry causes non-zero usage
	// to fall into Cost.Unpriced.
	InputPerMillion  int64
	OutputPerMillion int64

	// Prompt-cache reads. Supported by OpenAI, Anthropic, Google, and
	// Anthropic-on-Bedrock; providers without prompt caching leave
	// this at zero and emit no cached-input tokens.
	CachedInputPerMillion int64

	// Prompt-cache writes. Anthropic-family providers expose distinct rates
	// per TTL; OpenAI exposes an aggregate cache-write count.
	// CacheCreationUnknownTTL is the fallback bucket when the provider does
	// not report a TTL. All three stay zero on providers that do not bill
	// cache writes separately.
	CacheCreation5mPerMillion         int64
	CacheCreation1hPerMillion         int64
	CacheCreationUnknownTTLPerMillion int64
}

// NewRates is a convenience constructor for the common flat-rate case
// where a model has standard input, output, and cached-input prices
// with no separate cache-write pricing.
//
// Arguments are USD-per-million-tokens (the format providers publish).
// A rate of 5.00 means $5.00 per million input tokens. Pass RateFree
// (untyped -1) to mark a bucket as intentionally free.
func NewRates(inputUSD, outputUSD, cachedUSD float64) Rates {
	return Rates{
		InputPerMillion:       usdToMicrocents(inputUSD),
		OutputPerMillion:      usdToMicrocents(outputUSD),
		CachedInputPerMillion: usdToMicrocents(cachedUSD),
	}
}

// WithCacheCreation returns a copy of the Rates with cache-write
// pricing filled in. Arguments are USD-per-million-tokens, matching
// the NewRates convention.
func (r Rates) WithCacheCreation(ttl5mUSD, ttl1hUSD, unknownTTLUSD float64) Rates {
	r.CacheCreation5mPerMillion = usdToMicrocents(ttl5mUSD)
	r.CacheCreation1hPerMillion = usdToMicrocents(ttl1hUSD)
	r.CacheCreationUnknownTTLPerMillion = usdToMicrocents(unknownTTLUSD)

	return r
}

// Bracket overrides a RateCard once the request crosses a context-size
// threshold.
//
// MinContextTokens is an inclusive lower bound and must be > 0. A zero
// threshold would shadow RateCard.Base, so it is rejected at Build
// time.
type Bracket struct {
	MinContextTokens int64
	Rates            Rates
}

// RateCard groups the base Rates for one resolved pricing context with
// any context-size Brackets that refine them.
//
// Every selector match resolves to one RateCard. Within that card, the
// base Rates apply below the first bracket threshold; each Bracket
// replaces the base Rates once the request's context size crosses its
// MinContextTokens.
type RateCard struct {
	Base     Rates
	Brackets []Bracket
}

// FlatInfo constructs an Info for a model with a single flat rate card
// and no selector-specific overrides. Arguments are USD-per-million-
// tokens, matching NewRates.
func FlatInfo(inputUSD, outputUSD, cachedUSD float64) Info {
	return FlatInfoFromRates(NewRates(inputUSD, outputUSD, cachedUSD))
}

// FlatInfoFromRates constructs an Info for a model with a single flat
// rate card.
func FlatInfoFromRates(rates Rates) Info {
	return Info{Default: RateCard{Base: rates}}
}

// TieredInfo constructs an Info whose default rate card has explicit
// context-size brackets above the provided base rates.
func TieredInfo(base Rates, brackets ...Bracket) Info {
	return Info{Default: RateCard{Base: base, Brackets: slices.Clone(brackets)}}
}
