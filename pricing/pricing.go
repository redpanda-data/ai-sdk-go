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
// calculation. Prices are stored as int64 microcents per million tokens
// (1 cent = 1,000,000 microcents).
//
// # Why microcents?
//
// LLM providers price models with sub-cent granularity. For example, GPT-5
// Nano cached input costs $0.005/M (= 0.5 cents). Whole-cent integers would
// truncate that to 0 — a 100% error. Dollars as float64 would be the most
// readable option, but floats introduce rounding semantics into cost
// aggregation pipelines. Microcents are the smallest integer unit that
// represents all current provider prices exactly while keeping pure integer
// arithmetic end-to-end.
//
// To convert a dollar price from a provider's pricing page:
//
//	$2.50/M  → 250_000_000 microcents/M   (dollars × 100_000_000)
//	$0.005/M →     500_000 microcents/M
//
// Always add a dollar-amount comment next to each pricing value for
// readability.
package pricing

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"time"
)

// Info holds pricing for a model. Embed this in provider ModelDefinition
// structs so pricing is defined alongside capabilities.
// All prices are in microcents per million tokens.
//
// For tiered models, set only Tiers — the flat fields (InputPerMillion etc.)
// are automatically populated from the first (lowest) tier when building
// a Catalog. For flat-rate models, set the flat fields and leave Tiers nil.
type Info struct {
	InputPerMillion       int64
	OutputPerMillion      int64
	CachedInputPerMillion int64
	Tiers                 []Tier // optional context-length tiers

	// Provider-specific pricing extensions. nil when not applicable.
	Anthropic *AnthropicPricing

	// EffectiveFrom marks when this pricing became effective. Used for
	// historical rate lookups — when multiple rates exist for a model,
	// the correct rate is selected by comparing EffectiveFrom against the
	// event timestamp. Zero value means "effective since the beginning of time."
	EffectiveFrom time.Time
}

// AnthropicPricing holds Anthropic-specific pricing extensions.
//
// # Cache write pricing
//
// Anthropic charges a premium for tokens written to the prompt cache
// (cache_creation_input_tokens). The cost depends on the cache TTL:
//   - 5-minute (default): 1.25× base input price
//   - 1-hour ("ttl":"1h"):  2× base input price
//
// Other providers handle caching differently:
//   - OpenAI: cache writes are free (automatic caching, no write cost).
//   - Google: no per-token write cost; charges per-hour storage instead.
//
// # Fast mode pricing
//
// Anthropic offers a fast inference mode (speed: "fast") for select models
// (currently Opus 4.6 only). Fast mode charges a premium (6× standard rates
// for Opus 4.6). The response usage object includes a "speed" field confirming
// which speed was used. Cache write multipliers stack on top of fast mode
// pricing. Fast mode fields are zero when the model doesn't support it.
type AnthropicPricing struct {
	CacheWrite5mPerMillion int64 // 5-minute TTL write cost (default, 1.25× input)
	CacheWrite1hPerMillion int64 // 1-hour TTL write cost (extended, 2× input)

	// Fast mode (speed: "fast") pricing. Zero means fast mode not supported.
	FastInputPerMillion        int64 // fast mode input rate
	FastOutputPerMillion       int64 // fast mode output rate
	FastCachedInputPerMillion  int64 // fast mode cached input rate (0.1× fast input)
	FastCacheWrite5mPerMillion int64 // fast mode 5m cache write (1.25× fast input)
	FastCacheWrite1hPerMillion int64 // fast mode 1h cache write (2× fast input)
}

// CacheWriteRate returns the cache write rate for the given TTL string.
// Recognized values: "5m" (default) and "1h" (extended). Returns 0 for
// unrecognized TTLs.
func (a *AnthropicPricing) CacheWriteRate(ttl string) int64 {
	if a == nil {
		return 0
	}

	switch ttl {
	case "1h":
		return a.CacheWrite1hPerMillion
	default:
		return a.CacheWrite5mPerMillion
	}
}

// WithSpeed returns a copy of the Info with rates adjusted for the given speed.
// For "fast", uses Anthropic fast mode rates if the model supports it
// (FastInputPerMillion > 0). For "", "standard", or any unrecognized speed,
// returns the original Info unchanged (no copy).
//
// The returned Info can be used with CalculateCost and CacheWriteRate without
// any special handling — all rate fields are swapped to fast mode rates.
func (info *Info) WithSpeed(speed string) *Info {
	if speed != "fast" || info.Anthropic == nil || info.Anthropic.FastInputPerMillion == 0 {
		return info
	}

	cp := *info
	cp.InputPerMillion = info.Anthropic.FastInputPerMillion
	cp.OutputPerMillion = info.Anthropic.FastOutputPerMillion
	cp.CachedInputPerMillion = info.Anthropic.FastCachedInputPerMillion

	cpAnth := *info.Anthropic
	cpAnth.CacheWrite5mPerMillion = info.Anthropic.FastCacheWrite5mPerMillion
	cpAnth.CacheWrite1hPerMillion = info.Anthropic.FastCacheWrite1hPerMillion
	cp.Anthropic = &cpAnth

	return &cp
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

// Cost represents the calculated cost breakdown for a request.
// All values are in microcents.
type Cost struct {
	InputCostMicrocents      int64
	OutputCostMicrocents     int64
	CachedCostMicrocents     int64
	CacheWriteCostMicrocents int64
	TotalCostMicrocents      int64
}

// CalculateCost computes the cost for a given pricing info and token counts.
// Cost = tokens * pricePerMillion / 1_000_000.
//
// When the info has Tiers, the tier is selected based on total context size
// (inputTokens + cachedTokens). Otherwise the flat fields are used.
//
// cacheWriteTokens and cacheWriteRate handle Anthropic's cache write pricing.
// Pass 0 for both when the provider doesn't charge for cache writes (OpenAI,
// Google). For Anthropic, get the rate from Info.Anthropic.CacheWriteRate(ttl).
//
// Note: integer division truncates toward zero, under-counting by up to
// ~1 microcent per component per request. This is acceptable for cost
// reporting; if this is ever used for billing, switch to rounding:
// (tokens * rate + 500_000) / 1_000_000.
func CalculateCost(info *Info, inputTokens, outputTokens, cachedTokens, cacheWriteTokens int, cacheWriteRate int64) Cost {
	inputRate := info.InputPerMillion
	outputRate := info.OutputPerMillion
	cachedRate := info.CachedInputPerMillion

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
	cacheWrite := int64(cacheWriteTokens) * cacheWriteRate / 1_000_000

	return Cost{
		InputCostMicrocents:      input,
		OutputCostMicrocents:     output,
		CachedCostMicrocents:     cached,
		CacheWriteCostMicrocents: cacheWrite,
		TotalCostMicrocents:      input + output + cached + cacheWrite,
	}
}

// Catalog is an in-memory lookup table of model pricing.
type Catalog struct {
	models  map[string]*Info
	version string // content-derived hash for event tagging
}

// NewCatalog creates a Catalog from one or more model→pricing maps.
// For tiered models, flat fields are auto-populated from the first tier.
func NewCatalog(providers ...map[string]Info) *Catalog {
	m := make(map[string]*Info)

	for _, provider := range providers {
		for id, info := range provider {
			resolved := info
			if len(info.Tiers) > 0 {
				first := info.Tiers[0]
				resolved.InputPerMillion = first.InputPerMillion
				resolved.OutputPerMillion = first.OutputPerMillion
				resolved.CachedInputPerMillion = first.CachedInputPerMillion
			}

			m[id] = &resolved
		}
	}

	return &Catalog{models: m, version: computeVersion(m)}
}

// computeVersion produces a short content hash of the catalog for event tagging.
// Deterministic: sorted keys, fixed format.
func computeVersion(m map[string]*Info) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}

	sort.Strings(keys)

	h := sha256.New()

	for _, k := range keys {
		info := m[k]

		var cw5m, cw1h, fi, fo, fc, fcw5m, fcw1h int64
		if info.Anthropic != nil {
			cw5m = info.Anthropic.CacheWrite5mPerMillion
			cw1h = info.Anthropic.CacheWrite1hPerMillion
			fi = info.Anthropic.FastInputPerMillion
			fo = info.Anthropic.FastOutputPerMillion
			fc = info.Anthropic.FastCachedInputPerMillion
			fcw5m = info.Anthropic.FastCacheWrite5mPerMillion
			fcw1h = info.Anthropic.FastCacheWrite1hPerMillion
		}

		fmt.Fprintf(h, "%s:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d\n",
			k, info.InputPerMillion, info.OutputPerMillion,
			info.CachedInputPerMillion, cw5m, cw1h,
			fi, fo, fc, fcw5m, fcw1h)
	}

	sum := h.Sum(nil)

	return hex.EncodeToString(sum[:8]) // 16 hex chars — short but collision-resistant for catalogs
}

// Lookup returns the pricing Info for the given model ID.
func (c *Catalog) Lookup(modelID string) (*Info, bool) {
	info, ok := c.models[modelID]

	return info, ok
}

// Version returns a content-derived hash identifying this catalog's pricing
// data. Use this to tag spending events so you can determine which catalog
// version priced a given event.
func (c *Catalog) Version() string {
	return c.version
}
