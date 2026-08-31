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

package pricing_test

import (
	"errors"
	"fmt"
	"log"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

// Example_standalone shows pricing-only usage: you already have token
// counts from your own source (request logs, OTel spans, a database,
// a CSV) and want to compute costs, without using the SDK's provider
// machinery to actually send LLM requests.
//
// You still import a provider package to pick up its curated pricing
// map (openai.Catalog().PricingByID() here) and the llm package for the
// TokenUsage shape — the request/response stack is just not used.
func Example_standalone() {
	// 1. Build a catalog from the SDK's shipped pricing data. Mix
	//    providers as needed; each map is provider-scoped.
	catalog, err := pricing.NewCatalog(
		pricing.WithProvider("openai", openai.Catalog().PricingByID()),
		// pricing.WithProvider("anthropic", anthropic.Catalog().PricingByID()),
		// pricing.WithProvider("google",    google.Catalog().PricingByID()),
		// pricing.WithProvider("bedrock",   bedrock.Catalog().PricingByID()),
	)
	if err != nil {
		log.Fatal(err)
	}

	// 2. Price a call. Token counts come from wherever you have them;
	//    wrap them in an llm.TokenUsage.
	cost, err := catalog.Calculate("gpt-5", &llm.TokenUsage{
		InputTokens:       1_000_000, // 1M fresh input tokens
		OutputTokens:      100_000,   // 100k output tokens
		CachedInputTokens: 500_000,   // 500k cache-read tokens
	}, pricing.CalcRequest{})
	if err != nil {
		log.Fatal(err)
	}

	// 3. Total is int64 microcents. Divide by 100_000_000 for dollars,
	//    or by 1_000_000 for cents.
	fmt.Printf("total: $%.4f (%d microcents)\n",
		float64(cost.Total)/100_000_000,
		cost.Total,
	)
	fmt.Printf("catalog version stamped: %t\n", cost.CatalogVersion != "")

	// Output:
	// total: $2.3125 (231250000 microcents)
	// catalog version stamped: true
}

// Example_unknownModelIsAnError shows the billing-safe error path.
// Calculate does NOT silently price an unknown model as zero — the
// caller must decide how to surface the miswiring.
func Example_unknownModelIsAnError() {
	catalog, _ := pricing.NewCatalog(
		pricing.WithProvider("openai", map[string]pricing.Info{
			"gpt-5": pricing.FlatInfo(1.25, 10.00, 0.125),
		}),
	)

	_, err := catalog.Calculate("gpt-99", &llm.TokenUsage{InputTokens: 1000}, pricing.CalcRequest{})

	if errors.Is(err, pricing.ErrUnknownModel) {
		fmt.Println("unknown model — emit metric, don't bill as $0")
	}

	// Output:
	// unknown model — emit metric, don't bill as $0
}

// Example_selectorOverrides shows how to encode a premium rate card
// that only applies under specific request conditions. Resolution
// picks the most specific matching selector; unmatched requests fall
// back to Default with a Fallback recorded for audit.
func Example_selectorOverrides() {
	info := pricing.FlatInfoFromRates(
		pricing.NewRates(5.00, 25.00, 0.50).
			WithCacheCreation(6.25, 10.00, 0),
	).WithOverride(
		// 6x premium for fast-mode requests.
		pricing.Selector{Speed: llm.SpeedFast},
		pricing.RateCard{
			Base: pricing.NewRates(30.00, 150.00, 3.00).
				WithCacheCreation(37.50, 60.00, 0),
		},
	)

	catalog, err := pricing.NewCatalog(
		pricing.WithProvider("anthropic", map[string]pricing.Info{
			"claude-opus-4-6": info,
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	// A fast-mode call matches the override.
	fast, _ := catalog.Calculate("claude-opus-4-6", &llm.TokenUsage{
		InputTokens: 1_000_000,
	}, pricing.CalcRequest{
		Selector: pricing.Selector{Speed: llm.SpeedFast},
	})

	// A standard call falls back to the Default rate card.
	standard, _ := catalog.Calculate("claude-opus-4-6", &llm.TokenUsage{
		InputTokens: 1_000_000,
	}, pricing.CalcRequest{})

	fmt.Printf("fast:     $%.2f (applied=%s)\n",
		float64(fast.Total)/100_000_000,
		fast.AppliedSelector.Speed)
	fmt.Printf("standard: $%.2f (applied=default)\n",
		float64(standard.Total)/100_000_000)

	// Output:
	// fast:     $30.00 (applied=fast)
	// standard: $5.00 (applied=default)
}

// Example_contextBrackets shows context-size tiered pricing — Gemini
// and Anthropic Opus charge more above a context threshold. Brackets
// are inclusive lower bounds; Base applies below the first one.
func Example_contextBrackets() {
	info := pricing.TieredInfo(
		pricing.NewRates(1.25, 10.00, 0.125), // below threshold
		pricing.Bracket{
			MinContextTokens: 200_001,                             // > 200k tokens
			Rates:            pricing.NewRates(2.50, 15.00, 0.25), // doubled
		},
	)

	catalog, _ := pricing.NewCatalog(
		pricing.WithProvider("google", map[string]pricing.Info{
			"gemini-2.5-pro": info,
		}),
	)

	// Small call — under 200k, base rates apply.
	small, _ := catalog.Calculate("gemini-2.5-pro", &llm.TokenUsage{
		InputTokens: 100_000,
	}, pricing.CalcRequest{ContextTokens: 100_000})

	// Large call — above 200k, bracket rates apply.
	large, _ := catalog.Calculate("gemini-2.5-pro", &llm.TokenUsage{
		InputTokens: 300_000,
	}, pricing.CalcRequest{ContextTokens: 300_000})

	fmt.Printf("small: $%.4f (bracket=%d)\n",
		float64(small.Total)/100_000_000, small.AppliedBracketMinContextTokens)
	fmt.Printf("large: $%.4f (bracket=%d)\n",
		float64(large.Total)/100_000_000, large.AppliedBracketMinContextTokens)

	// Output:
	// small: $0.1250 (bracket=0)
	// large: $0.7500 (bracket=200001)
}

// Example_unpricedBuckets shows how the catalog reports usage that
// had no configured rate — useful for spotting missing cache-write
// pricing, unexpected tool-use tokens, or a catalog entry that needs
// an update.
func Example_unpricedBuckets() {
	// A minimal catalog that prices only input and output. Cache-read
	// is not configured — any cached tokens will land in Unpriced.
	catalog, _ := pricing.NewCatalog(
		pricing.WithProvider("minimal", map[string]pricing.Info{
			"m": pricing.FlatInfo(1.00, 2.00, 0), // cached=0 means "unconfigured"
		}),
	)

	cost, _ := catalog.Calculate("m", &llm.TokenUsage{
		InputTokens:       1_000,
		OutputTokens:      500,
		CachedInputTokens: 200, // non-zero but no configured rate
	}, pricing.CalcRequest{})

	fmt.Printf("total: %d microcents\n", cost.Total)
	fmt.Printf("unpriced buckets: %v\n", cost.Unpriced)

	// Output:
	// total: 200000 microcents
	// unpriced buckets: [cached_input]
}
