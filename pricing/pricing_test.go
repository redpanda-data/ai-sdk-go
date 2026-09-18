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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const gpt5 = "gpt-5"

// Real provider keys, pinned here because package pricing cannot import
// a provider to reach its ProviderName const.
const (
	testGeminiProvider  ProviderKey = "gcp.gemini"
	testBedrockProvider ProviderKey = "aws.bedrock"
)

func TestCatalogBuilderAndLookup(t *testing.T) {
	t.Parallel()

	catalog, err := NewCatalog(
		WithProvider("openai", map[string]Info{
			gpt5: FlatInfo(0.625, 5.00, 0.125),
		}),
	)
	require.NoError(t, err)

	info, err := catalog.Lookup("openai", gpt5)
	require.NoError(t, err)
	assert.Equal(t, int64(62_500_000), info.Default.Base.InputPerMillion)
	assert.Equal(t, int64(500_000_000), info.Default.Base.OutputPerMillion)
	assert.Equal(t, int64(12_500_000), info.Default.Base.CachedInputPerMillion)

	// Lookup must return a deep copy so callers can't mutate catalog state.
	info.Default.Base.InputPerMillion = 999
	again, _ := catalog.Lookup("openai", gpt5)
	assert.Equal(t, int64(62_500_000), again.Default.Base.InputPerMillion)
}

func TestCalculate_BasicBuckets(t *testing.T) {
	t.Parallel()

	catalog, err := NewCatalog(
		WithProvider("openai", map[string]Info{
			"gpt-5": FlatInfo(2.50, 10.00, 1.25),
		}),
	)
	require.NoError(t, err)

	// llm.TokenUsage counters are disjoint, so the calculator multiplies
	// each field by its matching rate with no subset arithmetic.
	usage := &llm.TokenUsage{
		InputTokens:       1_000,
		CachedInputTokens: 200,
		OutputTokens:      500,
		ReasoningTokens:   50,
	}

	cost, err := catalog.Calculate("openai", gpt5, usage, CalcRequest{})
	require.NoError(t, err)

	assert.Equal(t, int64(250_000), cost.Breakdown[UsageFieldInput])
	assert.Equal(t, int64(25_000), cost.Breakdown[UsageFieldCachedInput])
	assert.Equal(t, int64(500_000), cost.Breakdown[UsageFieldOutput])
	assert.Equal(t, int64(50_000), cost.Breakdown[UsageFieldReasoning])
	assert.Equal(t, int64(825_000), cost.Total)
	assert.Empty(t, cost.Unpriced)
	assert.Empty(t, cost.Fallbacks)
	assert.Equal(t, catalog.Version(), cost.CatalogVersion)
}

func TestCalculate_SelectorWildcardOverride(t *testing.T) {
	t.Parallel()

	info := FlatInfoFromRates(
		NewRates(5.00, 25.00, 0.50).
			WithCacheCreation(6.25, 10.00, 0),
	).WithOverride(
		Selector{Speed: "fast"},
		RateCard{
			Base: NewRates(30.00, 150.00, 3.00).
				WithCacheCreation(37.50, 60.00, 0),
		},
	)

	catalog, err := NewCatalog(
		WithProvider("anthropic", map[string]Info{"claude-opus-4-6": info}),
	)
	require.NoError(t, err)

	cost, err := catalog.Calculate("anthropic", "claude-opus-4-6", &llm.TokenUsage{
		InputTokens:           1_000_000,
		CachedInputTokens:     100_000,
		CacheCreation5mTokens: 50_000,
		OutputTokens:          500_000,
	}, CalcRequest{
		Selector:      Selector{Speed: "FAST", Region: "US-East-1"},
		ContextTokens: 1_100_000,
	})
	require.NoError(t, err)

	assert.Equal(t, Selector{Speed: "fast"}, cost.AppliedSelector)
	require.Len(t, cost.Fallbacks, 1)
	assert.Equal(t, Fallback{
		Requested: Selector{Speed: "fast", Region: "us-east-1"},
		Resolved:  Selector{Speed: "fast"},
	}, cost.Fallbacks[0])
	assert.Equal(t, `selector "speed=fast,region=us-east-1" -> "speed=fast"`, cost.Fallbacks[0].String())

	assert.Equal(t, int64(3_000_000_000), cost.Breakdown[UsageFieldInput])
	assert.Equal(t, int64(30_000_000), cost.Breakdown[UsageFieldCachedInput])
	assert.Equal(t, int64(7_500_000_000), cost.Breakdown[UsageFieldOutput])
	assert.Equal(t, int64(187_500_000), cost.Breakdown[UsageFieldCacheCreation5m])
	assert.Equal(t, int64(10_717_500_000), cost.Total)
}

func TestCalculate_ContextTier(t *testing.T) {
	t.Parallel()

	info := TieredInfo(
		NewRates(1.25, 10.00, 0.125),
		Bracket{
			MinContextTokens: 200_001,
			Rates:            NewRates(2.50, 15.00, 0.25),
		},
	)

	catalog, err := NewCatalog(
		WithProvider(testGeminiProvider, map[string]Info{"gemini-2.5-pro": info}),
	)
	require.NoError(t, err)

	cost, err := catalog.Calculate(testGeminiProvider, "gemini-2.5-pro", &llm.TokenUsage{
		InputTokens:  150_000,
		OutputTokens: 1_000,
	}, CalcRequest{
		ContextTokens: 210_000,
	})
	require.NoError(t, err)

	assert.Equal(t, int64(37_500_000), cost.Breakdown[UsageFieldInput])
	assert.Equal(t, int64(1_500_000), cost.Breakdown[UsageFieldOutput])
	assert.Equal(t, int64(39_000_000), cost.Total)
	assert.Equal(t, int64(200_001), cost.AppliedBracketMinContextTokens)
}

// TestCalculate_ContextTierFallback_CachedDominant verifies the fallback
// uses full context-window size (not just fresh input) when ContextTokens
// is not supplied. A 210k prompt with 200k cached must still resolve to
// the >200k tier.
func TestCalculate_ContextTierFallback_CachedDominant(t *testing.T) {
	t.Parallel()

	info := TieredInfo(
		NewRates(1.25, 10.00, 0.125),
		Bracket{
			MinContextTokens: 200_001,
			Rates:            NewRates(2.50, 15.00, 0.25),
		},
	)

	catalog, err := NewCatalog(
		WithProvider(testGeminiProvider, map[string]Info{"gemini-2.5-pro": info}),
	)
	require.NoError(t, err)

	cost, err := catalog.Calculate(testGeminiProvider, "gemini-2.5-pro", &llm.TokenUsage{
		InputTokens:       10_000,
		CachedInputTokens: 200_000,
		OutputTokens:      1_000,
	}, CalcRequest{})
	require.NoError(t, err)

	// Fresh 10k * 250/M = 2_500_000; cached 200k * 25/M = 5_000_000; out 1k * 1_500/M = 1_500_000.
	assert.Equal(t, int64(200_001), cost.AppliedBracketMinContextTokens)
	assert.Equal(t, int64(2_500_000), cost.Breakdown[UsageFieldInput])
	assert.Equal(t, int64(5_000_000), cost.Breakdown[UsageFieldCachedInput])
	assert.Equal(t, int64(1_500_000), cost.Breakdown[UsageFieldOutput])
}

// TestCalculate_SelectorServiceTierNormalized verifies that selector
// ServiceTier casing/trimming does not cause a lookup miss against an
// override registered with the canonical lower-case value.
func TestCalculate_SelectorServiceTierNormalized(t *testing.T) {
	t.Parallel()

	info := FlatInfoFromRates(NewRates(5.00, 25.00, 0.50)).
		WithOverride(
			Selector{ServiceTier: llm.ServiceTierPriority},
			RateCard{Base: NewRates(10.00, 50.00, 1.00)},
		)

	catalog, err := NewCatalog(
		WithProvider("openai", map[string]Info{"gpt-5": info}),
	)
	require.NoError(t, err)

	cost, err := catalog.Calculate("openai", gpt5, &llm.TokenUsage{
		InputTokens:  1_000,
		OutputTokens: 500,
	}, CalcRequest{
		Selector: Selector{ServiceTier: "  PRIORITY  "},
	})
	require.NoError(t, err)

	assert.Equal(t, Selector{ServiceTier: llm.ServiceTierPriority}, cost.AppliedSelector)
	assert.Empty(t, cost.Fallbacks)
	assert.Equal(t, int64(1_000_000), cost.Breakdown[UsageFieldInput])
	assert.Equal(t, int64(2_500_000), cost.Breakdown[UsageFieldOutput])
}

func TestCalculate_UnpricedBuckets(t *testing.T) {
	t.Parallel()

	info := FlatInfoFromRates(NewRates(1.00, 2.00, 0))

	catalog, err := NewCatalog(
		WithProvider("test", map[string]Info{"m": info}),
	)
	require.NoError(t, err)

	cost, err := catalog.Calculate("test", "m", &llm.TokenUsage{
		InputTokens:           75,
		CachedInputTokens:     25,
		CacheCreation1hTokens: 10,
		ToolUseInputTokens:    5,
		OutputTokens:          10,
	}, CalcRequest{})
	require.NoError(t, err)

	assert.Equal(t, []UsageField{
		UsageFieldCacheCreation1h,
		UsageFieldCachedInput,
	}, cost.Unpriced)
	assert.Equal(t, int64(7_500), cost.Breakdown[UsageFieldInput])
	assert.Equal(t, int64(2_000), cost.Breakdown[UsageFieldOutput])
	assert.Equal(t, int64(500), cost.Breakdown[UsageFieldToolUseInput])
}

func TestLookup_MissingID(t *testing.T) {
	t.Parallel()

	catalog, err := NewCatalog(
		WithProvider("openai", map[string]Info{"gpt-5": FlatInfo(0.00000001, 0.00000002, 0.00000003)}),
	)
	require.NoError(t, err)

	_, err = catalog.Lookup("openai", "ghost")
	require.ErrorIs(t, err, ErrUnknownModel)

	_, err = catalog.Lookup("openai", gpt5)
	require.NoError(t, err)
}

func TestLookup_UnknownProvider(t *testing.T) {
	t.Parallel()

	catalog, err := NewCatalog(
		WithProvider("openai", map[string]Info{"gpt-5": FlatInfo(0.00000001, 0.00000002, 0.00000003)}),
	)
	require.NoError(t, err)

	// "gpt-5" exists, but not under "anthropic": the provider miss wins.
	_, err = catalog.Lookup("anthropic", gpt5)
	require.ErrorIs(t, err, ErrUnknownProvider)

	_, err = catalog.Calculate("anthropic", gpt5, &llm.TokenUsage{InputTokens: 1000}, CalcRequest{})
	require.ErrorIs(t, err, ErrUnknownProvider)
}

// TestCalculate_UnknownModelReturnsError verifies that an unknown
// model ID surfaces as ErrUnknownModel rather than silently pricing as
// zero. A miswired lookup must not be indistinguishable from a free
// call in a billing/logging path.
func TestCalculate_UnknownModelReturnsError(t *testing.T) {
	t.Parallel()

	catalog, err := NewCatalog(
		WithProvider("openai", map[string]Info{"gpt-5": FlatInfo(0.00000001, 0.00000002, 0.00000003)}),
	)
	require.NoError(t, err)

	cost, err := catalog.Calculate(
		"openai",
		"ghost",
		&llm.TokenUsage{InputTokens: 1000},
		CalcRequest{},
	)
	require.ErrorIs(t, err, ErrUnknownModel)
	assert.Contains(t, err.Error(), `"ghost"`)

	// Provenance still stamped so callers ignoring the error can at
	// least attribute the empty Cost to a specific catalog version.
	assert.Equal(t, int64(0), cost.Total)
	assert.Empty(t, cost.Breakdown)
	assert.Equal(t, catalog.Version(), cost.CatalogVersion)
}

func TestBuilder_SharedModelIDAcrossProviders(t *testing.T) {
	t.Parallel()

	catalog, err := NewCatalog(
		WithProvider("anthropic", map[string]Info{
			"claude-sonnet-4-5": FlatInfo(3.00, 15.00, 0.30),
		}),
		WithProvider(testBedrockProvider, map[string]Info{
			"claude-sonnet-4-5": FlatInfo(3.30, 16.50, 0.33),
		}),
	)
	require.NoError(t, err)

	direct, err := catalog.Lookup("anthropic", "claude-sonnet-4-5")
	require.NoError(t, err)
	bedrock, err := catalog.Lookup(testBedrockProvider, "claude-sonnet-4-5")
	require.NoError(t, err)

	assert.Equal(t, int64(300_000_000), direct.Default.Base.InputPerMillion)
	assert.Equal(t, int64(330_000_000), bedrock.Default.Base.InputPerMillion)
	assert.NotEqual(t, direct.Default.Base.InputPerMillion, bedrock.Default.Base.InputPerMillion)
}

// TestWithProvider_SameProviderInTwoPieces pins that one provider may
// register in several pieces — cloudv2 has two registration sites for one.
func TestWithProvider_SameProviderInTwoPieces(t *testing.T) {
	t.Parallel()

	cat, err := NewCatalog(
		WithProvider("openai", map[string]Info{"gpt-5": FlatInfo(0.00000001, 0.00000002, 0.00000003)}),
		WithProvider("openai", map[string]Info{"gpt-4o": FlatInfo(0.00000004, 0.00000005, 0.00000006)}),
	)
	require.NoError(t, err)

	_, err = cat.Lookup("openai", "gpt-5")
	require.NoError(t, err)
	_, err = cat.Lookup("openai", "gpt-4o")
	require.NoError(t, err)
}

func TestCatalogBuilder_Errors(t *testing.T) {
	t.Parallel()

	t.Run("duplicate within same provider", func(t *testing.T) {
		t.Parallel()

		_, err := NewCatalog(
			WithProvider("openai", map[string]Info{"m": FlatInfo(0.00000001, 0.00000002, 0.00000003)}),
			WithProvider("openai", map[string]Info{"m": FlatInfo(0.00000004, 0.00000005, 0.00000006)}),
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), `duplicate pricing for model "m"`)
	})

	t.Run("unknown override target", func(t *testing.T) {
		t.Parallel()

		_, err := NewCatalog(
			WithProvider("openai", map[string]Info{"m": FlatInfo(0.00000001, 0.00000002, 0.00000003)}),
			WithOverride("openai", "missing", FlatInfo(0.00000004, 0.00000005, 0.00000006)),
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), `override for unknown model "missing"`)
	})

	t.Run("duplicate override errors", func(t *testing.T) {
		t.Parallel()

		_, err := NewCatalog(
			WithProvider("openai", map[string]Info{"m": FlatInfo(0.00000001, 0.00000002, 0.00000003)}),
			WithOverride("openai", "m", FlatInfo(0.00000004, 0.00000005, 0.00000006)),
			WithOverride("openai", "m", FlatInfo(0.00000007, 0.00000008, 0.00000009)),
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), `duplicate override for model "m"`)
	})

	t.Run("ambiguous selectors rejected", func(t *testing.T) {
		t.Parallel()

		info := FlatInfo(0.00000001, 0.00000002, 0.00000003).
			WithOverride(Selector{Speed: "fast"}, RateCard{Base: NewRates(0.0000001, 0.0000002, 0.0000003)}).
			WithOverride(Selector{ServiceTier: llm.ServiceTierPriority}, RateCard{Base: NewRates(0.00000011, 0.00000022, 0.00000033)})

		_, err := NewCatalog(
			WithProvider("anthropic", map[string]Info{"m": info}),
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "ambiguous selectors")
	})

	t.Run("all ambiguous selector pairs reported", func(t *testing.T) {
		t.Parallel()

		// Three single-dimension overrides that all overlap pairwise at the same
		// specificity — three ambiguous pairs. Builder should surface all of
		// them in a single error so authors can fix in one pass.
		info := FlatInfo(0.00000001, 0.00000002, 0.00000003).
			WithOverride(Selector{Speed: "fast"}, RateCard{Base: NewRates(0.0000001, 0.0000002, 0.0000003)}).
			WithOverride(Selector{ServiceTier: llm.ServiceTierPriority}, RateCard{Base: NewRates(0.00000011, 0.00000022, 0.00000033)}).
			WithOverride(Selector{Region: "us-east-1"}, RateCard{Base: NewRates(0.00000012, 0.00000023, 0.00000034)})

		_, err := NewCatalog(
			WithProvider("anthropic", map[string]Info{"m": info}),
		)
		require.Error(t, err)
		assert.Equal(t, 3, strings.Count(err.Error(), "ambiguous selectors"),
			"every ambiguous pair should be reported once, got: %v", err)
	})

	t.Run("empty override selector rejected", func(t *testing.T) {
		t.Parallel()

		info := FlatInfo(0.00000001, 0.00000002, 0.00000003).
			WithOverride(Selector{}, RateCard{Base: NewRates(0.0000001, 0.0000002, 0.0000003)})

		_, err := NewCatalog(
			WithProvider("anthropic", map[string]Info{"m": info}),
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "empty Selector shadows Default")
	})

	t.Run("zero MinContextTokens tier rejected", func(t *testing.T) {
		t.Parallel()

		info := TieredInfo(
			NewRates(0.000001, 0.000002, 0.0000005),
			Bracket{MinContextTokens: 0, Rates: NewRates(0.00000101, 0.00000201, 0.00000051)},
		)

		_, err := NewCatalog(
			WithProvider(testGeminiProvider, map[string]Info{"m": info}),
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "MinContextTokens must be > 0")
	})
}

// TestCalculate_RateFreeIsPriced verifies that a Rates field set to
// RateFree prices as zero without flagging the bucket as Unpriced,
// while a literal 0 rate still surfaces in Unpriced.
func TestCalculate_RateFreeIsPriced(t *testing.T) {
	t.Parallel()

	// Cached reads explicitly free; cache creation (unset at 0) is
	// unpriced.
	rates := Rates{
		InputPerMillion:       100_000_000,
		OutputPerMillion:      200_000_000,
		CachedInputPerMillion: RateFree,
	}

	catalog, err := NewCatalog(
		WithProvider("promo", map[string]Info{"m": FlatInfoFromRates(rates)}),
	)
	require.NoError(t, err)

	cost, err := catalog.Calculate("promo", "m", &llm.TokenUsage{
		InputTokens:           10,
		CachedInputTokens:     1_000_000,
		CacheCreation5mTokens: 5,
		OutputTokens:          20,
	}, CalcRequest{})
	require.NoError(t, err)

	assert.Equal(t, int64(0), cost.Breakdown[UsageFieldCachedInput],
		"RateFree must price at zero")
	assert.NotContains(t, cost.Unpriced, UsageFieldCachedInput,
		"RateFree must NOT flag the bucket as unpriced")
	assert.Contains(t, cost.Unpriced, UsageFieldCacheCreation5m,
		"rate==0 must still surface as unpriced")
}

// TestCalculate_SelectorServiceTierAlias verifies that an override
// registered under ServiceTierDefault matches a response whose tier the
// provider reports as an alias ("standard", "auto").
func TestCalculate_SelectorServiceTierAlias(t *testing.T) {
	t.Parallel()

	info := FlatInfoFromRates(NewRates(1.00, 5.00, 0.10)).
		WithOverride(
			Selector{ServiceTier: llm.ServiceTierDefault},
			RateCard{Base: NewRates(1.11, 5.55, 0.11)},
		)

	catalog, err := NewCatalog(
		WithProvider("openai", map[string]Info{"gpt-5": info}),
	)
	require.NoError(t, err)

	for _, raw := range []string{"default", "standard", "auto", "DEFAULT", "Auto"} {
		t.Run(raw, func(t *testing.T) {
			t.Parallel()

			cost, err := catalog.Calculate("openai", gpt5, &llm.TokenUsage{
				InputTokens: 1_000,
			}, CalcRequest{
				Selector: Selector{ServiceTier: llm.ServiceTier(raw)},
			})
			require.NoError(t, err)

			assert.Equal(t, Selector{ServiceTier: llm.ServiceTierDefault}, cost.AppliedSelector,
				"alias %q must resolve to the Default override", raw)
			assert.Equal(t, int64(111_000), cost.Breakdown[UsageFieldInput])
		})
	}
}

// TestCatalogVersion_OrderStable verifies two catalogs that differ only
// in the append order of semantically equivalent overrides hash
// identically.
func TestCatalogVersion_OrderStable(t *testing.T) {
	t.Parallel()

	// Two non-overlapping overrides (different Speed values) so the
	// ambiguity check doesn't fire. The point is only that append
	// order should not change the hash.
	infoA := FlatInfoFromRates(NewRates(1.00, 2.00, 0.10)).
		WithOverride(Selector{Speed: "fast"}, RateCard{Base: NewRates(0.000003, 0.000004, 0.0000005)}).
		WithOverride(Selector{Speed: "slow"}, RateCard{Base: NewRates(0.000005, 0.000006, 0.0000007)})

	infoB := FlatInfoFromRates(NewRates(1.00, 2.00, 0.10)).
		WithOverride(Selector{Speed: "slow"}, RateCard{Base: NewRates(0.000005, 0.000006, 0.0000007)}).
		WithOverride(Selector{Speed: "fast"}, RateCard{Base: NewRates(0.000003, 0.000004, 0.0000005)})

	a, err := NewCatalog(
		WithProvider("p", map[string]Info{"m": infoA}),
	)
	require.NoError(t, err)

	b, err := NewCatalog(
		WithProvider("p", map[string]Info{"m": infoB}),
	)
	require.NoError(t, err)

	assert.Equal(t, a.Version(), b.Version(),
		"override append order must not affect Version()")
}

func TestCatalogVersion_DeterministicAndSensitive(t *testing.T) {
	t.Parallel()

	baseA, err := NewCatalog(
		WithProvider("openai", map[string]Info{"m": FlatInfo(0.000001, 0.000002, 0.0000005)}),
	)
	require.NoError(t, err)

	baseB, err := NewCatalog(
		WithProvider("openai", map[string]Info{"m": FlatInfo(0.000001, 0.000002, 0.0000005)}),
	)
	require.NoError(t, err)

	changed, err := NewCatalog(
		WithProvider("openai", map[string]Info{
			"m": TieredInfo(
				NewRates(0.000001, 0.000002, 0.0000005),
				Bracket{MinContextTokens: 1000, Rates: NewRates(0.00000101, 0.00000201, 0.00000051)},
			),
		}),
	)
	require.NoError(t, err)

	assert.Equal(t, baseA.Version(), baseB.Version())
	assert.NotEqual(t, baseA.Version(), changed.Version())
	assert.Len(t, baseA.Version(), 16)
}

// fakeSource stands in for a provider catalog, which pricing cannot import.
type fakeSource struct {
	provider llm.ProviderID
	prices   map[string]Info
}

func (s fakeSource) Provider() llm.ProviderID     { return s.provider }
func (s fakeSource) PricingByID() map[string]Info { return s.prices }

func TestWithSource_RegistersUnderProviderName(t *testing.T) {
	t.Parallel()

	src := fakeSource{
		provider: "gcp.vertex",
		prices:   map[string]Info{"claude-sonnet-5": FlatInfo(2.00, 10.00, 0.20)},
	}

	catalog, err := NewCatalog(WithSource(src))
	require.NoError(t, err)

	got, err := catalog.Lookup("gcp.vertex", "claude-sonnet-5")
	require.NoError(t, err)
	assert.Equal(t, int64(200_000_000), got.Default.Base.InputPerMillion)

	// Under an unregistered provider the miss is the provider, not the model.
	_, err = catalog.Lookup("anthropic", "claude-sonnet-5")
	require.ErrorIs(t, err, ErrUnknownProvider)
}

func TestWithSource_EmptyProviderFailsBuild(t *testing.T) {
	t.Parallel()

	src := fakeSource{
		provider: "",
		prices:   map[string]Info{"m": FlatInfo(1.00, 2.00, 0.10)},
	}

	_, err := NewCatalog(WithSource(src))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty provider name")
}

func TestCatalogVersion_FoldsInProvider(t *testing.T) {
	t.Parallel()

	underOpenAI, err := NewCatalog(
		WithProvider("openai", map[string]Info{"m": FlatInfo(0.000001, 0.000002, 0.0000005)}),
	)
	require.NoError(t, err)

	underAnthropic, err := NewCatalog(
		WithProvider("anthropic", map[string]Info{"m": FlatInfo(0.000001, 0.000002, 0.0000005)}),
	)
	require.NoError(t, err)

	assert.NotEqual(t, underOpenAI.Version(), underAnthropic.Version(),
		"identical model+rates under a different provider must change the version")
}

func TestWithProvider_EmptyKeyFailsBuild(t *testing.T) {
	t.Parallel()

	_, err := NewCatalog(WithProvider("", map[string]Info{"m": FlatInfo(1.00, 2.00, 0.10)}))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty provider key")
}

func TestWithSource_NilSourceFailsBuild(t *testing.T) {
	t.Parallel()

	_, err := NewCatalog(WithSource(nil))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil source")
}

// TestEmptyProviderMapStaysKnown guards against deriving the known-provider
// set from surviving models instead of from the registration itself.
func TestEmptyProviderMapStaysKnown(t *testing.T) {
	t.Parallel()

	cat, err := NewCatalog(WithProvider("openai", map[string]Info{}))
	require.NoError(t, err)

	_, err = cat.Lookup("openai", "gpt-5")
	require.ErrorIs(t, err, ErrUnknownModel,
		"a known provider with no models must miss on the model, not the provider")
	require.NotErrorIs(t, err, ErrUnknownProvider)
}

func TestWithOverride_LandsScopedToProvider(t *testing.T) {
	t.Parallel()

	cat, err := NewCatalog(
		WithProvider("anthropic", map[string]Info{"m": FlatInfo(3.00, 15.00, 0.30)}),
		WithProvider(testBedrockProvider, map[string]Info{"m": FlatInfo(3.00, 15.00, 0.30)}),
		WithOverride(testBedrockProvider, "m", FlatInfo(9.00, 45.00, 0.90)),
	)
	require.NoError(t, err)

	overridden, err := cat.Lookup(testBedrockProvider, "m")
	require.NoError(t, err)
	assert.Equal(t, int64(900_000_000), overridden.Default.Base.InputPerMillion,
		"WithOverride must replace the bedrock rate card")

	untouched, err := cat.Lookup("anthropic", "m")
	require.NoError(t, err)
	assert.Equal(t, int64(300_000_000), untouched.Default.Base.InputPerMillion,
		"the same model ID under another provider must be untouched by the override")
}

// TestCatalogVersion_MultiProviderOrderStable: drop the model tie-break
// from computeVersion's sort and two providers sharing a model ID hash in
// map order, so CatalogVersion flips per process.
func TestCatalogVersion_MultiProviderOrderStable(t *testing.T) {
	t.Parallel()

	build := func() *Catalog {
		c, err := NewCatalog(
			WithProvider("anthropic", map[string]Info{
				"m": FlatInfo(3.00, 15.00, 0.30),
				"n": FlatInfo(1.00, 5.00, 0.10),
			}),
			WithProvider(testBedrockProvider, map[string]Info{
				"m": FlatInfo(3.00, 15.00, 0.30),
				"n": FlatInfo(1.00, 5.00, 0.10),
			}),
		)
		require.NoError(t, err)

		return c
	}

	first := build().Version()
	for range 8 {
		assert.Equal(t, first, build().Version(),
			"multi-provider catalog version must not depend on map iteration order")
	}
}

func TestNilCatalog_ReportsUnknownProvider(t *testing.T) {
	t.Parallel()

	var cat *Catalog

	_, err := cat.Lookup("openai", gpt5)
	require.ErrorIs(t, err, ErrUnknownProvider)

	cost, err := cat.Calculate("openai", gpt5, &llm.TokenUsage{InputTokens: 1000}, CalcRequest{})
	require.ErrorIs(t, err, ErrUnknownProvider)
	assert.Equal(t, int64(0), cost.Total)
	assert.Empty(t, cost.CatalogVersion)
}
