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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCatalog_Lookup(t *testing.T) {
	t.Parallel()

	catalog := NewCatalog(map[string]Info{
		"gpt-4o": {
			InputPerMillion:       250_000_000,
			OutputPerMillion:      1_000_000_000,
			CachedInputPerMillion: 125_000_000,
		},
	})

	t.Run("found", func(t *testing.T) {
		t.Parallel()

		info, ok := catalog.Lookup("gpt-4o")
		assert.True(t, ok)
		require.NotNil(t, info)
		assert.Equal(t, int64(250_000_000), info.InputPerMillion)
	})

	t.Run("not found", func(t *testing.T) {
		t.Parallel()

		mp, ok := catalog.Lookup("nonexistent")
		assert.False(t, ok)
		assert.Nil(t, mp)
	})
}

func TestCalculateCost(t *testing.T) {
	t.Parallel()

	info := &Info{
		InputPerMillion:       250_000_000,   // $2.50 per million
		OutputPerMillion:      1_000_000_000, // $10.00 per million
		CachedInputPerMillion: 125_000_000,   // $1.25 per million
	}

	t.Run("basic calculation", func(t *testing.T) {
		t.Parallel()

		cost := CalculateCost(info, 1000, 500, 200, 0, 0)

		// input: 1000 * 250_000_000 / 1_000_000 = 250_000
		assert.Equal(t, int64(250_000), cost.InputCostMicrocents)
		// output: 500 * 1_000_000_000 / 1_000_000 = 500_000
		assert.Equal(t, int64(500_000), cost.OutputCostMicrocents)
		// cached: 200 * 125_000_000 / 1_000_000 = 25_000
		assert.Equal(t, int64(25_000), cost.CachedCostMicrocents)
		// total
		assert.Equal(t, int64(775_000), cost.TotalCostMicrocents)
	})

	t.Run("zero tokens", func(t *testing.T) {
		t.Parallel()

		cost := CalculateCost(info, 0, 0, 0, 0, 0)
		assert.Equal(t, int64(0), cost.TotalCostMicrocents)
	})
}

func TestCalculateCost_WithCacheWrite(t *testing.T) {
	t.Parallel()

	info := &Info{
		InputPerMillion:       300_000_000,   // $3.00/M
		OutputPerMillion:      1_500_000_000, // $15.00/M
		CachedInputPerMillion: 30_000_000,    // $0.30/M
		Anthropic: &AnthropicPricing{
			CacheWrite5mPerMillion: 375_000_000, // $3.75/M (1.25× input)
			CacheWrite1hPerMillion: 600_000_000, // $6.00/M (2× input)
		},
	}

	t.Run("5m TTL", func(t *testing.T) {
		t.Parallel()

		rate := info.Anthropic.CacheWriteRate("5m")
		cost := CalculateCost(info, 1000, 500, 200, 300, rate)

		// cache write 5m: 300 * 375_000_000 / 1_000_000 = 112_500
		assert.Equal(t, int64(112_500), cost.CacheWriteCostMicrocents)
	})

	t.Run("1h TTL", func(t *testing.T) {
		t.Parallel()

		rate := info.Anthropic.CacheWriteRate("1h")
		cost := CalculateCost(info, 1000, 500, 200, 300, rate)

		// cache write 1h: 300 * 600_000_000 / 1_000_000 = 180_000
		assert.Equal(t, int64(180_000), cost.CacheWriteCostMicrocents)
	})

	t.Run("no cache write rate means zero cost", func(t *testing.T) {
		t.Parallel()

		cost := CalculateCost(info, 1000, 500, 200, 300, 0)
		assert.Equal(t, int64(0), cost.CacheWriteCostMicrocents)
	})
}

func TestAnthropicPricing_CacheWriteRate_Nil(t *testing.T) {
	t.Parallel()

	// OpenAI model — no Anthropic pricing.
	info := &Info{
		InputPerMillion:  250_000_000,
		OutputPerMillion: 1_000_000_000,
	}

	assert.Equal(t, int64(0), info.Anthropic.CacheWriteRate("5m"))
	assert.Equal(t, int64(0), info.Anthropic.CacheWriteRate("1h"))
}

func TestCalculateCost_Tiered(t *testing.T) {
	t.Parallel()

	info := &Info{
		InputPerMillion:       125_000_000,   // $1.25 per M (default = low tier)
		OutputPerMillion:      1_000_000_000, // $10.00 per M
		CachedInputPerMillion: 31_250_000,    // $0.3125 per M
		Tiers: []Tier{
			{MaxInputTokens: 200_000, InputPerMillion: 125_000_000, OutputPerMillion: 1_000_000_000, CachedInputPerMillion: 31_250_000},
			{MaxInputTokens: 0, InputPerMillion: 250_000_000, OutputPerMillion: 1_500_000_000, CachedInputPerMillion: 62_500_000},
		},
	}

	t.Run("below threshold uses low tier", func(t *testing.T) {
		t.Parallel()

		cost := CalculateCost(info, 100_000, 1000, 0, 0, 0)
		assert.Equal(t, int64(12_500_000), cost.InputCostMicrocents)
		assert.Equal(t, int64(1_000_000), cost.OutputCostMicrocents)
	})

	t.Run("at threshold uses low tier", func(t *testing.T) {
		t.Parallel()

		cost := CalculateCost(info, 200_000, 1000, 0, 0, 0)
		assert.Equal(t, int64(25_000_000), cost.InputCostMicrocents)
		assert.Equal(t, int64(1_000_000), cost.OutputCostMicrocents)
	})

	t.Run("above threshold uses high tier", func(t *testing.T) {
		t.Parallel()

		cost := CalculateCost(info, 200_001, 1000, 0, 0, 0)
		assert.Equal(t, int64(50_000_250), cost.InputCostMicrocents)
		assert.Equal(t, int64(1_500_000), cost.OutputCostMicrocents)
	})

	t.Run("cached tokens count toward context size", func(t *testing.T) {
		t.Parallel()

		// 150k input + 60k cached = 210k context → high tier
		cost := CalculateCost(info, 150_000, 1000, 60_000, 0, 0)
		assert.Equal(t, int64(37_500_000), cost.InputCostMicrocents)
		assert.Equal(t, int64(3_750_000), cost.CachedCostMicrocents)
	})
}

func TestNewCatalog_TieredAutoPopulatesFlat(t *testing.T) {
	t.Parallel()

	catalog := NewCatalog(map[string]Info{
		"tiered-model": {
			// Flat fields intentionally left at zero — NewCatalog should
			// auto-populate them from the first tier.
			Tiers: []Tier{
				{MaxInputTokens: 200_000, InputPerMillion: 125_000_000, OutputPerMillion: 1_000_000_000, CachedInputPerMillion: 12_500_000},
				{MaxInputTokens: 0, InputPerMillion: 250_000_000, OutputPerMillion: 1_500_000_000, CachedInputPerMillion: 25_000_000},
			},
		},
	})

	info, ok := catalog.Lookup("tiered-model")
	require.True(t, ok)

	// Flat fields should be auto-populated from first tier.
	assert.Equal(t, int64(125_000_000), info.InputPerMillion)
	assert.Equal(t, int64(1_000_000_000), info.OutputPerMillion)
	assert.Equal(t, int64(12_500_000), info.CachedInputPerMillion)
}

func TestNewCatalog_MultipleProviders(t *testing.T) {
	t.Parallel()

	catalog := NewCatalog(
		map[string]Info{"model-a": {InputPerMillion: 100, OutputPerMillion: 200}},
		map[string]Info{"model-b": {InputPerMillion: 300, OutputPerMillion: 400}},
	)

	a, ok := catalog.Lookup("model-a")
	require.True(t, ok)
	assert.Equal(t, int64(100), a.InputPerMillion)

	b, ok := catalog.Lookup("model-b")
	require.True(t, ok)
	assert.Equal(t, int64(300), b.InputPerMillion)
}

func TestCatalog_Version(t *testing.T) {
	t.Parallel()

	t.Run("deterministic", func(t *testing.T) {
		t.Parallel()

		c1 := NewCatalog(map[string]Info{"m": {InputPerMillion: 100}})
		c2 := NewCatalog(map[string]Info{"m": {InputPerMillion: 100}})
		assert.Equal(t, c1.Version(), c2.Version())
	})

	t.Run("changes when pricing changes", func(t *testing.T) {
		t.Parallel()

		c1 := NewCatalog(map[string]Info{"m": {InputPerMillion: 100}})
		c2 := NewCatalog(map[string]Info{"m": {InputPerMillion: 200}})
		assert.NotEqual(t, c1.Version(), c2.Version())
	})

	t.Run("non-empty", func(t *testing.T) {
		t.Parallel()

		c := NewCatalog(map[string]Info{"m": {InputPerMillion: 100}})
		assert.NotEmpty(t, c.Version())
		assert.Len(t, c.Version(), 16) // 8 bytes hex-encoded
	})
}
