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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelPricing_RateAt(t *testing.T) {
	now := time.Date(2025, 6, 1, 0, 0, 0, 0, time.UTC)
	oldDate := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	mp := ModelPricing{
		ModelID: "gpt-4o",
		Rates: []Rate{
			{EffectiveFrom: now, InputPerMillion: 250_000_000, OutputPerMillion: 1_000_000_000, CachedInputPerMillion: 125_000_000},
			{EffectiveFrom: oldDate, InputPerMillion: 500_000_000, OutputPerMillion: 1_500_000_000, CachedInputPerMillion: 250_000_000},
		},
	}

	t.Run("current rate", func(t *testing.T) {
		rate := mp.RateAt(now.Add(24 * time.Hour))
		require.NotNil(t, rate)
		assert.Equal(t, int64(250_000_000), rate.InputPerMillion)
	})

	t.Run("historical rate", func(t *testing.T) {
		rate := mp.RateAt(oldDate.Add(24 * time.Hour))
		require.NotNil(t, rate)
		assert.Equal(t, int64(500_000_000), rate.InputPerMillion)
	})

	t.Run("exact boundary", func(t *testing.T) {
		rate := mp.RateAt(now)
		require.NotNil(t, rate)
		assert.Equal(t, int64(250_000_000), rate.InputPerMillion)
	})

	t.Run("before all rates returns nil", func(t *testing.T) {
		rate := mp.RateAt(oldDate.Add(-1 * time.Second))
		assert.Nil(t, rate)
	})

	t.Run("empty rates returns nil", func(t *testing.T) {
		empty := ModelPricing{ModelID: "empty"}
		assert.Nil(t, empty.RateAt(now))
	})
}

func TestModelPricing_CurrentRate(t *testing.T) {
	t.Run("returns first rate", func(t *testing.T) {
		mp := ModelPricing{
			ModelID: "gpt-4o",
			Rates: []Rate{
				{InputPerMillion: 250_000_000},
				{InputPerMillion: 500_000_000},
			},
		}
		rate := mp.CurrentRate()
		require.NotNil(t, rate)
		assert.Equal(t, int64(250_000_000), rate.InputPerMillion)
	})

	t.Run("empty rates returns nil", func(t *testing.T) {
		mp := ModelPricing{ModelID: "empty"}
		assert.Nil(t, mp.CurrentRate())
	})
}

func TestCatalog_Lookup(t *testing.T) {
	catalog := NewCatalog([]ModelPricing{
		{
			ModelID: "gpt-4o",
			Rates: []Rate{
				{InputPerMillion: 250_000_000, OutputPerMillion: 1_000_000_000, CachedInputPerMillion: 125_000_000},
			},
		},
	})

	t.Run("found", func(t *testing.T) {
		mp, ok := catalog.Lookup("gpt-4o")
		assert.True(t, ok)
		require.NotNil(t, mp)
		assert.Equal(t, "gpt-4o", mp.ModelID)
	})

	t.Run("not found", func(t *testing.T) {
		mp, ok := catalog.Lookup("nonexistent")
		assert.False(t, ok)
		assert.Nil(t, mp)
	})
}

func TestCalculateCost(t *testing.T) {
	rate := &Rate{
		InputPerMillion:       250_000_000,   // $2.50 per million
		OutputPerMillion:      1_000_000_000, // $10.00 per million
		CachedInputPerMillion: 125_000_000,   // $1.25 per million
	}

	t.Run("basic calculation", func(t *testing.T) {
		cost := CalculateCost(rate, 1000, 500, 200)

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
		cost := CalculateCost(rate, 0, 0, 0)
		assert.Equal(t, int64(0), cost.TotalCostMicrocents)
	})
}

func TestCalculateCost_Tiered(t *testing.T) {
	rate := &Rate{
		InputPerMillion:       125_000_000,   // $1.25 per M (default = low tier)
		OutputPerMillion:      1_000_000_000, // $10.00 per M
		CachedInputPerMillion: 31_250_000,    // $0.3125 per M
		Tiers: []Tier{
			{MaxInputTokens: 200_000, InputPerMillion: 125_000_000, OutputPerMillion: 1_000_000_000, CachedInputPerMillion: 31_250_000},
			{MaxInputTokens: 0, InputPerMillion: 250_000_000, OutputPerMillion: 1_500_000_000, CachedInputPerMillion: 62_500_000},
		},
	}

	t.Run("below threshold uses low tier", func(t *testing.T) {
		cost := CalculateCost(rate, 100_000, 1000, 0)
		assert.Equal(t, int64(12_500_000), cost.InputCostMicrocents)
		assert.Equal(t, int64(1_000_000), cost.OutputCostMicrocents)
	})

	t.Run("at threshold uses low tier", func(t *testing.T) {
		cost := CalculateCost(rate, 200_000, 1000, 0)
		assert.Equal(t, int64(25_000_000), cost.InputCostMicrocents)
		assert.Equal(t, int64(1_000_000), cost.OutputCostMicrocents)
	})

	t.Run("above threshold uses high tier", func(t *testing.T) {
		cost := CalculateCost(rate, 200_001, 1000, 0)
		assert.Equal(t, int64(50_000_250), cost.InputCostMicrocents)
		assert.Equal(t, int64(1_500_000), cost.OutputCostMicrocents)
	})

	t.Run("cached tokens count toward context size", func(t *testing.T) {
		// 150k input + 60k cached = 210k context → high tier
		cost := CalculateCost(rate, 150_000, 1000, 60_000)
		assert.Equal(t, int64(37_500_000), cost.InputCostMicrocents)
		assert.Equal(t, int64(3_750_000), cost.CachedCostMicrocents)
	})

	t.Run("default rate fields match low tier", func(t *testing.T) {
		assert.Equal(t, int64(125_000_000), rate.InputPerMillion)
		assert.Equal(t, int64(1_000_000_000), rate.OutputPerMillion)
		assert.Equal(t, int64(31_250_000), rate.CachedInputPerMillion)
	})
}
