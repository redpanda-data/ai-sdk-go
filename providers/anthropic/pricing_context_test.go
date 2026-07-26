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

package anthropic

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// TestNative1MModelsStayFlatAcrossContextWindow guards the models Anthropic
// currently documents as flat across their native 1M context windows.
func TestNative1MModelsStayFlatAcrossContextWindow(t *testing.T) {
	t.Parallel()

	for _, id := range []string{
		ModelClaudeFable5,
		ModelClaudeOpus46,
		ModelClaudeOpus47,
		ModelClaudeOpus48,
		ModelClaudeSonnet46,
		ModelClaudeSonnet5,
	} {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			def, ok := supportedModels[id]
			require.True(t, ok, "native 1M model %s must remain registered", id)
			assert.Emptyf(t, def.Pricing.Default.Brackets,
				"native 1M model %s must stay flat across its full context window", id)

			for _, ov := range def.Pricing.Overrides {
				assert.Emptyf(t, ov.RateCard.Brackets,
					"model %s override %s must stay flat across its full context window",
					id, fmt.Sprintf("%+v", ov.Match))
			}
		})
	}
}

func TestNativeCatalogHasNoContextBrackets(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		assert.Emptyf(t, def.Pricing.Default.Brackets,
			"native model %s must not have a context pricing bracket", id)

		for _, ov := range def.Pricing.Overrides {
			assert.Emptyf(t, ov.RateCard.Brackets,
				"model %s override %s must not have a context pricing bracket",
				id, fmt.Sprintf("%+v", ov.Match))
		}
	}
}

// TestNative1MContextPricingStaysFlatEndToEnd proves a large Opus 4.8 request
// uses the same rate card as a small request, including fast mode.
func TestNative1MContextPricingStaysFlatEndToEnd(t *testing.T) {
	t.Parallel()

	cat, err := pricing.NewCatalog(pricing.WithProvider("anthropic", ModelPricing()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 100_000, OutputTokens: 1_000}

	totals := make([]int64, 0, 2)

	for _, req := range []pricing.CalcRequest{
		{},
		{Selector: pricing.Selector{Speed: SpeedFast}},
	} {
		belowReq := req
		belowReq.ContextTokens = 100_000
		below, err := cat.Calculate(ModelClaudeOpus48, usage, belowReq)
		require.NoError(t, err)

		aboveReq := req
		aboveReq.ContextTokens = 900_000
		above, err := cat.Calculate(ModelClaudeOpus48, usage, aboveReq)
		require.NoError(t, err)

		assert.Equal(t, int64(0), below.AppliedBracketMinContextTokens)
		assert.Equal(t, int64(0), above.AppliedBracketMinContextTokens)
		assert.Equal(t, below.Total, above.Total,
			"identical usage must cost the same across the native 1M window")
		assert.Equal(t, below.Breakdown, above.Breakdown,
			"rate breakdown must stay unchanged across the native 1M window")
		totals = append(totals, above.Total)
	}

	require.Len(t, totals, 2)
	assert.Greater(t, totals[1], totals[0], "fast selector must use the premium rate card")
}
