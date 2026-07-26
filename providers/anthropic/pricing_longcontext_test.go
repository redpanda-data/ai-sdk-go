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

// TestAllModelsStayFlatAcrossContextWindow guards Anthropic's current pricing
// contract: every cataloged model is billed at its base rate across its full
// advertised context window. Claude 4.6 and later include the native 1M window
// without the legacy >200K premium.
func TestAllModelsStayFlatAcrossContextWindow(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			assert.Emptyf(t, def.Pricing.Default.Brackets,
				"model %s must stay flat across its full context window", id)

			for _, ov := range def.Pricing.Overrides {
				assert.Emptyf(t, ov.RateCard.Brackets,
					"model %s override %s must stay flat across its full context window",
					id, fmt.Sprintf("%+v", ov.Match))
			}
		})
	}
}

// TestNative1MContextPricingStaysFlatEndToEnd proves a large Opus 4.8 request
// uses the same rate card as a small request, including fast mode.
func TestNative1MContextPricingStaysFlatEndToEnd(t *testing.T) {
	t.Parallel()

	cat, err := pricing.NewCatalog(pricing.WithProvider("anthropic", ModelPricing()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 100_000, OutputTokens: 1_000}

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
	}
}
