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

import "time"

// Epoch is the default effective date for all initial pricing entries.
var Epoch = time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

// FlatModel creates a ModelPricing with flat (non-tiered) pricing.
// All prices are in microcents per million tokens.
func FlatModel(id string, input, output, cached int64) ModelPricing {
	return ModelPricing{
		ModelID: id,
		Rates: []Rate{
			{
				EffectiveFrom:         Epoch,
				InputPerMillion:       input,
				OutputPerMillion:      output,
				CachedInputPerMillion: cached,
			},
		},
	}
}

// TieredModel creates a ModelPricing with two context-length tiers.
// The default tier (accessible via Rate.InputPerMillion etc.) is always the
// lower tier, so callers that don't care about tiers get the common-case price.
func TieredModel(id string, threshold int64, lowInput, lowOutput, lowCached, highInput, highOutput, highCached int64) ModelPricing {
	return ModelPricing{
		ModelID: id,
		Rates: []Rate{
			{
				EffectiveFrom:         Epoch,
				InputPerMillion:       lowInput,
				OutputPerMillion:      lowOutput,
				CachedInputPerMillion: lowCached,
				Tiers: []Tier{
					{MaxInputTokens: threshold, InputPerMillion: lowInput, OutputPerMillion: lowOutput, CachedInputPerMillion: lowCached},
					{MaxInputTokens: 0, InputPerMillion: highInput, OutputPerMillion: highOutput, CachedInputPerMillion: highCached},
				},
			},
		},
	}
}
