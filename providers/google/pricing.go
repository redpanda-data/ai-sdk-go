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

package google

import "github.com/redpanda-data/ai-sdk-go/pricing"

// DefaultPricing returns pricing for all supported Google Gemini models.
// Prices are in microcents per million tokens.
//
// Source: https://ai.google.dev/gemini-api/docs/pricing (as of 2026-04)
func DefaultPricing() []pricing.ModelPricing {
	return []pricing.ModelPricing{
		// ── Gemini 3.x Preview ──────────────────────────────────
		// Gemini 3.1 Pro: tiered by context length (<=200k vs >200k tokens).
		pricing.TieredModel(ModelGemini31ProPreview, 200_000,
			200_000_000, 1_200_000_000, 20_000_000, // <=200k: $2.00/$12.00/$0.20 per M
			400_000_000, 1_800_000_000, 40_000_000, // >200k: $4.00/$18.00/$0.40 per M
		),
		// Gemini 3 Pro Preview: tiered by context length (<=200k vs >200k tokens).
		pricing.TieredModel(ModelGemini3ProPreview, 200_000,
			200_000_000, 1_200_000_000, 20_000_000, // <=200k: $2.00/$12.00/$0.20 per M
			400_000_000, 1_800_000_000, 40_000_000, // >200k: $4.00/$18.00/$0.40 per M
		),
		// Gemini 3 Flash Preview: flat pricing.
		pricing.FlatModel(ModelGemini3FlashPreview, 50_000_000, 300_000_000, 5_000_000),

		// ── Gemini 2.5 ──────────────────────────────────────────
		// Gemini 2.5 Pro: tiered by context length (<=200k vs >200k tokens).
		pricing.TieredModel(ModelGemini25Pro, 200_000,
			125_000_000, 1_000_000_000, 12_500_000, // <=200k: $1.25/$10.00/$0.125 per M
			250_000_000, 1_500_000_000, 25_000_000, // >200k: $2.50/$15.00/$0.25 per M
		),
		// Gemini 2.5 Flash: flat pricing.
		pricing.FlatModel(ModelGemini25Flash, 30_000_000, 250_000_000, 3_000_000),
		// Gemini 2.5 Flash Lite: flat pricing.
		pricing.FlatModel(ModelGemini25FlashLite, 10_000_000, 40_000_000, 1_000_000),

		// ── Gemini 2.0 ──────────────────────────────────────────
		pricing.FlatModel(ModelGemini20Flash, 10_000_000, 40_000_000, 2_500_000),
	}
}
