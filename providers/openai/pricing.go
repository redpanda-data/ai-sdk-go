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

package openai

import "github.com/redpanda-data/ai-sdk-go/pricing"

// DefaultPricing returns pricing for all supported OpenAI models.
// Prices are in microcents per million tokens.
//
// Source: https://openai.com/api/pricing/ (as of 2026-04)
func DefaultPricing() []pricing.ModelPricing {
	return []pricing.ModelPricing{
		// ── GPT-5.4 Series (March 2026 Flagship) ───────────────
		//                                           input          output         cached
		pricing.FlatModel("gpt-5.4", 250_000_000, 1_500_000_000, 25_000_000),
		pricing.FlatModel("gpt-5.4-mini", 75_000_000, 450_000_000, 7_500_000),
		pricing.FlatModel("gpt-5.4-nano", 20_000_000, 125_000_000, 2_000_000),

		// ── GPT-5.3 Series ─────────────────────────────────────
		pricing.FlatModel("gpt-5.3-chat-latest", 175_000_000, 1_400_000_000, 17_500_000),

		// ── GPT-5.2 Series ─────────────────────────────────────
		pricing.FlatModel("gpt-5.2", 87_500_000, 700_000_000, 17_500_000),
		pricing.FlatModel("gpt-5.2-chat-latest", 87_500_000, 700_000_000, 17_500_000),
		pricing.FlatModel("gpt-5.2-pro", 1_050_000_000, 8_400_000_000, 0),

		// ── GPT-5.1 Series ─────────────────────────────────────
		pricing.FlatModel("gpt-5.1", 62_500_000, 500_000_000, 12_500_000),

		// ── GPT-5 Series ───────────────────────────────────────
		pricing.FlatModel("gpt-5", 62_500_000, 500_000_000, 12_500_000),
		pricing.FlatModel("gpt-5-mini", 12_500_000, 100_000_000, 2_500_000),
		pricing.FlatModel("gpt-5-nano", 5_000_000, 40_000_000, 500_000),

		// ── GPT-4.1 Series ─────────────────────────────────────
		pricing.FlatModel("gpt-4.1", 200_000_000, 800_000_000, 50_000_000),
		pricing.FlatModel("gpt-4.1-mini", 40_000_000, 160_000_000, 10_000_000),

		// ── GPT-4o Series ──────────────────────────────────────
		pricing.FlatModel("gpt-4o", 250_000_000, 1_000_000_000, 125_000_000),
		pricing.FlatModel("gpt-4o-mini", 15_000_000, 60_000_000, 7_500_000),

		// ── Legacy Models ──────────────────────────────────────
		pricing.FlatModel("gpt-4-turbo", 500_000_000, 1_500_000_000, 0),
		pricing.FlatModel("gpt-3.5-turbo", 50_000_000, 150_000_000, 0),

		// ── O-Series Reasoning Models ──────────────────────────
		pricing.FlatModel("o3", 200_000_000, 800_000_000, 50_000_000),
		pricing.FlatModel("o3-pro", 2_000_000_000, 8_000_000_000, 0),
		pricing.FlatModel("o4-mini", 110_000_000, 440_000_000, 27_500_000),
		pricing.FlatModel("o1-pro", 15_000_000_000, 60_000_000_000, 7_500_000_000),
	}
}
