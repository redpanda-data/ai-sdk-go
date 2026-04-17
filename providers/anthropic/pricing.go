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

import "github.com/redpanda-data/ai-sdk-go/pricing"

// DefaultPricing returns pricing for all supported Anthropic Claude models.
// Prices are in microcents per million tokens.
//
// Source: https://docs.anthropic.com/en/docs/about-claude/pricing (as of 2026-04)
func DefaultPricing() []pricing.ModelPricing {
	return []pricing.ModelPricing{
		// ── Claude 4.6 ─────────────────────────────────────────
		//                                                  input          output         cached
		pricing.FlatModel(ModelClaudeSonnet46, 300_000_000, 1_500_000_000, 30_000_000),
		pricing.FlatModel(ModelClaudeOpus46, 500_000_000, 2_500_000_000, 50_000_000),

		// ── Claude 4.5 ─────────────────────────────────────────
		pricing.FlatModel(ModelClaudeSonnet45, 300_000_000, 1_500_000_000, 30_000_000),
		pricing.FlatModel(ModelClaudeOpus45, 500_000_000, 2_500_000_000, 50_000_000),
		pricing.FlatModel(ModelClaudeHaiku45, 100_000_000, 500_000_000, 10_000_000),

		// ── Claude 4.1 ─────────────────────────────────────────
		pricing.FlatModel(ModelClaudeOpus41, 1_500_000_000, 7_500_000_000, 150_000_000),
	}
}
