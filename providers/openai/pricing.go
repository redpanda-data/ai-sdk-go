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

// ModelPricing returns a model ID → pricing map for all supported OpenAI
// models, including official aliases ("gpt-5.6").
// Source: https://developers.openai.com/api/docs/pricing (as of 2026-08).
//
// Deprecated: use Catalog().PricingByID(), which this now wraps. It
// remains only until every provider has migrated to the catalog surface
// and will be removed with it.
func ModelPricing() map[string]pricing.Info {
	return Catalog().PricingByID()
}
