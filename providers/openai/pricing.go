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
// Pricing is derived from model definitions in supportedModels to ensure
// every model always has pricing defined.
//
// Source: https://openai.com/api/pricing/ (as of 2026-04).
func DefaultPricing() []pricing.ModelPricing {
	models := make([]pricing.ModelPricing, 0, len(supportedModels))
	for id, def := range supportedModels {
		models = append(models, def.Pricing.ToModelPricing(id))
	}

	return models
}
