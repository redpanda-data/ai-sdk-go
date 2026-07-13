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

// ModelPricing returns a model ID → pricing map for all supported OpenAI models.
// Source: https://openai.com/api/pricing/ (as of 2026-07).
func ModelPricing() map[string]pricing.Info {
	m := make(map[string]pricing.Info, len(supportedModels)+len(modelAliases))
	for id, def := range supportedModels {
		m[id] = def.Pricing
	}

	for alias, family := range modelAliases {
		m[alias] = supportedModels[family].Pricing
	}

	return m
}
