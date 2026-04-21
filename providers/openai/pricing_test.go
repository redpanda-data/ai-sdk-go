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

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAllModelsHavePricing(t *testing.T) {
	t.Parallel()

	// Models that predate prompt caching or don't support it.
	noCacheModels := map[string]bool{
		ModelGPT4Turbo:  true, // Legacy model, no prompt caching.
		ModelGPT35Turbo: true, // Legacy model, no prompt caching.
		ModelGPT5_2Pro:  true, // Pro tier, no caching listed.
		ModelO3Pro:      true, // Pro tier, no caching listed.
	}

	for id, def := range supportedModels {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			assert.Positive(t, def.Pricing.Default.Base.InputPerMillion,
				"model %s missing input pricing — add Pricing to its ModelDefinition", id)
			assert.Positive(t, def.Pricing.Default.Base.OutputPerMillion,
				"model %s missing output pricing — add Pricing to its ModelDefinition", id)

			if !noCacheModels[id] {
				assert.Positive(t, def.Pricing.Default.Base.CachedInputPerMillion,
					"model %s missing cached pricing — add CachedInputPerMillion or add to noCacheModels if intentional", id)
			}
		})
	}
}

func TestModelPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	assert.Len(t, pricingMap, len(supportedModels),
		"ModelPricing should return exactly one entry per supported model")
}
