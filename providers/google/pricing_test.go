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

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAllModelsHavePricing(t *testing.T) {
	t.Parallel()

	for id, def := range supportedModels {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			p := def.Pricing
			if len(p.Tiers) > 0 {
				// Tiered models: flat fields are zero in the struct definition
				// (auto-populated by NewCatalog). Validate first tier instead.
				assert.Positive(t, p.Tiers[0].InputPerMillion,
					"model %s missing input pricing in first tier", id)
				assert.Positive(t, p.Tiers[0].OutputPerMillion,
					"model %s missing output pricing in first tier", id)
				assert.Positive(t, p.Tiers[0].CachedInputPerMillion,
					"model %s missing cached pricing in first tier", id)
			} else {
				assert.Positive(t, p.InputPerMillion,
					"model %s missing input pricing — add Pricing to its ModelDefinition", id)
				assert.Positive(t, p.OutputPerMillion,
					"model %s missing output pricing — add Pricing to its ModelDefinition", id)
				assert.Positive(t, p.CachedInputPerMillion,
					"model %s missing cached pricing — add CachedInputPerMillion to its ModelDefinition", id)
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
