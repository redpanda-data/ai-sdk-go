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

	for _, def := range Catalog().All() {
		id := def.ID
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			p := def.Pricing
			assert.Positive(t, p.Default.Base.InputPerMillion,
				"model %s missing input pricing — add Pricing to its ModelDefinition", id)
			assert.Positive(t, p.Default.Base.OutputPerMillion,
				"model %s missing output pricing — add Pricing to its ModelDefinition", id)
			assert.Positive(t, p.Default.Base.CachedInputPerMillion,
				"model %s missing cached pricing — add CachedInputPerMillion to its ModelDefinition", id)

			for i, tier := range p.Default.Brackets {
				assert.Positive(t, tier.Rates.InputPerMillion,
					"model %s missing input pricing in tier %d", id, i)
				assert.Positive(t, tier.Rates.OutputPerMillion,
					"model %s missing output pricing in tier %d", id, i)
				assert.Positive(t, tier.Rates.CachedInputPerMillion,
					"model %s missing cached pricing in tier %d", id, i)
			}
		})
	}
}

func TestModelPricingMatchesModels(t *testing.T) {
	t.Parallel()

	pricingMap := ModelPricing()
	assert.Len(t, pricingMap, Catalog().Len(),
		"ModelPricing should return exactly one entry per supported model")
}
