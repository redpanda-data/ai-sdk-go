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
	for id, def := range supportedModels {
		t.Run(id, func(t *testing.T) {
			assert.Greater(t, def.Pricing.InputPerMillion, int64(0),
				"model %s missing input pricing — add Pricing to its ModelDefinition", id)
			assert.Greater(t, def.Pricing.OutputPerMillion, int64(0),
				"model %s missing output pricing — add Pricing to its ModelDefinition", id)
		})
	}
}

func TestDefaultPricingMatchesModels(t *testing.T) {
	pricingList := DefaultPricing()
	assert.Equal(t, len(supportedModels), len(pricingList),
		"DefaultPricing should return exactly one entry per supported model")
}
