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

// TestThinkingBudgetTableMatchesCatalog pins the bijection between the
// catalog's Reasoning.Budget flag (the public signal) and the
// provider-local thinkingBudgets side table (the wire-level numeric
// ranges): an offering advertising budget support without a range would
// reject every budget, and a range without the flag would be dead data.
func TestThinkingBudgetTableMatchesCatalog(t *testing.T) {
	t.Parallel()

	budgetOfferings := make(map[string]bool)

	for _, o := range Catalog().All() {
		if o.Reasoning.Budget {
			budgetOfferings[o.ID] = true
			assert.Contains(t, thinkingBudgets, o.ID,
				"offering %s advertises Reasoning.Budget but has no thinkingBudgets range", o.ID)
		}
	}

	for id := range thinkingBudgets {
		assert.True(t, budgetOfferings[id],
			"thinkingBudgets has a range for %s but its catalog entry does not advertise Reasoning.Budget", id)
	}
}
