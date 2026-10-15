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

package main

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
)

// TestOfferedRegionsMatchCode holds each host's Offered regions row in RULES.md
// to the regions its catalogue has copies in (RULES.md region-not-offered).
// It runs in the PR job's sandbox, outside the bot's reach, so a patch that
// adds a copy in a region nobody offers fails the run.
func TestOfferedRegionsMatchCode(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", reconcile.RulesPath))
	require.NoError(t, err)

	rules, _ := reconcile.ParseRules(raw)
	require.NotEmpty(t, rules.Offered, "RULES.md offers no regions")

	regions := map[string]map[string]bool{} // provider: the regions it has copies in

	for _, cat := range allCatalogs() {
		for _, o := range cat.All() {
			if geo := o.Attributes[bedrock.ModelMetadataInferenceGeo]; geo != "" {
				if regions[string(cat.Provider())] == nil {
					regions[string(cat.Provider())] = map[string]bool{}
				}

				regions[string(cat.Provider())][geo] = true
			}
		}
	}

	for host, offered := range rules.Offered {
		assert.ElementsMatch(t, offered, slices.Collect(maps.Keys(regions[decisionProviders[host]])),
			"%s: RULES.md's Offered regions (A) vs the regions its catalogue has copies in (B)", host)
	}
}
