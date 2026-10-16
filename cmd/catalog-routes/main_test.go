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
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
)

// TestRoutes checks what holds whatever the routing tables say, because the
// PR job runs it on patched code and a patch may change any route: an
// explicit profile ID and a mantle model resolve to themselves from any
// region, an unknown name is refused, and there is one sorted line per pair.
// The IDs come from the catalogue, so a patch can't strand them.
func TestRoutes(t *testing.T) {
	t.Parallel()

	var profile, mantle string

	for _, o := range bedrock.Catalog().All() {
		switch {
		case profile == "" && o.Attributes[bedrock.ModelMetadataInferenceGeo] == "global":
			profile = o.ID
		case mantle == "" && bedrock.IsMantleModel(o.ID):
			mantle = o.ID
		}
	}

	require.NotEmpty(t, profile, "no global profile in the catalogue")
	require.NotEmpty(t, mantle, "no mantle model in the catalogue")

	const unknown = "anthropic.no-such-model"

	regions := []string{"us-west-2", "eu-west-1"}
	lines, err := routes(regions, []string{unknown, profile, mantle})
	require.NoError(t, err)

	assert.Len(t, lines, 6, "one line per region and name")
	assert.True(t, slices.IsSorted(lines), "sorted")

	for _, region := range regions {
		assert.Contains(t, lines, region+" "+profile+" "+profile, "an explicit profile ID keeps its route")
		assert.Contains(t, lines, region+" "+mantle+" "+mantle, "a mantle model goes by its bare ID")
		assert.Contains(t, lines, region+" "+unknown+" -", "an unknown name is refused")
	}
}

func TestBareNames(t *testing.T) {
	t.Parallel()

	names := bareNames()
	assert.Contains(t, names, bedrock.ModelClaudeSonnet55)
	assert.NotContains(t, names, bedrock.ModelClaudeSonnet55US, "a profile ID is a route's result, not a name")
	assert.IsIncreasing(t, names)
}
