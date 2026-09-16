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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/meta"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

// allCatalogs is every provider catalog the snapshot covers. This is the
// only package that may import them all (catalog's architecture test
// forbids it there), so cross-provider invariants live here.
func allCatalogs() []*catalog.Catalog {
	return []*catalog.Catalog{
		anthropic.Catalog(), bedrock.Catalog(), google.Catalog(), meta.Catalog(), openai.Catalog(), vertex.Catalog(),
	}
}

// TestVertexPricingDoesNotCollide guards the {provider, model} keying:
// Vertex model IDs are byte-identical to the Gemini-API and
// Anthropic-direct ones and must still coexist in one pricing catalog.
//
// It pairs Vertex against each other provider in turn (rather than merging
// all five at once) so the check isolates Vertex: a pre-existing collision
// between two other providers, which production never merges, cannot fail
// this test. NewCatalog errors on any {provider, model} clash, so a
// NoError result proves the composite key does its job.
func TestVertexPricingDoesNotCollide(t *testing.T) {
	t.Parallel()

	others := []*catalog.Catalog{
		anthropic.Catalog(), bedrock.Catalog(), google.Catalog(), meta.Catalog(), openai.Catalog(),
	}

	for _, other := range others {
		// Pin the premise, or the guard passes vacuously once no ID is shared.
		if other.Provider() == anthropic.ProviderName {
			require.Contains(t, other.PricingByID(), vertex.ModelClaudeSonnet5,
				"anthropic must still share the bare claude-sonnet-5 ID for this guard to mean anything")
		}

		_, err := pricing.NewCatalog(
			pricing.WithSource(other),
			pricing.WithSource(vertex.Catalog()),
		)
		require.NoErrorf(t, err, "vertex pricing collides with %s", other.Provider())
	}
}

// TestVertexAndAnthropicPriceSonnet5Differently compares the whole rate
// card, not Default.Base: Vertex resells claude-sonnet-5 at Anthropic's
// base rate and differs only in its regional premium overrides.
func TestVertexAndAnthropicPriceSonnet5Differently(t *testing.T) {
	t.Parallel()

	cat, err := pricing.NewCatalog(
		pricing.WithSource(anthropic.Catalog()),
		pricing.WithSource(vertex.Catalog()),
	)
	require.NoError(t, err)

	direct, err := cat.Lookup(anthropic.ProviderName, anthropic.ModelClaudeSonnet5)
	require.NoError(t, err)

	viaVertex, err := cat.Lookup(vertex.ProviderName, vertex.ModelClaudeSonnet5)
	require.NoError(t, err)

	assert.NotEqual(t, direct, viaVertex,
		"claude-sonnet-5 must price differently under gcp.vertex (regional premiums) than under anthropic-direct")
	assert.Empty(t, direct.Overrides,
		"anthropic-direct claude-sonnet-5 has no regional overrides")
	assert.NotEmpty(t, viaVertex.Overrides,
		"vertex claude-sonnet-5 carries per-region premium overrides")
}

// TestDeprecatedOfferingsNameAReplacement is a tripwire: an offering with a
// deprecation or retirement date must tell callers where to go. Missing
// ReplacedBy is how o3, o3-pro and o4-mini sat deprecated with no migration
// target.
//
// If a vendor's recommended replacement is genuinely not an offering we
// carry, CLAUDE.md says to skip ReplacedBy — add the ID here with a comment
// naming the uncarried replacement rather than deleting the assertion.
func TestDeprecatedOfferingsNameAReplacement(t *testing.T) {
	t.Parallel()

	// Keyed by "provider/ID": bare IDs are no longer unique across providers.
	noCarriedReplacement := map[string]string{
		// Google recommends Gemini 3.1 Flash-Lite or Gemma 4; the google
		// catalog carries neither.
		"gcp.gemini/gemini-2.5-flash-lite": "gemini-3.1-flash-lite / gemma-4",
	}

	for _, cat := range allCatalogs() {
		for _, o := range cat.All() {
			if o.Life.Deprecated.IsZero() && o.Life.Retires.IsZero() {
				continue
			}

			if _, ok := noCarriedReplacement[cat.Provider()+"/"+o.ID]; ok {
				continue
			}

			assert.NotEmptyf(t, o.Life.ReplacedBy,
				"%s/%s is deprecated or retiring but names no ReplacedBy", cat.Provider(), o.ID)
		}
	}
}

// TestReplacementsAreNotThemselvesRetired stops the catalog pointing callers
// at a dead model: a migration target that has already passed its own
// retirement date is worse than none.
func TestReplacementsAreNotThemselvesRetired(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC()

	for _, cat := range allCatalogs() {
		for _, o := range cat.All() {
			if o.Life.ReplacedBy == "" {
				continue
			}

			target, ok := cat.Lookup(o.Life.ReplacedBy)
			if !ok {
				continue // New already rejects an unresolvable ReplacedBy.
			}

			if target.Life.Retires.IsZero() {
				continue
			}

			assert.Falsef(t, target.Life.Retires.Before(now),
				"%s/%s points at %s, which retired %s",
				cat.Provider(), o.ID, target.ID, target.Life.Retires.Format(time.DateOnly))
		}
	}
}
