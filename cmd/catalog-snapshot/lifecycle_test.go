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

// TestVertexPricingDoesNotCollide is the {provider, model} keying guard.
// Vertex model names are byte-identical to the Gemini-API and
// Anthropic-direct providers', so a catalog keyed by bare model ID would
// reject the shared IDs. Keying by {provider, model} keeps them distinct:
// Vertex registers under gcp.vertex, the others under their own provider
// keys, so the same bare "claude-sonnet-5" coexists.
//
// It pairs Vertex against each other provider in turn (rather than merging
// all five at once) so the check isolates Vertex: a pre-existing collision
// between two other providers, which production never merges, cannot fail
// this test. pricing.NewCatalog returns a duplicate-pricing error on any
// {provider, model} clash, so a NoError result proves the composite key
// does its job.
func TestVertexPricingDoesNotCollide(t *testing.T) {
	t.Parallel()

	others := []*catalog.Catalog{
		anthropic.Catalog(), bedrock.Catalog(), google.Catalog(), meta.Catalog(), openai.Catalog(),
	}

	for _, other := range others {
		// Pin the premise: the guard is only meaningful while some other
		// provider actually shares a bare model ID with Vertex. Anthropic
		// carries claude-sonnet-5 too, so if that ID ever leaves either
		// catalog this test would pass vacuously — assert the overlap.
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

// TestVertexAndAnthropicPriceSonnet5Differently is the other half of
// AI-2118 AC 3: keying by {provider, model} must let the same bare model
// ID resolve to a different rate card per provider. Vertex resells
// claude-sonnet-5 at Anthropic's global rate but adds Google's flat
// non-global regional premium, so its Info carries region overrides that
// the Anthropic-direct Info does not.
//
// The two share a Default.Base (both NewRates(2.00, 10.00, 0.20)...), so
// the distinguishing difference is the override set, not the base rate;
// the assertion compares the whole rate card rather than only the base.
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

	// Keyed by "provider/ID": bare model IDs are no longer unique across
	// providers (Vertex dropped its "vertex." prefix, so it shares
	// "claude-sonnet-5" with Anthropic), and an exemption must apply to
	// exactly the one catalog that earned it.
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
