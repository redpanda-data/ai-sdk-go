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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/meta"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

// TestCatalogProviderMatchesProviderName pins the invariant AI-2118 adds:
// once the pricing catalog keys by {provider, model}, a
// model catalog registered under a provider string that differs from the
// runtime provider's Name() means every lookup by that Name() misses and
// prices at zero. The Google catalog carried exactly that drift ("google"
// vs Name() "gcp.gemini") before this change.
//
// Vertex is guarded here too: it has a runtime Provider whose Name()
// returns providers/vertex.ProviderName, so the same drift between the
// catalog key and Name() would price every Vertex lookup at zero.
//
// Meta is the case the pinned want matters most for: its Name() and its
// catalog key are two separate bare literals with no shared const between
// them (providers/meta/provider.go:46 and providers/meta/models.go:34), so
// editing one and not the other is a one-character change away.
func TestCatalogProviderMatchesProviderName(t *testing.T) {
	t.Parallel()

	// provider is the real catalog.Provider surface (Name + Catalog), not
	// an ad-hoc Name()-only interface: each provider's Name() and Catalog()
	// have unnamed pointer receivers, so a typed nil satisfies it without
	// constructing a live provider (no API key or context), and the catalog
	// is read off the same value rather than passed separately.
	cases := []struct {
		name     string
		provider catalog.Provider
		want     string
	}{
		{"openai", (*openai.Provider)(nil), "openai"},
		{"anthropic", (*anthropic.Provider)(nil), "anthropic"},
		{"google", (*google.Provider)(nil), "gcp.gemini"},
		{"bedrock", (*bedrock.Provider)(nil), "aws.bedrock"},
		{"vertex", (*vertex.Provider)(nil), "gcp.vertex"},
		{"meta", (*meta.Provider)(nil), "meta"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Pin the literal on both sides. Comparing Name() to
			// Provider() alone cannot fail while both derive from one
			// ProviderName const; pinning want catches a const edited to
			// the wrong value — including the "google" vs "gcp.gemini"
			// drift AC 4 names.
			assert.Equalf(t, tc.want, tc.provider.Name(),
				"%s provider Name() drifted from its documented key", tc.name)
			assert.Equalf(t, tc.want, tc.provider.Catalog().Provider(),
				"%s catalog provider must equal the provider's Name() so pricing lookups by Name() hit", tc.name)
		})
	}
}

// TestProviderKeysArePairwiseDistinct pins the invariant the {provider,
// model} key now rests on: two providers sharing a key would silently
// merge their pricing namespaces again. Before AI-2118 the bare-model-ID
// key made provider separation self-enforcing — two providers registering
// an overlapping model ID failed the build — so nothing else states it.
func TestProviderKeysArePairwiseDistinct(t *testing.T) {
	t.Parallel()

	catalogs := map[string]*catalog.Catalog{
		"openai":    openai.Catalog(),
		"anthropic": anthropic.Catalog(),
		"google":    google.Catalog(),
		"bedrock":   bedrock.Catalog(),
		"vertex":    vertex.Catalog(),
		"meta":      meta.Catalog(),
	}

	seen := make(map[string]string, len(catalogs))
	for name, cat := range catalogs {
		key := cat.Provider()
		require.NotContainsf(t, seen, key,
			"provider key %q is shared by %s and %s; keys must be pairwise distinct", key, seen[key], name)
		seen[key] = name
	}
}
