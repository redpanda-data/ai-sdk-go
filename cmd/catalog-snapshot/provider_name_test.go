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
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/meta"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

// TestCatalogProviderMatchesProviderName pins that each model catalog
// registers under its runtime provider's Name(): a drift there prices
// every lookup by Name() at zero, as Google's "google" vs "gcp.gemini" did.
//
// Meta is the case the pinned want matters most for: its Name() and its
// catalog key are two separate bare literals with no shared const between
// them (providers/meta/provider.go:46 and providers/meta/models.go:35), so
// editing one and not the other is a one-character change away.
func TestCatalogProviderMatchesProviderName(t *testing.T) {
	t.Parallel()

	// A typed nil satisfies catalog.Provider: Name() and Catalog() have
	// pointer receivers that never dereference, so no live provider is needed.
	cases := []struct {
		name     string
		provider catalog.Provider
		want     llm.ProviderID
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

			// Pin the literal: Name() vs Provider() alone cannot fail while
			// both derive from one ProviderName const.
			assert.Equalf(t, tc.want, tc.provider.Name(),
				"%s provider Name() drifted from its documented key", tc.name)
			assert.Equalf(t, tc.want, tc.provider.Catalog().Provider(),
				"%s catalog provider must equal the provider's Name() so pricing lookups by Name() hit", tc.name)
		})
	}
}

// TestProviderKeysArePairwiseDistinct: two providers sharing a key would
// silently merge their pricing namespaces, and nothing else checks this.
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

	seen := make(map[llm.ProviderID]string, len(catalogs))
	for name, cat := range catalogs {
		key := cat.Provider()
		require.NotContainsf(t, seen, key,
			"provider key %q is shared by %s and %s; keys must be pairwise distinct", key, seen[key], name)
		seen[key] = name
	}
}
