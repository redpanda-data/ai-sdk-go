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

	"github.com/stretchr/testify/require"
)

func TestModelCatalogPreservesGeminiAliasAndRetirementFacts(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	tests := []struct {
		name        string
		model       string
		releaseDate string
		deprecated  bool
		retired     bool
		endOfLife   string
		replacement string
	}{
		{
			name:        "current Flash alias",
			model:       "models/gemini-flash-latest",
			releaseDate: "2026-05-19",
		},
		{
			name:        "deprecated stable Flash Lite remains callable",
			model:       ModelGemini31FlashLite,
			releaseDate: "2026-05-07",
			deprecated:  true,
			endOfLife:   "2027-05-07",
		},
		{
			name:        "retired Flash Lite preview",
			model:       "gemini-3.1-flash-lite-preview",
			releaseDate: "2026-03-03",
			deprecated:  true,
			retired:     true,
			endOfLife:   "2026-05-25",
			replacement: ModelGemini31FlashLite,
		},
		{
			name:        "retired Gemini 2 Flash",
			model:       "gemini-2.0-flash",
			releaseDate: "2025-02-05",
			deprecated:  true,
			retired:     true,
			endOfLife:   "2026-06-01",
			replacement: ModelGemini35Flash,
		},
		{
			name:        "retired Gemini 2.5 Pro preview snapshot",
			model:       "gemini-2.5-pro-preview-06-05",
			releaseDate: "2025-06-05",
			deprecated:  true,
			retired:     true,
			endOfLife:   "2025-12-02",
			replacement: ModelGemini31ProPreview,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			catalog, ok := provider.ModelCatalog(tt.model)
			require.True(t, ok)
			require.Equal(t, tt.releaseDate, catalog.ReleaseDate)
			require.Equal(t, tt.deprecated, catalog.Deprecated)
			require.Equal(t, tt.retired, catalog.Retired)
			require.Equal(t, tt.endOfLife, catalog.EndOfLifeDate)
			require.Equal(t, tt.replacement, catalog.ProviderReplacement)
		})
	}
}

func TestModelCatalogAcceptsOnlyKnownGeminiVersionShapes(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	_, ok := provider.ModelCatalog("models/gemini-2.5-flash-001")
	require.True(t, ok)

	for _, model := range []string{
		"gemini-3-pro-preview-custom",
		"gemini-3.1-flash-lite-preview-extra",
		"gemini-2.5-flash-foo",
	} {
		_, ok := provider.ModelCatalog(model)
		require.False(t, ok, model)
	}
}
