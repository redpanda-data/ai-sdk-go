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

	"github.com/stretchr/testify/require"
)

func TestModelCatalogPreservesExactAliasAndSnapshotFacts(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	tests := []struct {
		name        string
		model       string
		releaseDate string
		endOfLife   string
		deprecated  bool
		retired     bool
	}{
		{name: "GPT-5 alias remains callable", model: ModelGPT5, releaseDate: "2025-08-07"},
		{name: "GPT-5 snapshot is deprecated", model: "gpt-5-2025-08-07", releaseDate: "2025-08-07", endOfLife: "2026-12-11", deprecated: true},
		{name: "O3 alias remains callable", model: ModelO3, releaseDate: "2025-04-16"},
		{name: "O3 snapshot is deprecated", model: "o3-2025-04-16", releaseDate: "2025-04-16", endOfLife: "2026-12-11", deprecated: true},
		{name: "GPT-4o deprecation has no inferred EOL", model: ModelGPT4O, releaseDate: "2024-05-13", deprecated: true},
		{name: "retired preview stays exact", model: "gpt-4-turbo-preview", releaseDate: "2023-11-06", endOfLife: "2026-03-26", deprecated: true, retired: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			catalog, ok := provider.ModelCatalog(tt.model)
			require.True(t, ok)
			require.Equal(t, tt.releaseDate, catalog.ReleaseDate)
			require.Equal(t, tt.endOfLife, catalog.EndOfLifeDate)
			require.Equal(t, tt.deprecated, catalog.Deprecated)
			require.Equal(t, tt.retired, catalog.Retired)
		})
	}
}

func TestModelCatalogRejectsUnrecognizedFamilySiblings(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	for _, model := range []string{
		"gpt-5-custom",
		"gpt-4o-transcribe",
		"gpt-5.6-2026-07-09",
		"gpt-5-2025-8-7",
	} {
		_, ok := provider.ModelCatalog(model)
		require.False(t, ok, model)
	}
}

func TestGPT53CodexUsesCurrentFlagshipUpgradeTrack(t *testing.T) {
	t.Parallel()

	const modelName = "gpt-5.3-codex"

	catalog, ok := (&Provider{}).ModelCatalog(modelName)
	require.True(t, ok)
	require.Equal(t, "gpt", catalog.FamilyKey)
	require.Equal(t, "openai-gpt-flagship", catalog.RecommendationGroup)
	require.Equal(t, "2026-02-05", catalog.ReleaseDate)
	require.False(t, catalog.Retired)
	require.Equal(t, ModelGPT5_6Sol, catalog.Replacement)

	models := (&Provider{}).Models()
	for _, model := range models {
		if model.Name == modelName {
			require.Equal(t, 400_000, model.Constraints.MaxInputTokens)
			require.Equal(t, 128_000, model.Constraints.MaxOutputTokens)
			require.True(t, model.Capabilities.Streaming)
			require.True(t, model.Capabilities.Reasoning)

			return
		}
	}

	t.Fatalf("model %q was not discoverable", modelName)
}
