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

package vertex_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

func TestIsModelAvailableAtLocation(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		model    string
		location string
		want     bool
	}{
		{"gemini at global", vertex.ModelGemini36Flash, "global", true},
		{"gemini at us multi-region", vertex.ModelGemini36Flash, "us", true},
		// Google publishes no named-region availability for Gemini 3.6
		// Flash, only global and the us/eu multi-regions.
		{"gemini not at named region", vertex.ModelGemini36Flash, "us-east5", false},
		{"gemini location case-insensitive", vertex.ModelGemini36Flash, "EU", true},
		{"sonnet-5 at global", vertex.ModelClaudeSonnet5, "global", true},
		{"sonnet-5 at eu multi-region", vertex.ModelClaudeSonnet5, "eu", true},
		// Sonnet is published at asia-southeast1 but not at us-east5; each
		// model carries only the regions Google's page marks for it.
		{"sonnet-5 at asia-southeast1", vertex.ModelClaudeSonnet5, "asia-southeast1", true},
		{"sonnet-5 not at named region", vertex.ModelClaudeSonnet5, "us-east5", false},
		{"haiku at named region", vertex.ModelClaudeHaiku45, "europe-west1", true},
		// The matrix lists asia-east1 for Haiku.
		{"haiku not at asia-east1", vertex.ModelClaudeHaiku45, "asia-east1", false},
		{"haiku not at asia-southeast1", vertex.ModelClaudeHaiku45, "asia-southeast1", false},
		{"unknown model", "gemini-99-ultra", "global", false},
		{"prefixed id no longer resolves", "vertex." + vertex.ModelClaudeSonnet5, "eu", false},
		{"unknown location", vertex.ModelGemini36Flash, "mars-central1", false},
		{"empty location", vertex.ModelGemini36Flash, "", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tc.want, vertex.IsModelAvailableAtLocation(tc.model, tc.location))
		})
	}
}

// TestLocationsForModel checks the accessor returns a copy (the caller
// mutating it must not corrupt the shared table) and nil for an unknown
// model.
func TestLocationsForModel(t *testing.T) {
	t.Parallel()

	assert.Nil(t, vertex.LocationsForModel("gemini-99-ultra"))

	locs := vertex.LocationsForModel(vertex.ModelClaudeSonnet5)
	require.NotEmpty(t, locs)

	locs[0] = "tampered"
	again := vertex.LocationsForModel(vertex.ModelClaudeSonnet5)
	assert.NotEqual(t, "tampered", again[0], "LocationsForModel returned a slice aliasing the shared table")
}

// TestServedLocationsMatrix pins the exact served-location set for every
// catalogued model, so a hand edit to the transcribed matrix that drops or
// reorders a location is caught rather than silently changing availability.
// The expected slices are the full transcription dated
// LocationsMatrixTranscribed; update them together when the matrix changes.
func TestServedLocationsMatrix(t *testing.T) {
	t.Parallel()

	usRegions := []string{"us-central1", "us-east1", "us-east4", "us-east5", "us-south1", "us-west1", "us-west4"}
	globalPlus := func(parts ...[]string) []string {
		out := []string{"global"}
		for _, p := range parts {
			out = append(out, p...)
		}

		return out
	}
	namedClaude := []string{"global", "us-east5", "europe-west1", "asia-southeast1"}

	want := map[string][]string{
		vertex.ModelGemini38Flash: {"global", "us", "eu"},
		vertex.ModelGemini37Flash: {"global", "us", "eu"},
		vertex.ModelGemini36Flash: {"global", "us", "eu"},
		vertex.ModelGemini35Flash: {
			"global", "us", "eu",
			"northamerica-northeast1", "europe-west2", "europe-west3",
			"asia-northeast1", "asia-south1", "asia-southeast1", "australia-southeast1",
		},
		vertex.ModelGemini35FlashLite:   {"global", "us", "eu"},
		vertex.ModelGemini31FlashLite:   {"global", "us", "eu"},
		vertex.ModelGemini31ProPreview:  {"global"},
		vertex.ModelGemini3FlashPreview: {"global"},
		vertex.ModelGemini25Pro: globalPlus(usRegions,
			[]string{"northamerica-northeast1"},
			[]string{"europe-central2", "europe-north1", "europe-southwest1", "europe-west1", "europe-west4", "europe-west8", "europe-west9"},
			[]string{"asia-northeast1"},
		),
		vertex.ModelGemini25Flash: globalPlus(usRegions,
			[]string{"northamerica-northeast1", "southamerica-east1"},
			[]string{"europe-central2", "europe-north1", "europe-southwest1", "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west8", "europe-west9"},
			[]string{"asia-northeast1", "asia-northeast3", "asia-south1", "asia-southeast1", "australia-southeast1"},
		),
		vertex.ModelGemini25FlashLite: globalPlus(usRegions,
			[]string{"europe-central2", "europe-north1", "europe-southwest1", "europe-west1", "europe-west4", "europe-west8", "europe-west9"},
		),
		vertex.ModelClaudeFable51:  {"global", "us", "eu"},
		vertex.ModelClaudeFable5:   {"global", "us", "eu", "asia-southeast1"},
		vertex.ModelClaudeOpus55:   {"global", "us", "eu"},
		vertex.ModelClaudeOpus5:    {"global", "us", "eu", "asia-southeast1"},
		vertex.ModelClaudeOpus48:   {"global", "us", "eu"},
		vertex.ModelClaudeOpus47:   {"global", "us", "eu"},
		vertex.ModelClaudeOpus46:   namedClaude,
		vertex.ModelClaudeOpus45:   namedClaude,
		vertex.ModelClaudeSonnet55: {"global", "us", "eu"},
		vertex.ModelClaudeSonnet5:  {"global", "us", "eu", "asia-southeast1"},
		vertex.ModelClaudeSonnet46: namedClaude,
		vertex.ModelClaudeSonnet45: namedClaude,
		vertex.ModelClaudeHaiku45:  {"global", "us-east5", "europe-west1"},
	}

	require.Len(t, want, vertex.Catalog().Len(), "want covers every offering")

	for model, wantLocs := range want {
		got := vertex.LocationsForModel(model)
		assert.Equalf(t, wantLocs, got, "served locations for %s", model)
	}
}

// TestMatrixProvenance checks the transcription carries a source URL and a
// well-formed date, so a stale copy is always traceable to what it copied
// and when.
func TestMatrixProvenance(t *testing.T) {
	t.Parallel()

	assert.NotEmpty(t, vertex.LocationsMatrixSource)

	_, err := time.Parse("2006-01-02", vertex.LocationsMatrixTranscribed)
	assert.NoError(t, err, "LocationsMatrixTranscribed must be a YYYY-MM-DD date")
}

// TestLocationsFollowCatalogAliases checks a bare alias answers as its dated
// offering.
func TestLocationsFollowCatalogAliases(t *testing.T) {
	t.Parallel()

	for alias, id := range map[string]string{
		"claude-opus-4-5":   vertex.ModelClaudeOpus45,
		"claude-sonnet-4-5": vertex.ModelClaudeSonnet45,
	} {
		assert.Truef(t, vertex.IsModelAvailableAtLocation(alias, "us-east5"), "%s at us-east5", alias)
		assert.Equalf(t, vertex.LocationsForModel(id), vertex.LocationsForModel(alias), "%s locations", alias)
	}
}
