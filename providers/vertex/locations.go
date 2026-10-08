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

package vertex

import (
	"slices"
	"strings"
)

// LocationGlobal is the location value for Vertex's global endpoint. It
// is the one location every catalogued model is served at.
const LocationGlobal = "global"

// The availability matrix has no API behind it. A Vertex model ID carries
// no geography - unlike a Bedrock inference profile - so whether a model
// is served at a location comes from Google's published model-by-location
// matrix, transcribed by hand and dated. This is the analog of Bedrock's
// IsModelAllowedFromRegion, which reads AWS inference-profile metadata.
//
// A stale copy of a dated transcription must never turn away traffic
// Vertex would serve, so the table is consulted only at config time (a
// provider save, the model picker, the connection-test default), never on
// a live request.
const (
	// LocationsMatrixSource is the page the matrix is transcribed from.
	LocationsMatrixSource = "https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations"

	// LocationsMatrixTranscribed is the date, in YYYY-MM-DD form, the matrix
	// below was copied from LocationsMatrixSource. Every row present on this
	// date is what LocationsMatrixSource published on 2026-09-22, except
	// claude-opus-5-5, whose row is its model page's "Model availability"
	// section. Rows added on 2026-09-28 were read from the page on that day,
	// and the claude-fable-5-1 and gemini-3.5-flash rows come from their
	// model pages. The claude-sonnet-5-5 row, added 2026-09-29, is its model
	// page's section.
	LocationsMatrixTranscribed = "2026-09-22"
)

// servedLocations maps each catalogued bare model ID to the locations
// Google publishes it at. A location is listed wherever Google's page marks
// the model supported there, not by any one project's quota; the one
// exception is claude-haiku-4-5 at asia-east1.
var servedLocations = map[string][]string{
	ModelGemini38Flash: {
		LocationGlobal,
		"us", "eu",
	},
	ModelGemini37Flash: {
		LocationGlobal,
		"us", "eu",
	},
	ModelGemini36Flash: {
		LocationGlobal,
		"us", "eu",
	},
	// This row is the model page's "Model availability" list; its Standard
	// PayGo line names only global, us and eu.
	ModelGemini35Flash: {
		LocationGlobal,
		"us", "eu",
		"northamerica-northeast1", "europe-west2", "europe-west3",
		"asia-northeast1", "asia-south1", "asia-southeast1", "australia-southeast1",
	},
	ModelGemini35FlashLite: {
		LocationGlobal,
		"us", "eu",
	},
	ModelGemini31FlashLite: {
		LocationGlobal,
		"us", "eu",
	},
	ModelGemini31ProPreview:  {LocationGlobal},
	ModelGemini3FlashPreview: {LocationGlobal},
	ModelGemini25Pro: {
		LocationGlobal,
		"us-central1", "us-east1", "us-east4", "us-east5", "us-south1", "us-west1", "us-west4",
		"northamerica-northeast1",
		"europe-central2", "europe-north1", "europe-southwest1", "europe-west1", "europe-west4", "europe-west8", "europe-west9",
		"asia-northeast1",
	},
	ModelGemini25Flash: {
		LocationGlobal,
		"us-central1", "us-east1", "us-east4", "us-east5", "us-south1", "us-west1", "us-west4",
		"northamerica-northeast1", "southamerica-east1",
		"europe-central2", "europe-north1", "europe-southwest1", "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west8", "europe-west9",
		"asia-northeast1", "asia-northeast3", "asia-south1", "asia-southeast1", "australia-southeast1",
	},
	ModelGemini25FlashLite: {
		LocationGlobal,
		"us-central1", "us-east1", "us-east4", "us-east5", "us-south1", "us-west1", "us-west4",
		"europe-central2", "europe-north1", "europe-southwest1", "europe-west1", "europe-west4", "europe-west8", "europe-west9",
	},
	// From the Fable 5.1 model page (not yet in the matrix). Its "ML
	// processing" list adds asia-southeast1, but Google publishes neither
	// availability nor a price there yet.
	ModelClaudeFable51: {
		LocationGlobal,
		"us", "eu",
	},
	ModelClaudeFable5: {
		LocationGlobal,
		"us", "eu",
		"asia-southeast1",
	},
	// From the Opus 5.5 model page (not yet in the matrix). Its "ML
	// processing" list adds asia-southeast1, but Google publishes neither
	// availability nor a price there yet.
	ModelClaudeOpus55: {
		LocationGlobal,
		"us", "eu",
	},
	ModelClaudeOpus5: {
		LocationGlobal,
		"us", "eu",
		"asia-southeast1",
	},
	ModelClaudeOpus48: {
		LocationGlobal,
		"us", "eu",
	},
	ModelClaudeOpus47: {
		LocationGlobal,
		"us", "eu",
	},
	ModelClaudeOpus46: {
		LocationGlobal,
		"us-east5", "europe-west1",
		"asia-southeast1",
	},
	ModelClaudeOpus45: {
		LocationGlobal,
		"us-east5", "europe-west1",
		"asia-southeast1",
	},
	// From the Sonnet 5.5 model page (not yet in the matrix).
	ModelClaudeSonnet55: {
		LocationGlobal,
		"us", "eu",
	},
	ModelClaudeSonnet5: {
		LocationGlobal,
		"us", "eu",
		"asia-southeast1",
	},
	ModelClaudeSonnet46: {
		LocationGlobal,
		"us-east5", "europe-west1",
		"asia-southeast1",
	},
	ModelClaudeSonnet45: {
		LocationGlobal,
		"us-east5", "europe-west1",
		"asia-southeast1",
	},
	// From the Haiku 5.5 model page (not yet in the matrix). A europe-west1
	// pricing tab lists it too, but the page publishes availability only for
	// the US and EU multi-regions and the global endpoint.
	ModelClaudeHaiku55: {
		LocationGlobal,
		"us", "eu",
	},
	// The matrix and a pricing tab also list asia-east1, but the model page
	// does not, and rawPredict there returns 404 "Publisher model ... was
	// not found" (2026-09-29) from a project the other three serve.
	ModelClaudeHaiku45: {
		LocationGlobal,
		"us-east5", "europe-west1",
	},
}

// IsModelAvailableAtLocation reports whether the transcribed matrix lists
// the given bare model as published by Google at the given location. An
// alias or a version-stamped ID answers as the offering it resolves to. An
// unknown model or an unknown location returns false.
func IsModelAvailableAtLocation(model, location string) bool {
	return slices.Contains(servedLocationsFor(model), normalizeLocation(location))
}

// LocationsForModel returns the locations the transcribed matrix lists
// for the given model, or nil for an unknown model. The model resolves as in
// IsModelAvailableAtLocation.
func LocationsForModel(model string) []string {
	return slices.Clone(servedLocationsFor(model))
}

func servedLocationsFor(model string) []string {
	id, ok := Catalog().ResolveID(model)
	if !ok {
		return nil
	}

	return servedLocations[id]
}

// normalizeLocation matches pricing.Selector's region normalization
// (lowercase, trimmed), so an availability answer and a price lookup
// agree on what a location string means.
func normalizeLocation(location string) string {
	return strings.ToLower(strings.TrimSpace(location))
}
