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
	LocationsMatrixSource = "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations"

	// LocationsMatrixTranscribed is the date the matrix below was copied
	// from LocationsMatrixSource, in YYYY-MM-DD form. The full per-model
	// availability was reconciled against the live page on this date: every
	// row below is what LocationsMatrixSource published on 2026-09-08.
	LocationsMatrixTranscribed = "2026-09-08"
)

// servedLocations maps each catalogued bare model ID to the locations
// Google publishes it at, transcribed from LocationsMatrixSource on
// LocationsMatrixTranscribed. A location is listed wherever Google's page
// marks the model supported there.
//
// The gateway proxies traffic that already runs in the customer's own GCP
// project, so what governs this map is what Google publishes, not any one
// project's quota. Every published location is listed so the gateway prices
// and routes a customer who calls the model there.
//
//	                  global  us  eu  us-east5  europe-west1  asia-southeast1  asia-east1
//	gemini-3.6-flash  Y       Y   Y   -         -             -                -
//	claude-sonnet-5   Y       Y   Y   -         -             Y                -
//	claude-haiku-4-5  Y       -   -   Y         Y             -                Y
//
// Y = published by Google, so it appears in the map below; - = not
// published. Gemini publishes no named-region availability at all. Sonnet
// is published at the us and eu multi-regions and the asia-southeast1 named
// region. Haiku is published at the us-east5, europe-west1, and asia-east1
// named regions.
var servedLocations = map[string][]string{
	ModelGemini36Flash: {
		LocationGlobal,
		"us", "eu",
	},
	ModelClaudeSonnet5: {
		LocationGlobal,
		"us", "eu",
		"asia-southeast1",
	},
	ModelClaudeHaiku45: {
		LocationGlobal,
		"us-east5", "europe-west1",
		"asia-east1",
	},
}

// IsModelAvailableAtLocation reports whether the transcribed matrix lists
// the given bare model as served at the given location - that is, both
// published by Google and callable on a pay-as-you-go plan.
//
// It answers config-time questions only. A false is a reason to refuse a
// save or hide a model from a picker, never to reject a live request: the
// transcription is dated and may lag Google's own additions.
//
// The model may be given either as a bare publisher ID or as a
// namespaced vertex. offering ID; the prefix is stripped before lookup.
// An unknown model or an unknown location returns false.
func IsModelAvailableAtLocation(model, location string) bool {
	locs, ok := servedLocations[bareModelID(model)]
	if !ok {
		return false
	}

	return slices.Contains(locs, normalizeLocation(location))
}

// LocationsForModel returns the locations the transcribed matrix lists
// for the given model, or nil for an unknown model. The model may be a
// bare publisher ID or a namespaced vertex. offering ID. The result is a
// copy the caller may retain and mutate.
func LocationsForModel(model string) []string {
	locs, ok := servedLocations[bareModelID(model)]
	if !ok {
		return nil
	}

	return slices.Clone(locs)
}

// normalizeLocation matches pricing.Selector's region normalization
// (lowercase, trimmed), so an availability answer and a price lookup
// agree on what a location string means.
func normalizeLocation(location string) string {
	return strings.ToLower(strings.TrimSpace(location))
}
