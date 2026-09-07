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
	// row below is what LocationsMatrixSource published on 2026-09-07.
	LocationsMatrixTranscribed = "2026-09-07"
)

// servedLocations maps each catalogued bare model ID to the locations
// Google publishes it at, transcribed from LocationsMatrixSource on
// LocationsMatrixTranscribed and reconciled against the live page that
// day. The catalog's scope is global plus the US and EU buckets. Google
// also serves the Claude models in APAC (claude-sonnet-5 at
// asia-southeast1, claude-haiku-4-5 at asia-east1); that is out of this
// catalog's US/EU scope and is deliberately omitted.
//
//   - gemini-3.6-flash is served at global and the us/eu multi-regions
//     only. Google publishes no named-region availability for it.
//   - claude-sonnet-5 is served at global and the us/eu multi-regions. It
//     is on a shared-lineage quota bucket with no pay-as-you-go allocation
//     at a US or EU named region, so a call to one earns a 429 rather than
//     a completion.
//   - claude-haiku-4-5 holds a per-version quota and is served at the
//     us-east5 and europe-west1 named regions plus global. Google does not
//     publish it at the us/eu multi-regions.
var servedLocations = map[string][]string{
	ModelGemini36Flash: {
		LocationGlobal,
		"us", "eu",
	},
	ModelClaudeSonnet5: {
		LocationGlobal,
		"us", "eu",
	},
	ModelClaudeHaiku45: {
		LocationGlobal,
		"us-east5", "europe-west1",
	},
}

// IsModelAvailableAtLocation reports whether Google publishes the given
// bare model at the given location, according to the transcribed matrix.
//
// It answers config-time questions only. A false is a reason to refuse a
// save or hide a model from a picker, never to reject a live request: the
// transcription is dated and may lag Google's own additions.
//
// An unknown model or an unknown location returns false.
func IsModelAvailableAtLocation(bareModel, location string) bool {
	locs, ok := servedLocations[bareModel]
	if !ok {
		return false
	}

	return slices.Contains(locs, normalizeLocation(location))
}

// LocationsForModel returns the locations the transcribed matrix lists
// for the given bare model, or nil for an unknown model. The result is a
// copy the caller may retain and mutate.
func LocationsForModel(bareModel string) []string {
	locs, ok := servedLocations[bareModel]
	if !ok {
		return nil
	}

	return slices.Clone(locs)
}

// nonGlobalLocations returns the served locations of a model with the
// global endpoint removed. It is the set that carries the non-global
// pricing override, so pricing and availability read from one table.
func nonGlobalLocations(bareModel string) []string {
	var out []string

	for _, loc := range servedLocations[bareModel] {
		if loc != LocationGlobal {
			out = append(out, loc)
		}
	}

	return out
}

// normalizeLocation matches pricing.Selector's region normalization
// (lowercase, trimmed), so an availability answer and a price lookup
// agree on what a location string means.
func normalizeLocation(location string) string {
	return strings.ToLower(strings.TrimSpace(location))
}
