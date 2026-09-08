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
// Google both publishes it at and serves a pay-as-you-go call from,
// transcribed from LocationsMatrixSource on LocationsMatrixTranscribed and
// cross-checked with live rawPredict and count-tokens calls that day. A
// location is listed only where a live call is both published and
// callable, so a published-but-quota-refused region is left out.
//
//   - gemini-3.6-flash is served at global and the us/eu multi-regions
//     only. Google publishes no named-region availability for it.
//   - claude-sonnet-5 is served at global and the us/eu multi-regions. It
//     is on a shared-lineage quota bucket with no pay-as-you-go allocation
//     at a named region, so a call to any named region (a US or EU one, or
//     asia-southeast1) earns a 429 rather than a completion.
//   - claude-haiku-4-5 holds a per-version quota and is served at the
//     us-east5 and europe-west1 named regions plus global. Google does not
//     publish it at the us/eu multi-regions. It is published at
//     asia-southeast1 but a call there returns a 429 (no pay-as-you-go
//     allocation), and it is not published at asia-east1 at all, so neither
//     APAC region is listed.
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
