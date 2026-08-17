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

package catalog

import "maps"

// ModelID is the canonical, cross-provider identity of a model:
// "anthropic/claude-opus-5", "openai/gpt-5.6-sol". The vendor segment names
// the model's creator, not the host serving it — the same ModelID links one
// Anthropic offering and its Bedrock inference-profile variants to a single
// Facts record, which is what makes cross-provider drift structurally
// impossible rather than merely discouraged.
type ModelID string

// Facts are the host-independent truths about a model. They are authored
// once per ModelID in a Registry and referenced by every offering; New
// fails on an unregistered ModelID, so two providers cannot author
// conflicting facts for the same model.
//
// Capabilities, Constraints, and Modalities are deliberately NOT facts:
// the same model genuinely differs per host (Bedrock's Claude offerings
// do not enable vision; Vertex publishes a long-context pricing tier
// Anthropic does not).
type Facts struct {
	// Name is the undecorated display name: "Claude Opus 5".
	Name string

	// Series is the model's non-branching succession line with the version
	// stripped: "claude-opus", "gpt", "gpt-mini", "gemini-flash-lite".
	//
	// Series is defined strictly as a succession line, not a brand:
	// gpt-5, gpt-5-mini, and gpt-5-pro belong to three different series,
	// because a mini is not the successor of a base model. Successor and
	// View.Current/Previous group on it, so putting two concurrent
	// product lines in one series makes them supersede each other.
	//
	// Series is authored rather than parsed off the ID because vendors
	// rename their ladders mid-generation: gpt-5.4-mini and
	// gpt-5.6-terra are the same line.
	Series string

	// Released is the date the model first shipped anywhere. It is
	// required, and it is the sole sort key for generation ordering —
	// version strings are not comparable across naming-scheme changes
	// (gpt-4o vs gpt-4.1 vs gpt-5), release dates are.
	Released Date

	// Knowledge is the training-data cutoff. Zero when the vendor does
	// not publish one.
	Knowledge Date

	// OpenWeights reports whether the model's weights are published.
	OpenWeights bool
}

// Registry maps canonical ModelIDs to their Facts. Providers reference a
// ModelID on each catalog entry; New resolves it against the registry and
// fails on unregistered IDs.
type Registry map[ModelID]Facts

// DefaultRegistry returns a copy of the built-in registry covering every
// model shipped in this SDK's provider catalogs. The copy keeps callers
// from mutating shared state; WithRegistry copies again on the way in.
func DefaultRegistry() Registry {
	return maps.Clone(defaultRegistry)
}
