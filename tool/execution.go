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

package tool

import "encoding/json"

// Execution is the structured return type from Tool.Execute. It bundles
// the model-visible Output, an optional Await for pause/resume, optional
// Actions for SDK-side side effects, and observability Metadata.
//
// Tool errors are still returned as the second return value of Execute;
// the runtime ignores Execution when err != nil and encodes the error as
// a tool error response. See registry execution semantics for details.
type Execution struct {
	// Output is the JSON value sent back to the model when this
	// execution is reconciled into a llm.ToolResponsePart. For Await
	// pauses this is the placeholder that travels in history until the
	// runtime replaces it with the final resumed result.
	Output json.RawMessage

	// Await is non-nil when this tool call has not reached its terminal
	// result. The runtime persists a typed PendingToolCall and stops the
	// invocation with FinishReasonPaused.
	Await *Await

	// Actions contains side effects for the SDK to reconcile (currently
	// artifact emission, surfaced as agent.ToolArtifactEvent on the
	// event stream). Await is kept separate because it is part of the
	// control-flow contract, not a side effect.
	Actions []Action

	// Metadata is for observability and adapter-specific details. It is
	// not sent to the model by default. Promote keys to typed fields if
	// SDK code starts branching on them.
	Metadata map[string]any
}

// Action is an SDK-owned side-effect envelope returned alongside an
// Execution. Keep it small; new kinds should be additive.
type Action struct {
	// Kind discriminates the union members below. Exactly one of the
	// kind-specific pointers should be non-nil.
	Kind ActionKind

	// Artifact is set when Kind == ActionArtifact.
	Artifact *ArtifactAction

	// Metadata carries opaque per-action context not interpreted by the
	// SDK.
	Metadata map[string]any
}

// ActionKind discriminates the Action union.
type ActionKind string

// ActionArtifact is emitted when a tool produces a binary artifact
// (image, file, plot) that should be persisted or surfaced to a UI.
const ActionArtifact ActionKind = "artifact"

// ArtifactAction is the side-effect payload for ActionArtifact.
type ArtifactAction struct {
	// ID is a stable identifier for the artifact within the session.
	ID string

	// Name is a human-readable label.
	Name string

	// Description is an optional longer description for UI display.
	Description string

	// MediaType is the IANA media type (e.g. "image/png",
	// "application/json").
	MediaType string

	// Data is the raw artifact bytes. Storage is the runner's
	// responsibility.
	Data []byte
}
