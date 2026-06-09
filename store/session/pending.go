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

package session

import (
	"encoding/json"
	"maps"
	"time"
)

// PendingToolCallSchemaVersion is the current schema version of
// PendingToolCall. Increment when adding fields that older runners
// would silently misinterpret; MigratePendingToolCalls applies forward
// migrations on load.
const PendingToolCallSchemaVersion = 1

// PendingToolCall is the durable record of a paused tool execution. It
// is keyed by the originating tool call ID so resume APIs can locate
// the call without scanning history.
//
// All fields are designed to round-trip through JSON-encoded session
// stores; opaque blobs are kept as json.RawMessage so they survive
// distributed-store re-encoding without loss.
type PendingToolCall struct {
	// SchemaVersion is the version of this record. Older entries (v0,
	// produced before this field existed) are migrated on load.
	SchemaVersion int `json:"schema_version"`

	// ID is the model-issued tool call ID. Stable across the initial
	// call and any re-entry.
	ID string `json:"id"`

	// Name is the tool name the call targeted at pause time. Resume
	// uses this to look up the tool when ResumeMode == reentry.
	Name string `json:"name"`

	// Arguments is the original tool arguments JSON. Persisted so that
	// approvals/elicitations can re-invoke the tool with the original
	// payload after a deploy that does not preserve in-memory state.
	Arguments json.RawMessage `json:"arguments,omitempty"`

	// Reason mirrors tool.AwaitReason as a string so this package does
	// not need to import tool. Converted by the runner.
	Reason string `json:"reason"`

	// Resume mirrors tool.ResumeMode as a string.
	Resume string `json:"resume"`

	// Message is the UI-facing message captured from the Await. Useful
	// to adapters that need to display a prompt to the user.
	Message string `json:"message,omitempty"`

	// Prompt is optional structured prompt data (approval options,
	// elicitation schema, etc.).
	Prompt json.RawMessage `json:"prompt,omitempty"`

	// State is the opaque tool-private state. On re-entry the runner
	// hands this back as Call.Resume.PriorState.
	State json.RawMessage `json:"state,omitempty"`

	// CorrelationID is an app-supplied identifier (deployment ID,
	// webhook job ID, …).
	CorrelationID string `json:"correlation_id,omitempty"`

	// CreatedAt is when the pending call was stored.
	CreatedAt time.Time `json:"created_at"`

	// ExpiresAt is the wall-clock expiry computed from Await.Timeout.
	// Nil means no timeout.
	ExpiresAt *time.Time `json:"expires_at,omitempty"`

	// LastOutput is the most recent placeholder output sent to the
	// model. Updated as progress lands so a normalized re-render can
	// pick up the latest snapshot.
	LastOutput json.RawMessage `json:"last_output,omitempty"`

	// Progress accumulates non-terminal updates. Bounded by application
	// policy; the runtime does not trim.
	Progress []ProgressEntry `json:"progress,omitempty"`

	// Metadata carries opaque per-call audit/transport data. Never sent
	// to the model.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// ResumeReceipt records a completed pending call so subsequent
// duplicate resume submissions can be detected and acknowledged without
// mutating session state.
type ResumeReceipt struct {
	// CallID is the originating tool call ID.
	CallID string `json:"call_id"`

	// ResultHash is the JCS-canonical SHA-256 of the resume payload
	// (output or error). Used to detect at-least-once duplicates.
	ResultHash string `json:"result_hash"`

	// ResolvedAt is when the resume completed.
	ResolvedAt time.Time `json:"resolved_at"`

	// Metadata carries optional audit data captured on first resume.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// ProgressEntry is a single non-terminal update on a pending call.
// Progress is for UI/task state; it never reaches the model.
type ProgressEntry struct {
	At      time.Time       `json:"at"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

// clonePending deep-copies a PendingToolCall map for State.Clone.
func clonePending(src map[string]PendingToolCall) map[string]PendingToolCall {
	if src == nil {
		return nil
	}

	out := make(map[string]PendingToolCall, len(src))
	for k, v := range src {
		out[k] = clonePendingToolCall(v)
	}

	return out
}

func clonePendingToolCall(p PendingToolCall) PendingToolCall {
	clone := p

	if p.Arguments != nil {
		clone.Arguments = append(json.RawMessage(nil), p.Arguments...)
	}

	if p.Prompt != nil {
		clone.Prompt = append(json.RawMessage(nil), p.Prompt...)
	}

	if p.State != nil {
		clone.State = append(json.RawMessage(nil), p.State...)
	}

	if p.LastOutput != nil {
		clone.LastOutput = append(json.RawMessage(nil), p.LastOutput...)
	}

	if p.ExpiresAt != nil {
		t := *p.ExpiresAt
		clone.ExpiresAt = &t
	}

	if p.Progress != nil {
		clone.Progress = make([]ProgressEntry, len(p.Progress))
		for i, pe := range p.Progress {
			peCopy := pe
			if pe.Payload != nil {
				peCopy.Payload = append(json.RawMessage(nil), pe.Payload...)
			}

			clone.Progress[i] = peCopy
		}
	}

	if p.Metadata != nil {
		clone.Metadata = maps.Clone(p.Metadata)
	}

	return clone
}

// cloneReceipts deep-copies a ResumeReceipt map for State.Clone.
func cloneReceipts(src map[string]ResumeReceipt) map[string]ResumeReceipt {
	if src == nil {
		return nil
	}

	out := make(map[string]ResumeReceipt, len(src))
	for k, v := range src {
		clone := v
		if v.Metadata != nil {
			clone.Metadata = maps.Clone(v.Metadata)
		}

		out[k] = clone
	}

	return out
}

// MigratePendingToolCalls applies forward migrations to any pending
// call entries that predate the current schema version. The migration
// rule is conservative: missing versions are treated as v0, new fields
// keep their zero values, and unknown metadata is preserved. Field
// renames require an explicit case here.
func MigratePendingToolCalls(s *State) {
	if s == nil || len(s.PendingToolCalls) == 0 {
		return
	}

	for id, pc := range s.PendingToolCalls {
		if pc.SchemaVersion == PendingToolCallSchemaVersion {
			continue
		}

		// v0 -> v1: just stamp the version; v0 records never set any
		// of the v1 fields, so zero-value defaults are correct.
		if pc.SchemaVersion < PendingToolCallSchemaVersion {
			pc.SchemaVersion = PendingToolCallSchemaVersion
		}

		s.PendingToolCalls[id] = pc
	}
}
