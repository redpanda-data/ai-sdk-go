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

import (
	"encoding/json"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Call is the typed envelope handed to Tool.Execute. It carries the
// model-issued tool request, invocation context, and — on re-entry — the
// caller's resume payload. Keeping Call separate from context.Context
// avoids hiding mutable SDK state in context values.
type Call struct {
	// Request is the model-issued tool request. Request.ID is stable
	// across the initial call and any re-entry for the same pending call.
	Request llm.ToolRequestPart

	// Invocation provides invocation-scoped context (IDs, turn, agent).
	Invocation InvocationInfo

	// Resume is non-nil only when the runtime re-enters this tool after a
	// previous Await with ResumeWithReentry. Tools that do not need
	// re-entry can ignore it.
	Resume *ResumePayload
}

// InvocationInfo is the tool-facing subset of agent invocation metadata.
// It is a leaf type so the tool package does not import the agent or
// runner packages.
type InvocationInfo struct {
	// InvocationID uniquely identifies the current agent invocation.
	InvocationID string

	// SessionID identifies the session this invocation belongs to.
	SessionID string

	// Turn is the agentic loop iteration (0-based) at the time the tool
	// was called.
	Turn int

	// AgentName is the agent that issued this tool call.
	AgentName string
}

// ResumePayload carries the data the caller submitted to resume a pending
// tool call. It is only ever populated for ResumeWithReentry pauses;
// ResumeWithToolResponse and ResumeWithMessage pauses do not re-enter the
// tool.
type ResumePayload struct {
	// PriorState is the opaque bytes the tool returned in Await.State.
	// Tools use this to rehydrate any context they need to continue.
	PriorState json.RawMessage

	// Result is the caller-supplied resume payload. For approvals it
	// typically carries a decision; for elicitation it carries the
	// elicited answer; for handoff it carries the child agent's final
	// output or further pause state.
	Result json.RawMessage

	// Error is non-empty when the pending call was canceled or completed
	// with an external error. Tools that branch on success vs. failure
	// should check this first.
	Error string

	// Metadata is caller-supplied resume metadata (approver identity,
	// webhook delivery IDs, etc.). It is not sent to the model.
	Metadata map[string]any

	// Progress contains intermediate updates submitted before final
	// resume via Runner.Progress.
	Progress []ProgressEntry
}

// ProgressEntry is a single non-terminal update on a pending tool call.
// Progress is persisted on the pending call and surfaced via
// ToolProgressEvent, but is not sent to the model by default.
type ProgressEntry struct {
	// At is the wall-clock time the progress update was recorded.
	At time.Time

	// Payload is the caller-supplied progress data. Format is
	// application-defined.
	Payload json.RawMessage
}
