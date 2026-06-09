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
	"fmt"
	"maps"
	"time"
)

// Await describes a pause point in tool execution. A non-nil Await on an
// Execution tells the runtime to persist a typed pending tool call and stop
// the invocation with FinishReasonPaused until the caller submits a resume.
//
// Await is control-flow state. It is shaped by the runtime, not the model:
// the placeholder Output on the same Execution is what the model sees, while
// Await travels alongside it to the session store, the runner, and protocol
// adapters.
type Await struct {
	// Reason classifies why execution is paused. It determines how adapters
	// (A2A, UI streams, OTel) surface the pause to the outside world.
	Reason AwaitReason

	// Resume describes how the runtime should continue once the caller has
	// the answer. Valid Reason/Resume combinations are enforced by
	// Validate.
	Resume ResumeMode

	// Message is suitable for UI and adapters. It is not sent to the model
	// and should not be treated as a prompt instruction.
	Message string

	// Prompt is optional structured UI data (e.g. approval options, an
	// elicitation JSON schema, a form description). It is surfaced to SDK
	// consumers and adapters, not sent to the model by default.
	Prompt json.RawMessage

	// State is opaque tool-private data persisted with the pending call
	// and handed back as Call.Resume.PriorState when the runtime re-enters
	// a tool with ResumeWithReentry.
	State json.RawMessage

	// CorrelationID is an app/job identifier (e.g. a deployment ID, a
	// webhook job ID). Surfaced to adapters and used for cross-system
	// joins. Coalescing uses ArgumentsHash, not CorrelationID, so this is
	// purely an annotation.
	CorrelationID string

	// Timeout is an optional wall-clock timeout for the pending call. The
	// runtime computes ExpiresAt when storing pending state. Zero means
	// "no timeout" and is the safe default for human-driven pauses.
	Timeout time.Duration

	// ExpiresAt is computed by the runtime from Timeout when the pending
	// call is persisted. Tools should leave this unset; setting it directly
	// is supported for tests and replay scenarios.
	ExpiresAt *time.Time

	// Metadata carries opaque transport/audit/debug data. It is never sent
	// to the model and must not be used as a hidden control channel — if
	// SDK code starts branching on a metadata key, promote it to a typed
	// field.
	Metadata map[string]any
}

// AwaitReason classifies why a tool paused.
type AwaitReason string

// Recognized AwaitReason values. The set is intentionally closed: each
// reason maps onto a concrete adapter (A2A task state, OTel attribute,
// runner resume path).
const (
	// AwaitReasonToolResult is a long-running external job. The tool
	// kicked off remote work and is waiting for the result to arrive
	// (webhook, polling, queue notification).
	AwaitReasonToolResult AwaitReason = "tool_result"

	// AwaitReasonUserInput is a conversational pause: the assistant needs
	// the user to clarify or answer before continuing.
	AwaitReasonUserInput AwaitReason = "user_input"

	// AwaitReasonApproval is a human-in-the-loop approval gate, typically
	// emitted by an approval interceptor before the wrapped tool runs.
	AwaitReasonApproval AwaitReason = "approval"

	// AwaitReasonElicitation is an MCP elicitation: a server-driven prompt
	// for additional structured input mid-tool.
	AwaitReasonElicitation AwaitReason = "elicitation"

	// AwaitReasonHandoff is a child agent pause bubbled up through
	// agent-as-tool. The parent re-enters its child session on resume.
	AwaitReasonHandoff AwaitReason = "handoff"
)

// ResumeMode tells the runtime how to consume the caller's resume payload.
type ResumeMode string

const (
	// ResumeWithToolResponse means the caller submits the final tool
	// output JSON. The runtime records it as the tool response without
	// re-entering the tool.
	ResumeWithToolResponse ResumeMode = "tool_response"

	// ResumeWithMessage means the caller's next user message is the
	// meaningful continuation. The placeholder tool response remains in
	// history so providers still see a complete tool-call/tool-response
	// sequence, and the user message follows.
	ResumeWithMessage ResumeMode = "message"

	// ResumeWithReentry means the runtime re-invokes the same tool with
	// Call.Resume populated. Use this for approval interceptors, MCP
	// elicitation, and agent-as-tool handoffs that need to finish their
	// own logic after the caller answers.
	ResumeWithReentry ResumeMode = "reentry"
)

// allowedAwaitPairs is the closed set of valid Reason/Resume combinations.
// Other combinations are runtime errors.
var allowedAwaitPairs = map[AwaitReason]map[ResumeMode]struct{}{
	AwaitReasonToolResult: {
		ResumeWithToolResponse: {},
		ResumeWithReentry:      {},
	},
	AwaitReasonUserInput: {
		ResumeWithMessage: {},
	},
	AwaitReasonApproval: {
		ResumeWithReentry: {},
	},
	AwaitReasonElicitation: {
		ResumeWithReentry: {},
	},
	AwaitReasonHandoff: {
		ResumeWithReentry: {},
	},
}

// Validate reports whether a is a well-formed Await. It checks that Reason
// and Resume are set and that the pair is in the allowed table. Validation
// failures should be encoded as tool errors by the runtime, not as pending
// state.
func (a *Await) Validate() error {
	if a == nil {
		return nil
	}

	if a.Reason == "" {
		return ErrAwaitReasonEmpty
	}

	if a.Resume == "" {
		return ErrAwaitResumeEmpty
	}

	modes, ok := allowedAwaitPairs[a.Reason]
	if !ok {
		return fmt.Errorf("%w: unknown reason %q", ErrAwaitInvalid, a.Reason)
	}

	if _, ok := modes[a.Resume]; !ok {
		return fmt.Errorf("%w: reason %q cannot use resume mode %q",
			ErrAwaitInvalid, a.Reason, a.Resume)
	}

	return nil
}

// AwaitOption configures an Await built by helpers like Pending and
// NeedInput. Options compose: later options overwrite earlier ones.
type AwaitOption func(*Await)

// WithAwaitMessage sets a UI-facing message on the Await.
func WithAwaitMessage(msg string) AwaitOption {
	return func(a *Await) { a.Message = msg }
}

// WithAwaitPrompt sets structured UI prompt data on the Await.
func WithAwaitPrompt(prompt json.RawMessage) AwaitOption {
	return func(a *Await) { a.Prompt = prompt }
}

// WithAwaitState sets opaque tool-private state that is handed back via
// Call.Resume.PriorState when the runtime re-enters the tool.
func WithAwaitState(state json.RawMessage) AwaitOption {
	return func(a *Await) { a.State = state }
}

// WithCorrelationID sets an external correlation identifier (job ID,
// deployment ID, etc.).
func WithCorrelationID(id string) AwaitOption {
	return func(a *Await) { a.CorrelationID = id }
}

// WithAwaitTimeout sets a wall-clock timeout. The runtime computes
// ExpiresAt from Timeout when persisting the pending call.
func WithAwaitTimeout(d time.Duration) AwaitOption {
	return func(a *Await) { a.Timeout = d }
}

// WithAwaitMetadata sets opaque transport/audit metadata. The map is
// copied to prevent caller-side mutation after the helper returns.
func WithAwaitMetadata(md map[string]any) AwaitOption {
	return func(a *Await) {
		if md == nil {
			a.Metadata = nil
			return
		}

		a.Metadata = maps.Clone(md)
	}
}
