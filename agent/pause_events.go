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

package agent

import (
	"encoding/json"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// PendingCallSummary describes a single pending tool call surfaced via
// the agent event stream. It is the typed view of session.PendingToolCall
// that adapters (A2A, OTel, UI streams) can consume without importing
// the session package.
type PendingCallSummary struct {
	CallID        string           `json:"call_id"`
	ToolName      string           `json:"tool_name"`
	Reason        tool.AwaitReason `json:"reason"`
	Resume        tool.ResumeMode  `json:"resume"`
	Message       string           `json:"message,omitempty"`
	Prompt        json.RawMessage  `json:"prompt,omitempty"`
	CorrelationID string           `json:"correlation_id,omitempty"`
	ExpiresAt     *time.Time       `json:"expires_at,omitempty"`
}

// SummarizePendingCall projects a durable session.PendingToolCall into
// the event-facing summary shape. The single projection site shared by
// the agent loop and the runner.
func SummarizePendingCall(pc session.PendingToolCall) PendingCallSummary {
	return PendingCallSummary{
		CallID:        pc.ID,
		ToolName:      pc.Name,
		Reason:        tool.AwaitReason(pc.Reason),
		Resume:        tool.ResumeMode(pc.Resume),
		Message:       pc.Message,
		Prompt:        pc.Prompt,
		CorrelationID: pc.CorrelationID,
		ExpiresAt:     pc.ExpiresAt,
	}
}

// ToolPendingEvent reports that a tool call paused. It is emitted in the
// same turn as the corresponding ToolRequestEvent and ToolResponseEvent
// (which carries the placeholder response).
type ToolPendingEvent struct {
	Envelope    EventEnvelope        `json:"envelope"`
	PendingCall PendingCallSummary   `json:"pending_call"`
	Placeholder llm.ToolResponsePart `json:"placeholder"`
}

// GetEnvelope returns the event envelope.
func (e ToolPendingEvent) GetEnvelope() EventEnvelope { return e.Envelope }

func (ToolPendingEvent) isEvent() {}

// ToolProgressEvent reports a non-terminal progress update on a pending
// tool call. Progress is for UI/task layers only; it never triggers a
// model call and is not included in normalized model history.
type ToolProgressEvent struct {
	Envelope EventEnvelope   `json:"envelope"`
	CallID   string          `json:"call_id"`
	Payload  json.RawMessage `json:"payload"`
}

// GetEnvelope returns the event envelope.
func (e ToolProgressEvent) GetEnvelope() EventEnvelope { return e.Envelope }

func (ToolProgressEvent) isEvent() {}

// ResumeAcknowledgedEvent is emitted when runner.Resume receives a
// duplicate resume payload for an already-resolved call ID. Streams
// consumers can rely on a subsequent InvocationEndEvent (paused or
// terminal) to know that the resume request was processed without
// mutating the session.
type ResumeAcknowledgedEvent struct {
	Envelope EventEnvelope `json:"envelope"`
	CallID   string        `json:"call_id"`
	Status   string        `json:"status"`
}

// GetEnvelope returns the event envelope.
func (e ResumeAcknowledgedEvent) GetEnvelope() EventEnvelope { return e.Envelope }

func (ResumeAcknowledgedEvent) isEvent() {}
