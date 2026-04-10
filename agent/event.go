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

//nolint:funcorder // isEvent() marker methods are intentionally placed after type definitions for clarity
package agent

import (
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Event is a sealed interface representing all possible runtime events.
// This follows the same pattern as llm.StreamEvent for type safety and consistency.
//
// All events carry a common envelope with observability fields.
//
// The isEvent() method is unexported to seal the interface, ensuring only
// events defined in this package can satisfy it (compile-time safety).
type Event interface {
	isEvent()
	GetEnvelope() EventEnvelope
}

// EventEnvelope provides observability and correlation fields for all events.
type EventEnvelope struct {
	// InvocationID uniquely identifies the current invocation.
	// All events from the same invocation share this ID.
	// Format: "inv-" + UUID v4
	InvocationID string `json:"invocation_id"`
	// SessionID identifies the session this invocation belongs to.
	// Multiple invocations can share the same session as the conversation continues.
	SessionID string `json:"session_id"`
	// Turn is the agentic loop iteration (0-based). All events within the same
	// turn share the same Turn value. A turn typically includes: model call,
	// potential tool calls/results, and turn completion. Turn advances when
	// starting a new iteration after processing tool results.
	Turn int `json:"turn"`
	// At is the emission timestamp in UTC.
	// UTC is used for inter-service consistency (events from different regions/pods),
	// easier correlation with logs/metrics (most infra uses UTC), no DST issues,
	// and stable test assertions.
	At time.Time `json:"at"`
	// Meta is metadata for extensibility.
	Meta map[string]any `json:"meta,omitempty"`
}

// StatusEvent represents execution phase transitions and provides machine-readable
// status information about where the runtime is in its lifecycle.
// StatusEvents are NON-TERMINAL - they are informational breadcrumbs for observability.
// The stream only terminates after a StreamEndEvent is emitted.
type StatusEvent struct {
	Envelope EventEnvelope `json:"envelope"`
	// Stage indicates the current execution phase.
	Stage StatusStage `json:"stage"`
	// Details provides human-readable context about the current stage.
	Details string `json:"details,omitempty"`
	// Usage contains per-turn token usage when available (e.g., after a model call completes).
	// For multi-turn runs, each StatusStageTurnCompleted event carries that turn's usage.
	// To get cumulative usage across all turns, see StreamEndEvent.Usage.
	Usage *llm.TokenUsage `json:"usage,omitempty"`
}

func (StatusEvent) isEvent() {}

// GetEnvelope returns the event envelope containing observability metadata.
func (e StatusEvent) GetEnvelope() EventEnvelope { return e.Envelope }

// StatusStage represents the execution phase of the runtime.
type StatusStage string

// Status stage constants represent different phases of agent execution.
// Preferred names use snake_case for clarity. Old names are provided as
// aliases for backward compatibility but are deprecated.
const (
	StatusStageRunStarting    StatusStage = "run_starting"
	StatusStageModelCall      StatusStage = "model_call"
	StatusStageToolExec       StatusStage = "tool_exec"
	StatusStageInputRequired  StatusStage = "input_required"
	StatusStageRunCompleted   StatusStage = "run_completed"
	StatusStageRunFailed      StatusStage = "run_failed"
	StatusStageRunInterrupted StatusStage = "run_canceled"
	StatusStageTurnStarted    StatusStage = "turn_started"
	StatusStageTurnCompleted  StatusStage = "turn_completed"
)

// MessageEvent carries an assistant message from the LLM.
// This represents the complete message after model generation.
type MessageEvent struct {
	Envelope EventEnvelope `json:"envelope"`
	Response llm.Response  `json:"response"`
}

func (MessageEvent) isEvent() {}

// GetEnvelope returns the event envelope containing observability metadata.
func (e MessageEvent) GetEnvelope() EventEnvelope { return e.Envelope }

// AssistantDeltaEvent carries incremental content parts during streaming generation.
// This wraps llm.ContentPartEvent with our envelope for observability.
// If the model doesn't support streaming, only MessageEvent is emitted.
type AssistantDeltaEvent struct {
	Envelope EventEnvelope        `json:"envelope"`
	Delta    llm.ContentPartEvent `json:"delta"` // The incremental content from the LLM
}

func (AssistantDeltaEvent) isEvent() {}

// GetEnvelope returns the event envelope containing observability metadata.
func (e AssistantDeltaEvent) GetEnvelope() EventEnvelope { return e.Envelope }

// ToolRequestEvent represents a tool invocation request from the LLM.
type ToolRequestEvent struct {
	Envelope EventEnvelope   `json:"envelope"`
	Request  llm.ToolRequest `json:"request"`
}

func (ToolRequestEvent) isEvent() {}

// GetEnvelope returns the event envelope containing observability metadata.
func (e ToolRequestEvent) GetEnvelope() EventEnvelope { return e.Envelope }

// ToolResponseEvent carries the result of a tool execution.
type ToolResponseEvent struct {
	Envelope EventEnvelope    `json:"envelope"`
	Response llm.ToolResponse `json:"response"`
}

func (ToolResponseEvent) isEvent() {}

// GetEnvelope returns the event envelope containing observability metadata.
func (e ToolResponseEvent) GetEnvelope() EventEnvelope { return e.Envelope }

// ErrorEvent is NON-TERMINAL at the runtime layer.
// It reports a recoverable or fatal problem, but the stream only ends
// after a StreamEndEvent is emitted. Fatal paths MUST emit:
//
//	ErrorEvent -> StreamEndEvent{FinishReason: "error"} -> io.EOF
//
// Transport/protocol errors are returned via Recv() error, not as events.
type ErrorEvent struct {
	Envelope EventEnvelope `json:"envelope"`
	Err      error         `json:"-"`
	Message  string        `json:"message"` // Human-readable message
}

func (ErrorEvent) isEvent() {}

// GetEnvelope returns the event envelope containing observability metadata.
func (e ErrorEvent) GetEnvelope() EventEnvelope { return e.Envelope }

// StreamResetEvent signals that a stream is being retried. Consumers should
// discard any accumulated content from the previous attempt.
//
// This event is emitted when the retry interceptor catches a retryable error
// during streaming and restarts the generation.
type StreamResetEvent struct {
	Envelope EventEnvelope `json:"envelope"`
	// Attempt is the retry attempt number (1-based).
	Attempt int `json:"attempt"`
	// Reason describes why the stream is being reset.
	Reason string `json:"reason"`
}

func (StreamResetEvent) isEvent() {}

// GetEnvelope returns the event envelope containing observability metadata.
func (e StreamResetEvent) GetEnvelope() EventEnvelope { return e.Envelope }

// InvocationEndEvent signals completion of the invocation (success or failure).
// This event is ALWAYS the final event in a stream.
// After this event, the event stream ends.
//
// InvocationEndEvent contains only metadata about the invocation completion. The final assistant
// message is available via the last MessageEvent emitted before this event.
//
// On success: InvocationEndEvent{FinishReason: "stop"|"max_turns"|"length"|"input_required"}
// On failure: ErrorEvent -> InvocationEndEvent{FinishReason: "error"}.
type InvocationEndEvent struct {
	Envelope EventEnvelope `json:"envelope"`
	// FinishReason indicates why the invocation ended.
	FinishReason FinishReason `json:"finish_reason"`
	// Usage contains cumulative token usage across ALL turns in this invocation.
	// For multi-turn runs, this is the sum of all model calls (turn 0 + turn 1 + ... + turn N).
	// For per-turn usage breakdown, observe StatusEvent.Usage from each turn.
	// This field is suitable for billing, cost tracking, and quota management.
	Usage *llm.TokenUsage `json:"usage,omitempty"`
	// InputRequiredToolIDs lists tool call IDs that require external input.
	// Only populated when FinishReason is FinishReasonInputRequired.
	// These IDs correspond to ToolRequestEvent.Request.ID from earlier in the stream.
	InputRequiredToolIDs []string `json:"input_required_tool_ids,omitempty"`
	// PendingActions provides typed continuation details for paused invocations.
	PendingActions []PendingAction `json:"pending_actions,omitempty"`
}

func (InvocationEndEvent) isEvent() {}

// GetEnvelope returns the event envelope containing observability metadata.
func (e InvocationEndEvent) GetEnvelope() EventEnvelope { return e.Envelope }

// FinishReason indicates why the agent execution completed.
// Distinct from StatusStage which tracks execution phases.
type FinishReason string

// FinishReason constants indicate why the agent execution ended.
const (
	// FinishReasonStop indicates the LLM naturally completed without requesting tools.
	// This is the successful completion case - the agent finished its work.
	FinishReasonStop FinishReason = "stop"

	// FinishReasonMaxTurns indicates the turn limit was reached.
	// The agent hit the maximum allowed turns before completing.
	FinishReasonMaxTurns FinishReason = "max_turns"

	// FinishReasonLength indicates a context/token limit was hit.
	// The LLM ran out of context space during generation.
	FinishReasonLength FinishReason = "length"

	// FinishReasonInputRequired indicates execution paused waiting for external input.
	// One or more tools require user input, approval, or external data before continuing.
	// The session state is saved and can be resumed when input is provided.
	// Check InvocationEndEvent.InputRequiredToolIDs for the list of waiting tools.
	FinishReasonInputRequired FinishReason = "input_required"

	// FinishReasonError indicates a fatal error occurred.
	// The agent encountered an unrecoverable error during execution.
	FinishReasonError FinishReason = "error"

	// FinishReasonInterrupted indicates execution was canceled.
	// The context was canceled before the agent could complete.
	FinishReasonInterrupted FinishReason = "interrupted"

	// FinishReasonTransfer indicates execution was transferred to another agent.
	// Used for agent handoffs (future feature).
	FinishReasonTransfer FinishReason = "transfer"
)
