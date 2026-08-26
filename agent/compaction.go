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
	"fmt"
	"time"
)

// ContextUsage breaks a request's estimated token footprint down by category.
// Values use the runtime's conservative heuristic and are not provider-billed
// usage.
type ContextUsage struct {
	// Total is the sum of every category below.
	Total int `json:"total"`

	// SystemPrompt is the resolved system prompt.
	SystemPrompt int `json:"system_prompt"`

	// ToolDefinitions is the tool schemas sent with every request.
	ToolDefinitions int `json:"tool_definitions"`

	// Text is user and assistant text content.
	Text int `json:"text"`

	// Reasoning is model reasoning content carried in the history.
	Reasoning int `json:"reasoning"`

	// ToolCalls is the tool invocations (name + arguments).
	ToolCalls int `json:"tool_calls"`

	// ToolResults is the tool results, including compaction markers.
	ToolResults int `json:"tool_results"`

	// Framing is the per-message protocol overhead.
	Framing int `json:"framing"`
}

// CompactionPhase identifies which pass produced a CompactionReport.
type CompactionPhase string

// Compaction phases, as reported in CompactionReport.Phase.
const (
	// CompactionPhaseProactive is the top-of-turn pass that runs when the
	// estimated request crosses the trigger.
	CompactionPhaseProactive CompactionPhase = "proactive"

	// CompactionPhaseReactive is the forced pass after a provider rejected
	// the request as too large.
	CompactionPhaseReactive CompactionPhase = "reactive"
)

// CompactionReport describes one context compaction pass: when it ran, why,
// what it removed, and the request's token footprint before and after.
// Aggregate only - applications needing a full transcript must persist the
// event stream separately.
type CompactionReport struct {
	// At is when the pass ran.
	At time.Time `json:"at"`

	// Phase is CompactionPhaseProactive or CompactionPhaseReactive.
	Phase CompactionPhase `json:"phase"`

	// PrunedResults is how many tool results were replaced with markers.
	PrunedResults int `json:"pruned_results"`

	// DroppedMessages is how many whole messages were removed.
	DroppedMessages int `json:"dropped_messages"`

	// Before and After are the request footprint around the pass.
	Before ContextUsage `json:"before"`
	After  ContextUsage `json:"after"`
}

// String renders the report as the one-line human description used by
// terminal UIs and logs.
func (r CompactionReport) String() string {
	s := fmt.Sprintf("pruned %d results, dropped %d messages, %dk -> %dk tokens",
		r.PrunedResults, r.DroppedMessages, r.Before.Total/1000, r.After.Total/1000)

	if r.Phase == CompactionPhaseReactive {
		s += " (after provider overflow)"
	}

	return s
}

// CompactionEvent reports that the runtime rewrote the session to fit the
// model's context window. Emitted at the moment the pass ran, before the
// request it made room for.
type CompactionEvent struct {
	Envelope EventEnvelope    `json:"envelope"`
	Report   CompactionReport `json:"report"`
}

func (CompactionEvent) isEvent() {}

// GetEnvelope returns the event envelope containing observability metadata.
func (e CompactionEvent) GetEnvelope() EventEnvelope { return e.Envelope }
