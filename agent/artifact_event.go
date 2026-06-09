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

import "github.com/redpanda-data/ai-sdk-go/tool"

// ToolArtifactEvent reports that a tool produced an artifact (image,
// file, plot) via tool.Execution.Actions. The SDK does not store
// artifact bytes itself: adapters and applications consume this event
// to persist or surface the artifact. It is emitted after the
// corresponding ToolResponseEvent in the same turn.
type ToolArtifactEvent struct {
	Envelope EventEnvelope `json:"envelope"`

	// CallID is the tool call that produced the artifact.
	CallID string `json:"call_id"`

	// ToolName is the tool that produced the artifact.
	ToolName string `json:"tool_name"`

	// Artifact carries the artifact payload and its metadata.
	Artifact tool.ArtifactAction `json:"artifact"`
}

// GetEnvelope returns the event envelope.
func (e ToolArtifactEvent) GetEnvelope() EventEnvelope { return e.Envelope }

func (ToolArtifactEvent) isEvent() {}
