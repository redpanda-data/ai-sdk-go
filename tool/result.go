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

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// PendingKind identifies the kind of continuation required after a tool call.
type PendingKind string

const (
	// PendingKindUserInput indicates the tool needs a human reply before execution can continue.
	PendingKindUserInput PendingKind = "user_input"

	// PendingKindExternalResult indicates the tool started work elsewhere and must be resumed
	// later with a final machine-produced result.
	PendingKindExternalResult PendingKind = "external_result"
)

// Pending describes a continuation requested by a tool invocation.
type Pending struct {
	// Kind specifies what type of continuation is required.
	Kind PendingKind `json:"kind"`

	// Message is a human-readable explanation of what is pending.
	Message string `json:"message,omitempty"`

	// InputType categorizes the user input request (e.g. clarification, approval).
	InputType string `json:"input_type,omitempty"`

	// RequestedSchema describes the input shape when structured data is expected.
	RequestedSchema any `json:"requested_schema,omitempty"`
}

// Result is the outcome of a tool execution.
//
// Tools always return an Output for the LLM transcript. When Pending is non-nil,
// the runtime stores the output, pauses execution, and exposes continuation metadata
// to the caller.
type Result struct {
	Output  json.RawMessage `json:"output,omitempty"`
	Pending *Pending        `json:"pending,omitempty"`
}

// ExecutionResult is the registry-level representation of a tool execution.
type ExecutionResult struct {
	Response *llm.ToolResponse `json:"response,omitempty"`
	Pending  *Pending          `json:"pending,omitempty"`
}
