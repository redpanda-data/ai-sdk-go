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

	"github.com/redpanda-data/ai-sdk-go/store/session"
)

const pendingActionsMetadataKey = "agent_pending_actions"

// PendingAction records a continuation the caller must resolve before the agent can continue.
type PendingAction struct {
	// ID is the stable continuation identifier exposed to callers.
	ID string `json:"id"`

	// ToolCallID is the original tool call ID from the model transcript.
	ToolCallID string `json:"tool_call_id"`

	// ToolName is the name of the tool that created this continuation.
	ToolName string `json:"tool_name"`

	// Kind identifies what kind of continuation is required.
	Kind string `json:"kind"`

	// Message is the human-readable description of what is pending.
	Message string `json:"message,omitempty"`

	// InputType categorizes the expected user input when Kind == "user_input".
	InputType string `json:"input_type,omitempty"`

	// RequestedSchema is an optional JSON schema describing expected structured input.
	RequestedSchema any `json:"requested_schema,omitempty"`
}

// GetPendingActions returns the persisted pending actions for the session.
func GetPendingActions(sess *session.State) ([]PendingAction, error) {
	if sess == nil || sess.Metadata == nil {
		return nil, nil
	}

	raw, ok := sess.Metadata[pendingActionsMetadataKey]
	if !ok || raw == nil {
		return nil, nil
	}

	data, err := json.Marshal(raw)
	if err != nil {
		return nil, err
	}

	var actions []PendingAction
	if err := json.Unmarshal(data, &actions); err != nil {
		return nil, err
	}

	return actions, nil
}

// SetPendingActions replaces the session's pending action list.
func SetPendingActions(sess *session.State, actions []PendingAction) {
	if sess == nil {
		return
	}

	if sess.Metadata == nil {
		sess.Metadata = make(map[string]any)
	}

	if len(actions) == 0 {
		delete(sess.Metadata, pendingActionsMetadataKey)
		return
	}

	sess.Metadata[pendingActionsMetadataKey] = actions
}
