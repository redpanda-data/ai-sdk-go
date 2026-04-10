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

package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// RequireInputRequest represents the input to the require input tool.
type RequireInputRequest struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
}

// RequireInputTool implements a tool for marking tasks as requiring user input.
type RequireInputTool struct{}

// NewRequireInputTool creates a new RequireInputTool instance.
func NewRequireInputTool() tool.Tool {
	return &RequireInputTool{}
}

// Definition returns the tool definition for the LLM.
func (*RequireInputTool) Definition() llm.ToolDefinition {
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"message": {
				"type": "string",
				"minLength": 1,
				"description": "A clear message explaining what input is needed from the user"
			},
			"type": {
				"type": "string",
				"enum": ["clarification", "decision", "information", "approval"],
				"description": "The type of input needed: clarification (unclear requirements), decision (user choice needed), information (missing data), approval (permission required)"
			}
		},
		"required": ["message"],
		"additionalProperties": false
	}`)

	return llm.ToolDefinition{
		Name: "require_input",
		Description: `Use this tool when you need input, clarification, or decisions from the user before proceeding with a task.

WHEN TO USE:
- Requirements are unclear or ambiguous
- Multiple implementation options exist and user choice is needed
- Missing information required to complete the task
- User approval needed before making significant changes
- Task cannot proceed without user guidance

WHEN NOT TO USE:
- For simple questions that don't block task progress
- When reasonable defaults can be assumed
- For purely informational updates

IMPORTANT:
- Provide a clear, specific message about what input is needed
- Use appropriate type to categorize the input request
- This will pause task execution until user responds`,
		Parameters: schema,
		Type:       llm.ToolTypeFunction,
	}
}

// Execute processes the require input request.
func (*RequireInputTool) Execute(_ context.Context, args json.RawMessage) (tool.Result, error) {
	var req RequireInputRequest

	err := json.Unmarshal(args, &req)
	if err != nil {
		return tool.Result{}, fmt.Errorf("failed to parse require input request: %w", err)
	}

	// Validate the request
	if req.Message == "" {
		return tool.Result{}, errors.New("message cannot be empty")
	}

	// Set default type if not provided
	if req.Type == "" {
		req.Type = "clarification"
	}

	// Validate type
	validTypes := map[string]bool{
		"clarification": true,
		"decision":      true,
		"information":   true,
		"approval":      true,
	}
	if !validTypes[req.Type] {
		return tool.Result{}, fmt.Errorf("invalid type %q", req.Type)
	}

	// Include the original request in the response for the reconciler to process
	responseWithDetails := map[string]any{
		"message":       "Awaiting user " + req.Type + ": " + req.Message,
		"status":        "awaiting_input",
		"input_message": req.Message,
		"input_type":    req.Type,
	}

	output, err := json.Marshal(responseWithDetails)
	if err != nil {
		return tool.Result{}, err
	}

	return tool.Result{
		Output: output,
		Pending: &tool.Pending{
			Kind:      tool.PendingKindUserInput,
			Message:   req.Message,
			InputType: req.Type,
		},
	}, nil
}
