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

	"github.com/redpanda-data/ai-sdk-go/tool"
)

// RequireInputRequest is the model-issued tool argument for require_input.
type RequireInputRequest struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
}

// RequireInputResponse is the placeholder result the model sees while the
// runtime waits for the user's next message.
type RequireInputResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	// Status is retained for callers that grep for the old reconciler
	// status string. New consumers should rely on the typed pause state
	// instead.
	Status       string `json:"status"`
	InputMessage string `json:"input_message"`
	InputType    string `json:"input_type"`
}

const requireInputDescription = `Use this tool when you need input, clarification, or decisions from the user before proceeding with a task.

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
- Use appropriate type to categorize the input request`

var requireInputSchema = json.RawMessage(`{
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

// validRequireInputTypes is the closed set of supported input types.
var validRequireInputTypes = map[string]bool{
	"clarification": true,
	"decision":      true,
	"information":   true,
	"approval":      true,
}

// NewRequireInputTool returns the require_input built-in tool. The tool
// pauses execution with AwaitReasonUserInput + ResumeWithMessage: the
// runtime stops the invocation and resumes when the next user message
// arrives via runner.Run.
func NewRequireInputTool() tool.Tool {
	return tool.Must(tool.Func(
		tool.Spec{
			Name:        "require_input",
			Description: requireInputDescription,
			InputSchema: requireInputSchema,
			Async:       tool.AsyncUserInput(),
		},
		func(_ context.Context, in RequireInputRequest) (tool.Result[RequireInputResponse], error) {
			if in.Message == "" {
				return tool.Result[RequireInputResponse]{}, errors.New("message cannot be empty")
			}

			if in.Type == "" {
				in.Type = "clarification"
			}

			if !validRequireInputTypes[in.Type] {
				return tool.Result[RequireInputResponse]{}, fmt.Errorf("invalid type %q", in.Type)
			}

			out := RequireInputResponse{
				Success:      true,
				Message:      "Task marked as requiring user input: " + in.Message,
				Status:       "require_input",
				InputMessage: in.Message,
				InputType:    in.Type,
			}

			return tool.NeedInput(out, in.Message), nil
		},
	))
}
