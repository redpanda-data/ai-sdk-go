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

// RequireInputResponse represents the output from the require input tool.
type RequireInputResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	Status  string `json:"status"`
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
func (*RequireInputTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	var req RequireInputRequest

	err := json.Unmarshal(args, &req)
	if err != nil {
		return nil, fmt.Errorf("failed to parse require input request: %w", err)
	}

	// Validate the request
	if req.Message == "" {
		return nil, errors.New("message cannot be empty")
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
		return nil, fmt.Errorf("invalid type %q", req.Type)
	}

	response := RequireInputResponse{
		Success: true,
		Message: "Task marked as requiring user input: " + req.Message,
		Status:  "require_input",
	}

	// Include the original request in the response for the reconciler to process
	responseWithDetails := map[string]any{
		"success":       response.Success,
		"message":       response.Message,
		"status":        response.Status,
		"input_message": req.Message,
		"input_type":    req.Type,
	}

	return json.Marshal(responseWithDetails)
}
