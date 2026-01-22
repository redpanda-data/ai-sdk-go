package todo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Item represents a todo item with name and status.
type Item struct {
	Name   string `json:"name"`   // Name of the todo item
	Status string `json:"status"` // Status: PENDING, IN_PROGRESS, COMPLETED, FAILED, ABANDONED
}

// Update represents an update to a specific todo item.
type Update struct {
	Name   string `json:"name"`   // Name of the todo to update
	Status string `json:"status"` // New status to set
}

// UpdateTodoStateRequest represents the input to the update todo state tool.
type UpdateTodoStateRequest struct {
	Updates []Update `json:"updates"`
}

// UpdateTodoStateResponse represents the output from the update todo state tool.
type UpdateTodoStateResponse struct{}

// AddTodoRequest represents the input to the add todo tool.
type AddTodoRequest struct {
	Todos []Item `json:"todos"`
}

// AddTodoResponse represents the output from the add todo tool.
type AddTodoResponse struct{}

// UpdateTodoStateTool implements a tool for updating the status of existing todos.
type UpdateTodoStateTool struct{}

// NewUpdateStateTool creates a new UpdateTodoStateTool instance.
func NewUpdateStateTool() tool.Tool {
	return &UpdateTodoStateTool{}
}

// Definition returns the tool definition for the LLM.
func (*UpdateTodoStateTool) Definition() llm.ToolDefinition {
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"updates": {
				"description": "List of todo status updates to apply",
				"type": "array",
				"items": {
					"type": "object",
					"properties": {
						"name": {
							"type": "string",
							"description": "Name of the todo to update"
						},
						"status": {
							"type": "string",
							"enum": ["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "ABANDONED"],
							"description": "New status to set for the todo"
						}
					},
					"required": ["name", "status"],
					"additionalProperties": false
				}
			}
		},
		"required": ["updates"],
		"additionalProperties": false
	}`)

	return llm.ToolDefinition{
		Name: "update_todos",
		Description: `Update the status of existing todos in your task list. Use this to change the state of specific todos without rebuilding the entire list.

SUPPORTED STATES:
- PENDING: Task not yet started
- IN_PROGRESS: Currently working (limit to ONE task at a time)
- COMPLETED: Task finished successfully  
- FAILED: Task attempted but failed to complete
- ABANDONED: Task intentionally stopped/cancelled without completion

TARGETING:
- Specify the exact name of the todo to update

IMPORTANT RULES:
- Only ONE task should be IN_PROGRESS at any time
- Mark tasks COMPLETED immediately after finishing
- Use FAILED for tasks that were attempted but couldn't be completed
- Use ABANDONED for tasks that are no longer relevant or needed`,
		Parameters: schema,
		Type:       llm.ToolTypeFunction,
	}
}

// Execute processes the update todo state request.
func (t *UpdateTodoStateTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	var req UpdateTodoStateRequest

	err := json.Unmarshal(args, &req)
	if err != nil {
		return nil, fmt.Errorf("failed to parse update todo state request: %w", err)
	}

	// Validate the updates
	err = t.validateUpdates(req.Updates)
	if err != nil {
		return nil, fmt.Errorf("invalid updates: %w", err)
	}

	// The actual todo list updates will be handled by the reconciler
	// Return empty response
	return json.Marshal(UpdateTodoStateResponse{})
}

// validateUpdates validates the structure and constraints of updates.
func (*UpdateTodoStateTool) validateUpdates(updates []Update) error {
	if len(updates) == 0 {
		return errors.New("updates list cannot be empty")
	}

	validStates := map[string]bool{
		"PENDING":     true,
		"IN_PROGRESS": true,
		"COMPLETED":   true,
		"FAILED":      true,
		"ABANDONED":   true,
	}

	inProgressCount := 0

	var inProgressTasks []string

	for i, update := range updates {
		// Must have name
		if update.Name == "" {
			return fmt.Errorf("update %d: name cannot be empty", i)
		}

		// Validate status
		if !validStates[update.Status] {
			return fmt.Errorf("update %d: invalid status %q", i, update.Status)
		}

		// Count in_progress states
		if update.Status == "IN_PROGRESS" {
			inProgressCount++

			inProgressTasks = append(inProgressTasks, update.Name)
		}
	}

	// Warn about multiple in_progress (but don't fail - reconciler will handle)
	if inProgressCount > 1 {
		return fmt.Errorf("only one task can be IN_PROGRESS at a time. These tasks would be IN_PROGRESS: %s", strings.Join(inProgressTasks, ", "))
	}

	return nil
}

// AddTodoTool implements a tool for adding new todos to the task list.
type AddTodoTool struct{}

// NewAddTool creates a new AddTodoTool instance.
func NewAddTool() tool.Tool {
	return &AddTodoTool{}
}

// Definition returns the tool definition for the LLM.
func (*AddTodoTool) Definition() llm.ToolDefinition {
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"todos": {
				"description": "New todos to add to the task list",
				"type": "array",
				"items": {
					"type": "object",
					"properties": {
						"name": {
							"type": "string",
							"minLength": 1,
							"description": "The name/description of the todo task"
						},
						"status": {
							"type": "string",
							"enum": ["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "ABANDONED"],
							"description": "Initial status of the todo"
						}
					},
					"required": ["name", "status"],
					"additionalProperties": false
				}
			}
		},
		"required": ["todos"],
		"additionalProperties": false
	}`)

	return llm.ToolDefinition{
		Name: "add_todos",
		Description: `Add new todos to your existing task list. Use this to expand your task list as new work is discovered.

SUPPORTED STATES:
- PENDING: Task not yet started (typical for new todos)
- IN_PROGRESS: Currently working (use sparingly)
- COMPLETED: Task finished successfully (rare for new todos)
- FAILED: Task attempted but failed to complete  
- ABANDONED: Task intentionally stopped/cancelled

WHEN TO USE:
- New requirements discovered during work
- Breaking down existing tasks into smaller steps
- Adding follow-up tasks after completing others
- Capturing additional work identified during analysis

IMPORTANT RULES:
- New todos typically start as PENDING
- Only mark as IN_PROGRESS if immediately starting work
- Provide clear, actionable descriptions
- Use specific, measurable content`,
		Parameters: schema,
		Type:       llm.ToolTypeFunction, // Explicit: local execution
	}
}

// Execute processes the add todo request.
func (t *AddTodoTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	var req AddTodoRequest

	err := json.Unmarshal(args, &req)
	if err != nil {
		return nil, fmt.Errorf("failed to parse add todo request: %w", err)
	}

	// Validate the new todos
	err = t.validateTodos(req.Todos)
	if err != nil {
		return nil, fmt.Errorf("invalid todos: %w", err)
	}

	// The actual todo list updates will be handled by the reconciler
	// Return empty response
	return json.Marshal(AddTodoResponse{})
}

// validateTodos validates the structure and constraints of new todos.
func (*AddTodoTool) validateTodos(todos []Item) error {
	if len(todos) == 0 {
		return errors.New("todos list cannot be empty")
	}

	validStates := map[string]bool{
		"PENDING":     true,
		"IN_PROGRESS": true,
		"COMPLETED":   true,
		"FAILED":      true,
		"ABANDONED":   true,
	}

	inProgressCount := 0

	var inProgressTasks []string

	for i, todo := range todos {
		if todo.Name == "" {
			return fmt.Errorf("todo %d: name cannot be empty", i)
		}

		if !validStates[todo.Status] {
			return fmt.Errorf("todo %d: invalid status %q", i, todo.Status)
		}

		if todo.Status == "IN_PROGRESS" {
			inProgressCount++

			inProgressTasks = append(inProgressTasks, todo.Name)
		}
	}

	// Warn about multiple in_progress states
	if inProgressCount > 1 {
		return fmt.Errorf("only one task can be IN_PROGRESS at a time. These tasks would be IN_PROGRESS: %s", strings.Join(inProgressTasks, ", "))
	}

	return nil
}
