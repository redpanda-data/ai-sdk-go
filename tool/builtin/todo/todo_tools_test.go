package todo

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/builtin"
)

func TestTodoToolsIntegration(t *testing.T) {
	t.Parallel()

	// Create tool registry
	registry := tool.NewRegistry(tool.RegistryConfig{})

	// Register all builtin tools
	tools := []tool.Tool{
		NewUpdateStateTool(),
		NewAddTool(),
		builtin.NewArtifactEmitTool(),
		builtin.NewRequireInputTool(),
	}

	for _, tool := range tools {
		err := registry.Register(tool)
		require.NoError(t, err, "Failed to register tool")
	}

	ctx := context.Background()

	t.Run("update_todos_tool", func(t *testing.T) {
		t.Parallel()

		updateArgs := `{
			"updates": [
				{
					"name": "Implement feature A",
					"status": "IN_PROGRESS"
				},
				{
					"name": "Test feature B",
					"status": "COMPLETED"
				}
			]
		}`

		toolReq := &llm.ToolRequest{
			Name:      "update_todos",
			Arguments: json.RawMessage(updateArgs),
		}

		result, err := registry.Execute(ctx, toolReq)
		require.NoError(t, err, "Failed to execute update_todos")
		// Verify empty response
		var response UpdateTodoStateResponse
		require.NoError(t, json.Unmarshal(result.Result, &response))
	})

	t.Run("add_todos_tool", func(t *testing.T) {
		t.Parallel()

		addArgs := `{
			"todos": [
				{
					"name": "Deploy to production",
					"status": "PENDING"
				},
				{
					"name": "Monitor metrics",
					"status": "FAILED"
				}
			]
		}`

		toolReq := &llm.ToolRequest{
			Name:      "add_todos",
			Arguments: json.RawMessage(addArgs),
		}

		result, err := registry.Execute(ctx, toolReq)
		require.NoError(t, err, "Failed to execute add_todos")
		// Verify empty response
		var response AddTodoResponse
		require.NoError(t, json.Unmarshal(result.Result, &response))
	})

	t.Run("validation_rejects_invalid_state", func(t *testing.T) {
		t.Parallel()

		invalidArgs := `{
			"updates": [
				{
					"name": "Test task",
					"status": "INVALID_STATE"
				}
			]
		}`

		toolReq := &llm.ToolRequest{
			Name:      "update_todos",
			Arguments: json.RawMessage(invalidArgs),
		}

		result, err := registry.Execute(ctx, toolReq)
		require.NoError(t, err, "Registry should not return Go errors for tool validation failures")
		assert.NotEmpty(t, result.Error, "Expected validation error in ToolResponse")
		assert.Contains(t, result.Error, "invalid status", "Error should mention invalid status")
	})

	t.Run("validation_rejects_empty_name", func(t *testing.T) {
		t.Parallel()

		invalidArgs := `{
			"todos": [
				{
					"name": "",
					"status": "PENDING"
				}
			]
		}`

		toolReq := &llm.ToolRequest{
			Name:      "add_todos",
			Arguments: json.RawMessage(invalidArgs),
		}

		result, err := registry.Execute(ctx, toolReq)
		require.NoError(t, err, "Registry should not return Go errors for tool validation failures")
		assert.NotEmpty(t, result.Error, "Expected validation error in ToolResponse")
		assert.Contains(t, result.Error, "name cannot be empty", "Error should mention empty name")
	})

	t.Run("validation_rejects_multiple_in_progress", func(t *testing.T) {
		t.Parallel()

		invalidArgs := `{
			"todos": [
				{
					"name": "Task 1",
					"status": "IN_PROGRESS"
				},
				{
					"name": "Task 2", 
					"status": "IN_PROGRESS"
				}
			]
		}`

		toolReq := &llm.ToolRequest{
			Name:      "add_todos",
			Arguments: json.RawMessage(invalidArgs),
		}

		result, err := registry.Execute(ctx, toolReq)
		require.NoError(t, err, "Registry should not return Go errors for tool validation failures")
		assert.NotEmpty(t, result.Error, "Expected validation error in ToolResponse")
		assert.Contains(t, result.Error, "IN_PROGRESS", "Error should mention IN_PROGRESS constraint")
	})
}
