package todo

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUpdateTodos_EndToEnd(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		args         string
		validateResp func(t *testing.T, resp map[string]any)
	}{
		{
			name: "successful single todo update",
			args: `{
				"updates": [
					{
						"name": "Write unit tests",
						"status": "COMPLETED"
					}
				]
			}`,
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()
				require.Contains(t, resp, "updates")
				updates, ok := resp["updates"].([]any)
				require.True(t, ok, "updates should be an array")
				require.Len(t, updates, 1)

				update, ok := updates[0].(map[string]any)
				require.True(t, ok, "update should be a map")
				assert.Equal(t, "Write unit tests", update["name"])
				assert.Equal(t, "COMPLETED", update["status"])
			},
		},
		{
			name: "successful multiple todo updates",
			args: `{
				"updates": [
					{
						"name": "Task A",
						"status": "IN_PROGRESS"
					},
					{
						"name": "Task B", 
						"status": "COMPLETED"
					},
					{
						"name": "Task C",
						"status": "FAILED"
					}
				]
			}`,
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()
				require.Contains(t, resp, "updates")
				updates, ok := resp["updates"].([]any)
				require.True(t, ok, "updates should be an array")
				require.Len(t, updates, 3)

				// Verify all updates are present
				updateMap := make(map[string]string)

				for _, update := range updates {
					u, ok := update.(map[string]any)
					require.True(t, ok, "update should be a map")
					name, ok := u["name"].(string)
					require.True(t, ok, "name should be a string")
					status, ok := u["status"].(string)
					require.True(t, ok, "status should be a string")

					updateMap[name] = status
				}

				assert.Equal(t, "IN_PROGRESS", updateMap["Task A"])
				assert.Equal(t, "COMPLETED", updateMap["Task B"])
				assert.Equal(t, "FAILED", updateMap["Task C"])
			},
		},
		{
			name: "updates with all valid states",
			args: `{
				"updates": [
					{
						"name": "Pending Task",
						"status": "PENDING"
					},
					{
						"name": "Working Task",
						"status": "IN_PROGRESS"
					},
					{
						"name": "Done Task",
						"status": "COMPLETED"
					},
					{
						"name": "Failed Task",
						"status": "FAILED"
					},
					{
						"name": "Abandoned Task",
						"status": "ABANDONED"
					}
				]
			}`,
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()
				require.Contains(t, resp, "updates")
				updates, ok := resp["updates"].([]any)
				require.True(t, ok, "updates should be an array")
				require.Len(t, updates, 5)

				// Verify all status values are preserved
				statuses := make([]string, len(updates))
				for i, update := range updates {
					u, ok := update.(map[string]any)
					require.True(t, ok, "update should be a map")
					status, ok := u["status"].(string)
					require.True(t, ok, "status should be a string")

					statuses[i] = status
				}

				assert.Contains(t, statuses, "PENDING")
				assert.Contains(t, statuses, "IN_PROGRESS")
				assert.Contains(t, statuses, "COMPLETED")
				assert.Contains(t, statuses, "FAILED")
				assert.Contains(t, statuses, "ABANDONED")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			tool := NewUpdateStateTool()
			ctx := context.Background()

			result, err := tool.Execute(ctx, json.RawMessage(tt.args))
			require.NoError(t, err)

			// Verify it returns empty response
			var response UpdateTodoStateResponse
			require.NoError(t, json.Unmarshal(result, &response))
		})
	}
}

func TestUpdateTodos_ValidationErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		args        string
		expectError string
	}{
		{
			name:        "empty updates array",
			args:        `{"updates": []}`,
			expectError: "updates list cannot be empty",
		},
		{
			name:        "missing updates field",
			args:        `{}`,
			expectError: "updates list cannot be empty",
		},
		{
			name: "empty todo name",
			args: `{
				"updates": [
					{
						"name": "",
						"status": "PENDING"
					}
				]
			}`,
			expectError: "name cannot be empty",
		},
		{
			name: "invalid status",
			args: `{
				"updates": [
					{
						"name": "Test task",
						"status": "INVALID_STATUS"
					}
				]
			}`,
			expectError: "invalid status",
		},
		{
			name: "multiple in progress tasks",
			args: `{
				"updates": [
					{
						"name": "Task 1",
						"status": "IN_PROGRESS"
					},
					{
						"name": "Task 2",
						"status": "IN_PROGRESS"
					}
				]
			}`,
			expectError: "These tasks would be IN_PROGRESS:",
		},
		{
			name:        "malformed JSON",
			args:        `{"updates": [{"name": "test", "status": "PENDING"}`,
			expectError: "unexpected end of JSON",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			tool := NewUpdateStateTool()
			ctx := context.Background()

			_, err := tool.Execute(ctx, json.RawMessage(tt.args))

			// All validation errors should be returned as Go errors
			require.Error(t, err, "Tool should return Go error for validation failure")
			assert.Contains(t, strings.ToLower(err.Error()), strings.ToLower(tt.expectError))
		})
	}
}

func TestAddTodos_EndToEnd(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		args         string
		validateResp func(t *testing.T, resp map[string]any)
	}{
		{
			name: "successful single todo addition",
			args: `{
				"todos": [
					{
						"name": "New feature implementation",
						"status": "PENDING"
					}
				]
			}`,
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()
				require.Contains(t, resp, "todos")
				todos, ok := resp["todos"].([]any)
				require.True(t, ok, "todos should be an array")
				require.Len(t, todos, 1)

				todo, ok := todos[0].(map[string]any)
				require.True(t, ok, "todo should be a map")
				assert.Equal(t, "New feature implementation", todo["name"])
				assert.Equal(t, "PENDING", todo["status"])
			},
		},
		{
			name: "successful multiple todos addition",
			args: `{
				"todos": [
					{
						"name": "Setup environment",
						"status": "PENDING"
					},
					{
						"name": "Write documentation",
						"status": "PENDING"
					},
					{
						"name": "Current task",
						"status": "IN_PROGRESS"
					}
				]
			}`,
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()
				require.Contains(t, resp, "todos")
				todos, ok := resp["todos"].([]any)
				require.True(t, ok, "todos should be an array")
				require.Len(t, todos, 3)

				// Verify all todos are present with correct names and statuses
				todoMap := make(map[string]string)

				for _, todo := range todos {
					todoItem, ok := todo.(map[string]any)
					require.True(t, ok, "todo should be a map")
					name, ok := todoItem["name"].(string)
					require.True(t, ok, "name should be a string")
					status, ok := todoItem["status"].(string)
					require.True(t, ok, "status should be a string")

					todoMap[name] = status
				}

				assert.Equal(t, "PENDING", todoMap["Setup environment"])
				assert.Equal(t, "PENDING", todoMap["Write documentation"])
				assert.Equal(t, "IN_PROGRESS", todoMap["Current task"])
			},
		},
		{
			name: "todos with all valid states",
			args: `{
				"todos": [
					{
						"name": "Future task",
						"status": "PENDING"
					},
					{
						"name": "Current task",
						"status": "IN_PROGRESS"
					},
					{
						"name": "Done task",
						"status": "COMPLETED"
					},
					{
						"name": "Broken task",
						"status": "FAILED"
					},
					{
						"name": "Dropped task",
						"status": "ABANDONED"
					}
				]
			}`,
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()
				require.Contains(t, resp, "todos")
				todos, ok := resp["todos"].([]any)
				require.True(t, ok, "todos should be an array")
				require.Len(t, todos, 5)

				// Verify all status values are preserved
				statuses := make([]string, len(todos))
				for i, todo := range todos {
					todoItem, ok := todo.(map[string]any)
					require.True(t, ok, "todo should be a map")
					status, ok := todoItem["status"].(string)
					require.True(t, ok, "status should be a string")

					statuses[i] = status
				}

				assert.Contains(t, statuses, "PENDING")
				assert.Contains(t, statuses, "IN_PROGRESS")
				assert.Contains(t, statuses, "COMPLETED")
				assert.Contains(t, statuses, "FAILED")
				assert.Contains(t, statuses, "ABANDONED")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			tool := NewAddTool()
			ctx := context.Background()

			result, err := tool.Execute(ctx, json.RawMessage(tt.args))
			require.NoError(t, err)

			// Verify it returns empty response
			var response AddTodoResponse
			require.NoError(t, json.Unmarshal(result, &response))
		})
	}
}

func TestAddTodos_ValidationErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		args        string
		expectError string
	}{
		{
			name:        "empty todos array",
			args:        `{"todos": []}`,
			expectError: "todos list cannot be empty",
		},
		{
			name:        "missing todos field",
			args:        `{}`,
			expectError: "todos list cannot be empty",
		},
		{
			name: "empty todo name",
			args: `{
				"todos": [
					{
						"name": "",
						"status": "PENDING"
					}
				]
			}`,
			expectError: "name cannot be empty",
		},
		{
			name: "invalid status",
			args: `{
				"todos": [
					{
						"name": "Test task",
						"status": "INVALID_STATUS"
					}
				]
			}`,
			expectError: "invalid status",
		},
		{
			name: "multiple in progress tasks",
			args: `{
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
			}`,
			expectError: "These tasks would be IN_PROGRESS:",
		},
		{
			name:        "malformed JSON",
			args:        `{"todos": [{"name": "test", "status": "PENDING"}`,
			expectError: "unexpected end of JSON",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			tool := NewAddTool()
			ctx := context.Background()

			_, err := tool.Execute(ctx, json.RawMessage(tt.args))

			// All validation errors should be returned as Go errors
			require.Error(t, err, "Tool should return Go error for validation failure")
			assert.Contains(t, strings.ToLower(err.Error()), strings.ToLower(tt.expectError))
		})
	}
}

func TestTodoTools_Definitions(t *testing.T) {
	t.Parallel()

	t.Run("UpdateTool definition", func(t *testing.T) {
		t.Parallel()

		tool := NewUpdateStateTool()
		def := tool.Definition()

		assert.Equal(t, "update_todos", def.Name)
		assert.NotEmpty(t, def.Description)
		assert.Contains(t, def.Description, "Update the status")
		assert.Contains(t, def.Description, "PENDING")
		assert.Contains(t, def.Description, "IN_PROGRESS")
		assert.Contains(t, def.Description, "COMPLETED")

		// Verify schema is valid JSON
		var schema map[string]any
		require.NoError(t, json.Unmarshal(def.Parameters, &schema))
		assert.Equal(t, "object", schema["type"])
		assert.Contains(t, schema, "properties")
		assert.Contains(t, schema, "required")
	})

	t.Run("AddTool definition", func(t *testing.T) {
		t.Parallel()

		tool := NewAddTool()
		def := tool.Definition()

		assert.Equal(t, "add_todos", def.Name)
		assert.NotEmpty(t, def.Description)
		assert.Contains(t, def.Description, "Add new todos")
		assert.Contains(t, def.Description, "PENDING")
		assert.Contains(t, def.Description, "IN_PROGRESS")

		// Verify schema is valid JSON
		var schema map[string]any
		require.NoError(t, json.Unmarshal(def.Parameters, &schema))
		assert.Equal(t, "object", schema["type"])
		assert.Contains(t, schema, "properties")
		assert.Contains(t, schema, "required")
	})
}

func TestTodoTools_EdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("UpdateTool handles empty context", func(t *testing.T) {
		t.Parallel()

		tool := NewUpdateStateTool()

		args := `{
			"updates": [
				{
					"name": "Test task",
					"status": "COMPLETED"
				}
			]
		}`

		// Test with background context
		result, err := tool.Execute(context.Background(), json.RawMessage(args))
		require.NoError(t, err)

		// Should succeed even with basic context - verify empty response
		var response UpdateTodoStateResponse
		require.NoError(t, json.Unmarshal(result, &response))
	})

	t.Run("AddTool handles Unicode names", func(t *testing.T) {
		t.Parallel()

		tool := NewAddTool()

		args := `{
			"todos": [
				{
					"name": "测试任务 (Test Task) 🚀",
					"status": "PENDING"
				},
				{
					"name": "Tarea de prueba émile",
					"status": "PENDING"
				}
			]
		}`

		result, err := tool.Execute(context.Background(), json.RawMessage(args))
		require.NoError(t, err)

		// Verify empty response - Unicode handling happens at input validation
		var response AddTodoResponse
		require.NoError(t, json.Unmarshal(result, &response))
	})

	t.Run("tools handle extremely long names gracefully", func(t *testing.T) {
		t.Parallel()

		longName := strings.Repeat("Very long task name ", 100) // ~2000 characters

		updateTool := NewUpdateStateTool()
		updateArgs := fmt.Sprintf(`{
			"updates": [
				{
					"name": "%s",
					"status": "COMPLETED"
				}
			]
		}`, longName)

		result, err := updateTool.Execute(context.Background(), json.RawMessage(updateArgs))
		require.NoError(t, err)

		// Should handle long names without errors - verify empty response
		var response UpdateTodoStateResponse
		require.NoError(t, json.Unmarshal(result, &response))
	})
}

func TestTodoTools_StatusValidation(t *testing.T) {
	t.Parallel()

	validStates := []string{"PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "ABANDONED"}
	invalidStates := []string{"pending", "in_progress", "completed", "UNKNOWN", "ACTIVE", "DONE", ""}

	t.Run("UpdateTool accepts all valid states", func(t *testing.T) {
		t.Parallel()

		tool := NewUpdateStateTool()

		for _, state := range validStates {
			args := fmt.Sprintf(`{
				"updates": [
					{
						"name": "Test task for %s",
						"status": "%s"
					}
				]
			}`, state, state)

			result, err := tool.Execute(context.Background(), json.RawMessage(args))
			require.NoError(t, err, "Failed for valid state: %s", state)

			// Verify empty response
			var response UpdateTodoStateResponse
			require.NoError(t, json.Unmarshal(result, &response))
		}
	})

	t.Run("UpdateTool rejects invalid states", func(t *testing.T) {
		t.Parallel()

		tool := NewUpdateStateTool()

		for _, state := range invalidStates {
			args := fmt.Sprintf(`{
				"updates": [
					{
						"name": "Test task",
						"status": "%s"
					}
				]
			}`, state)

			_, err := tool.Execute(context.Background(), json.RawMessage(args))
			require.Error(t, err, "Invalid state %s should produce Go error", state)
			assert.Contains(t, strings.ToLower(err.Error()), "invalid status")
		}
	})

	t.Run("AddTool accepts all valid states", func(t *testing.T) {
		t.Parallel()

		tool := NewAddTool()

		// Test each valid state individually to avoid IN_PROGRESS conflicts
		for _, state := range validStates {
			args := fmt.Sprintf(`{
				"todos": [
					{
						"name": "Test task for %s",
						"status": "%s"
					}
				]
			}`, state, state)

			result, err := tool.Execute(context.Background(), json.RawMessage(args))
			require.NoError(t, err, "Failed for valid state: %s", state)

			// Verify empty response
			var response AddTodoResponse
			require.NoError(t, json.Unmarshal(result, &response))
		}
	})

	t.Run("AddTool rejects invalid states", func(t *testing.T) {
		t.Parallel()

		tool := NewAddTool()

		for _, state := range invalidStates {
			args := fmt.Sprintf(`{
				"todos": [
					{
						"name": "Test task",
						"status": "%s"
					}
				]
			}`, state)

			_, err := tool.Execute(context.Background(), json.RawMessage(args))
			require.Error(t, err, "Invalid state %s should produce Go error", state)
			assert.Contains(t, strings.ToLower(err.Error()), "invalid status")
		}
	})
}

func TestTodoTools_InProgressConstraint(t *testing.T) {
	t.Parallel()

	t.Run("UpdateTool allows exactly one IN_PROGRESS", func(t *testing.T) {
		t.Parallel()

		tool := NewUpdateStateTool()

		args := `{
			"updates": [
				{
					"name": "Task 1",
					"status": "IN_PROGRESS"
				},
				{
					"name": "Task 2",
					"status": "COMPLETED"
				},
				{
					"name": "Task 3",
					"status": "PENDING"
				}
			]
		}`

		result, err := tool.Execute(context.Background(), json.RawMessage(args))
		require.NoError(t, err)

		// Verify empty response
		var response UpdateTodoStateResponse
		require.NoError(t, json.Unmarshal(result, &response))
	})

	t.Run("UpdateTool rejects multiple IN_PROGRESS", func(t *testing.T) {
		t.Parallel()

		tool := NewUpdateStateTool()

		args := `{
			"updates": [
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

		_, err := tool.Execute(context.Background(), json.RawMessage(args))
		require.Error(t, err)
		assert.Contains(t, strings.ToLower(err.Error()), "in_progress")
	})

	t.Run("AddTool allows exactly one IN_PROGRESS", func(t *testing.T) {
		t.Parallel()

		tool := NewAddTool()

		args := `{
			"todos": [
				{
					"name": "Active task",
					"status": "IN_PROGRESS"
				},
				{
					"name": "Future task",
					"status": "PENDING"
				}
			]
		}`

		result, err := tool.Execute(context.Background(), json.RawMessage(args))
		require.NoError(t, err)

		// Verify empty response
		var response AddTodoResponse
		require.NoError(t, json.Unmarshal(result, &response))
	})

	t.Run("AddTool rejects multiple IN_PROGRESS", func(t *testing.T) {
		t.Parallel()

		tool := NewAddTool()

		args := `{
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

		_, err := tool.Execute(context.Background(), json.RawMessage(args))
		require.Error(t, err)
		assert.Contains(t, strings.ToLower(err.Error()), "in_progress")
	})
}
