package todo_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/builtin/todo"
)

const testTimeout = 60 * time.Second

// TestTodoTools_Integration validates that the todo tools work correctly with
// an actual LLM in a multi-turn conversation scenario.
//
// NOTE: This test is intentionally NOT parallelized because:
// - It makes real external API calls to OpenAI (expensive, rate-limited)
// - All subtests share the same provider/model instances created in parent setup
// - Parallel API calls would increase latency and cost without benefit
// - These are end-to-end integration tests best run sequentially.
//
//nolint:paralleltest // makes real OpenAI API calls (expensive, rate-limited); all subtests share provider/model/registry/context instances
func TestTodoTools_Integration(t *testing.T) {
	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Create OpenAI provider and model
	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName)
	require.NoError(t, err)

	// Verify model supports tools
	caps := model.Capabilities()
	require.True(t, caps.Tools, "Model must support tool calling")

	// Create registry with todo tools
	registry := tool.NewRegistry(tool.RegistryConfig{})

	updateTool := todo.NewUpdateStateTool()
	err = registry.Register(updateTool)
	require.NoError(t, err)

	addTool := todo.NewAddTool()
	err = registry.Register(addTool)
	require.NoError(t, err)

	// Get tool definitions
	toolDefinitions := registry.List()
	require.Len(t, toolDefinitions, 2)

	t.Run("add_todos tool integration", func(t *testing.T) { //nolint:paralleltest // shares parent's provider/model/registry instances and makes external OpenAI API calls
		systemPrompt := `You are a helpful assistant that manages todo lists. When asked to create todos, use the add_todos tool.`
		userRequest := "Please add these three tasks to my todo list: 'Review code changes', 'Update documentation', and 'Run integration tests'. The first two should be pending and the last one should be in progress."

		// Step 1: Initial request
		initialRequest := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleSystem,
					Content: []*llm.Part{
						llm.NewTextPart(systemPrompt),
					},
				},
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart(userRequest),
					},
				},
			},
			Tools: toolDefinitions,
		}

		// Get initial response (should request tool)
		response, err := model.Generate(ctx, initialRequest)
		require.NoError(t, err)
		require.NotNil(t, response)

		// Should want to use the add_todos tool
		toolRequests := response.ToolRequests()
		if len(toolRequests) > 0 {
			// Find the add_todos tool request
			var addTodosRequest *llm.ToolRequest

			for _, req := range toolRequests {
				if req.Name == "add_todos" {
					addTodosRequest = req
					break
				}
			}

			require.NotNil(t, addTodosRequest, "LLM should use add_todos tool")

			// Execute the tool
			toolResponse, err := registry.Execute(ctx, addTodosRequest)
			require.NoError(t, err)
			require.Empty(t, toolResponse.Error, "Tool should execute successfully")

			// Tool returns empty response, just verify it succeeded
			var toolResult map[string]any

			err = json.Unmarshal(toolResponse.Result, &toolResult)
			require.NoError(t, err)
		}
	})

	t.Run("update_todos tool integration", func(t *testing.T) { //nolint:paralleltest // shares parent's provider/model/registry instances and makes external OpenAI API calls
		systemPrompt := `You are a helpful assistant that manages todo lists. When asked to update todos, use the update_todos tool.`
		userRequest := "Please mark 'Review code changes' as completed and change 'Update documentation' to in progress."

		// Step 1: Initial request
		initialRequest := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleSystem,
					Content: []*llm.Part{
						llm.NewTextPart(systemPrompt),
					},
				},
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart(userRequest),
					},
				},
			},
			Tools: toolDefinitions,
		}

		// Get initial response (should request tool)
		response, err := model.Generate(ctx, initialRequest)
		require.NoError(t, err)
		require.NotNil(t, response)

		// Should want to use the update_todos tool
		toolRequests := response.ToolRequests()
		if len(toolRequests) > 0 {
			// Find the update_todos tool request
			var updateTodosRequest *llm.ToolRequest

			for _, req := range toolRequests {
				if req.Name == "update_todos" {
					updateTodosRequest = req
					break
				}
			}

			require.NotNil(t, updateTodosRequest, "LLM should use update_todos tool")

			// Execute the tool
			toolResponse, err := registry.Execute(ctx, updateTodosRequest)
			require.NoError(t, err)
			require.Empty(t, toolResponse.Error, "Tool should execute successfully")

			// Tool returns empty response, just verify it succeeded
			var toolResult map[string]any

			err = json.Unmarshal(toolResponse.Result, &toolResult)
			require.NoError(t, err)
		}
	})

	t.Run("multi-turn todo management workflow", func(t *testing.T) { //nolint:paralleltest // shares parent's provider/model/registry instances and makes external OpenAI API calls
		systemPrompt := `You are a helpful assistant that manages todo lists. Use the todo tools to help users manage their tasks.`

		// Simulate a conversation where we add todos and then update them
		conversationHistory := []llm.Message{
			{
				Role: llm.RoleSystem,
				Content: []*llm.Part{
					llm.NewTextPart(systemPrompt),
				},
			},
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("I need to add two tasks: 'Write unit tests' (pending) and 'Deploy to staging' (pending)."),
				},
			},
		}

		// Step 1: Add todos
		addRequest := &llm.Request{
			Messages: conversationHistory,
			Tools:    toolDefinitions,
		}

		addResponse, err := model.Generate(ctx, addRequest)
		require.NoError(t, err)
		require.NotNil(t, addResponse)

		// Execute any tool calls from the add response
		var lastToolResponse *llm.ToolResponse

		addToolRequests := addResponse.ToolRequests()
		if len(addToolRequests) > 0 {
			for _, toolReq := range addToolRequests {
				if toolReq.Name == "add_todos" {
					toolResp, err := registry.Execute(ctx, toolReq)
					require.NoError(t, err)
					require.Empty(t, toolResp.Error)
					lastToolResponse = toolResp
				}
			}
		}

		// Step 2: Update todos
		conversationHistory = append(conversationHistory, addResponse.Message)
		if lastToolResponse != nil {
			conversationHistory = append(conversationHistory, llm.Message{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewToolResponsePart(lastToolResponse),
				},
			})
		}

		conversationHistory = append(conversationHistory, llm.Message{
			Role: llm.RoleUser,
			Content: []*llm.Part{
				llm.NewTextPart("Now please mark 'Write unit tests' as in progress."),
			},
		})

		updateRequest := &llm.Request{
			Messages: conversationHistory,
			Tools:    toolDefinitions,
		}

		updateResponse, err := model.Generate(ctx, updateRequest)
		require.NoError(t, err)
		require.NotNil(t, updateResponse)

		// Execute any tool calls from the update response
		updateToolRequests := updateResponse.ToolRequests()

		var finalResponse *llm.Response

		if len(updateToolRequests) > 0 {
			for _, toolReq := range updateToolRequests {
				if toolReq.Name == "update_todos" {
					toolResp, err := registry.Execute(ctx, toolReq)
					require.NoError(t, err)
					require.Empty(t, toolResp.Error, "Update tool should execute successfully")

					// Tool returns empty response, just verify it succeeded
					var toolResult map[string]any

					err = json.Unmarshal(toolResp.Result, &toolResult)
					require.NoError(t, err)

					// Continue conversation after tool execution to get final response
					conversationHistory = append(conversationHistory, updateResponse.Message)
					conversationHistory = append(conversationHistory, llm.Message{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewToolResponsePart(toolResp),
						},
					})

					finalRequest := &llm.Request{
						Messages: conversationHistory,
						Tools:    toolDefinitions,
					}

					finalResponse, err = model.Generate(ctx, finalRequest)
					require.NoError(t, err)
					require.NotNil(t, finalResponse)
				}
			}
		}

		// Verify the final response makes sense
		var finalText string
		if finalResponse != nil {
			finalText = finalResponse.TextContent()
		} else {
			finalText = updateResponse.TextContent()
		}

		assert.NotEmpty(t, finalText)
		assert.True(t, strings.Contains(strings.ToLower(finalText), "progress") ||
			strings.Contains(strings.ToLower(finalText), "updated") ||
			strings.Contains(strings.ToLower(finalText), "marked"),
			"Final response should acknowledge the update, got: %s", finalText)
	})
}
