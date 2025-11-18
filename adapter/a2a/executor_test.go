package a2a

import (
	"context"
	"encoding/json"
	"log/slog"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const (
	dataTypeToolRequest  = "tool_request"
	dataTypeToolResponse = "tool_response"
)

func getMetadataKeys(metadata map[string]any) []string {
	if metadata == nil {
		return nil
	}

	keys := make([]string, 0, len(metadata))
	for k := range metadata {
		keys = append(keys, k)
	}

	return keys
}

func TestExecutor_Integration_OpenAI(t *testing.T) {
	t.Parallel()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	// Create OpenAI provider
	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err)

	// Create model - using gpt-5
	model, err := provider.NewModel("gpt-5")
	require.NoError(t, err)

	// Create agent
	agentInstance, err := llmagent.New("test-agent", "You are a helpful assistant for testing.", model)
	require.NoError(t, err)

	// Create runner
	sessionStore := session.NewInMemoryStore()
	runnerInstance, err := runner.New(agentInstance, sessionStore)
	require.NoError(t, err)

	// Create A2A executor
	executor := NewExecutor(agentInstance, runnerInstance, slog.Default())

	// Create request context
	reqCtx := &a2asrv.RequestContext{
		ContextID:  "test-context-1",
		TaskID:     "test-task-1",
		StoredTask: nil, // New task
		Message: a2a.NewMessage(
			a2a.MessageRoleUser,
			a2a.TextPart{Text: "Tell me a brief joke about programming."},
		),
	}

	// Create real in-memory queue from a2a-go
	queue := eventqueue.NewInMemoryQueue(100)

	// Execute
	ctx := context.Background()

	// Collect events in background
	events := []a2a.Event{}
	eventsDone := make(chan struct{})

	go func() {
		defer close(eventsDone)

		for {
			event, err := queue.Read(ctx)
			if err != nil {
				return // Queue closed or error
			}

			events = append(events, event)
		}
	}()

	err = executor.Execute(ctx, reqCtx, queue)
	require.NoError(t, err)

	// Close queue to signal no more events
	queue.Close()

	// Wait for event reader to finish
	<-eventsDone

	// Verify events were written
	require.NotEmpty(t, events, "Should have written events to queue")

	t.Logf("Total events written: %d", len(events))

	// Check for task submitted event
	hasSubmitted := false
	hasArtifact := false
	hasCompleted := false

	for i, event := range events {
		t.Logf("Event %d: %T", i, event)

		switch ev := event.(type) {
		case *a2a.TaskStatusUpdateEvent:
			t.Logf("  Status: %s, Final: %v", ev.Status.State, ev.Final)

			if ev.Status.State == a2a.TaskStateSubmitted {
				hasSubmitted = true
			}

			if ev.Status.State == a2a.TaskStateCompleted && ev.Final {
				hasCompleted = true
				// Check for usage metadata
				if ev.Metadata == nil {
					continue
				}

				usage, ok := ev.Metadata["usage"].(map[string]any)
				if !ok {
					continue
				}

				t.Logf("  Token usage: %+v", usage)

				if totalTokens, ok := usage["total_tokens"].(int); ok {
					assert.Positive(t, totalTokens, "Should have token usage")
				}

				if maxInputTokens, ok := usage["max_input_tokens"].(int); ok {
					assert.Positive(t, maxInputTokens, "Should have max_input_tokens")
				}
			}

		case *a2a.TaskArtifactUpdateEvent:
			hasArtifact = true

			t.Logf("  Artifact ID: %s, Append: %v, LastChunk: %v", ev.Artifact.ID, ev.Append, ev.LastChunk)
			t.Logf("  Parts count: %d, Metadata keys: %v", len(ev.Artifact.Parts), getMetadataKeys(ev.Artifact.Metadata))

			if len(ev.Artifact.Parts) > 0 {
				for j, part := range ev.Artifact.Parts {
					if textPart, ok := part.(a2a.TextPart); ok {
						t.Logf("    Part %d: %s", j, textPart.Text)
					}
				}
			}
		}
	}

	assert.True(t, hasSubmitted, "Should have submitted status event")
	assert.True(t, hasArtifact, "Should have artifact events with response text")
	assert.True(t, hasCompleted, "Should have completed status event")

	// Verify final event is completion with final=true
	lastEvent := events[len(events)-1]
	if statusEvent, ok := lastEvent.(*a2a.TaskStatusUpdateEvent); ok {
		assert.Equal(t, a2a.TaskStateCompleted, statusEvent.Status.State, "Final event should be completed status")
		assert.True(t, statusEvent.Final, "Final status event should have Final=true")
	} else {
		t.Errorf("Last event should be TaskStatusUpdateEvent, got %T", lastEvent)
	}

	// Verify we got a text response
	foundText := false

	for _, event := range events {
		if artifactEvent, ok := event.(*a2a.TaskArtifactUpdateEvent); ok {
			for _, part := range artifactEvent.Artifact.Parts {
				if textPart, ok := part.(a2a.TextPart); ok && len(textPart.Text) > 0 {
					foundText = true

					t.Logf("Response text found: %s", textPart.Text)

					break
				}
			}

			if foundText {
				break
			}
		}
	}

	assert.True(t, foundText, "Should have received text response from LLM")
}

func TestExecutor_ToolUse_MessageHistory(t *testing.T) {
	t.Parallel()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	// Create OpenAI provider
	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err)

	// Create model
	model, err := provider.NewModel("gpt-5")
	require.NoError(t, err)

	// Create tool registry with mock weather tool
	toolRegistry := tool.NewRegistry(tool.RegistryConfig{})
	weatherTool := &mockWeatherTool{}
	err = toolRegistry.Register(weatherTool)
	require.NoError(t, err)

	// Create agent with tool registry
	agentInstance, err := llmagent.New(
		"test-agent",
		"You are a test assistant. Use the get_weather tool when asked about weather.",
		model,
		llmagent.WithTools(toolRegistry),
	)
	require.NoError(t, err)

	// Create runner
	sessionStore := session.NewInMemoryStore()
	runnerInstance, err := runner.New(agentInstance, sessionStore)
	require.NoError(t, err)

	// Create A2A executor
	executor := NewExecutor(agentInstance, runnerInstance, slog.Default())

	// Create request asking about weather (should trigger tool use)
	reqCtx := &a2asrv.RequestContext{
		ContextID:  "test-context-tool",
		TaskID:     "test-task-tool",
		StoredTask: nil,
		Message: a2a.NewMessage(
			a2a.MessageRoleUser,
			a2a.TextPart{Text: "What's the weather in San Francisco?"},
		),
	}

	// Create queue
	queue := eventqueue.NewInMemoryQueue(100)

	// Execute
	ctx := context.Background()

	// Collect events
	events := []a2a.Event{}
	eventsDone := make(chan struct{})

	go func() {
		defer close(eventsDone)

		for {
			event, err := queue.Read(ctx)
			if err != nil {
				return
			}

			events = append(events, event)
		}
	}()

	err = executor.Execute(ctx, reqCtx, queue)
	require.NoError(t, err)

	queue.Close()
	<-eventsDone

	// Extract message history from status updates
	var messageHistory []*a2a.Message

	for _, event := range events {
		if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok {
			if statusEvent.Status.Message != nil {
				messageHistory = append(messageHistory, statusEvent.Status.Message)
			}
		}
	}

	// Verify message history structure
	require.NotEmpty(t, messageHistory, "Should have messages in history")

	// Should have at least:
	// 1. Agent message with tool request
	// 2. User message with tool response
	// 3. Agent message with final response
	require.GreaterOrEqual(t, len(messageHistory), 3, "Should have at least 3 messages (tool req, tool resp, final)")

	// Check for tool request in agent message
	foundToolRequest := false

	for _, msg := range messageHistory {
		if msg.Role == a2a.MessageRoleAgent {
			for _, part := range msg.Parts {
				if dataPart, ok := part.(a2a.DataPart); ok {
					if dataType, ok := dataPart.Metadata["data_type"].(string); ok && dataType == dataTypeToolRequest {
						foundToolRequest = true

						t.Logf("Found tool request in agent message: %+v", dataPart.Data)

						// Verify tool request structure
						assert.Contains(t, dataPart.Data, "name", "Tool request should have name")
						assert.Contains(t, dataPart.Data, "id", "Tool request should have id")
						assert.Contains(t, dataPart.Data, "arguments", "Tool request should have arguments")
					}
				}
			}
		}
	}

	assert.True(t, foundToolRequest, "Should have tool request in message history")

	// Check for tool response in user message
	foundToolResponse := false

	for _, msg := range messageHistory {
		if msg.Role == a2a.MessageRoleUser {
			for _, part := range msg.Parts {
				if dataPart, ok := part.(a2a.DataPart); ok {
					if dataType, ok := dataPart.Metadata["data_type"].(string); ok && dataType == dataTypeToolResponse {
						foundToolResponse = true

						t.Logf("Found tool response in user message: %+v", dataPart.Data)

						// Verify tool response structure
						assert.Contains(t, dataPart.Data, "name", "Tool response should have name")
						assert.Contains(t, dataPart.Data, "id", "Tool response should have id")
						// Should have either result or error
						hasResult := dataPart.Data["result"] != nil
						hasError := dataPart.Data["error"] != nil
						assert.True(t, hasResult || hasError, "Tool response should have result or error")
					}
				}
			}
		}
	}

	assert.True(t, foundToolResponse, "Should have tool response in message history")

	// Verify the order: tool request should come before tool response
	toolReqIndex := -1
	toolRespIndex := -1

	for i, msg := range messageHistory {
		for _, part := range msg.Parts {
			dataPart, ok := part.(a2a.DataPart)
			if !ok {
				continue
			}

			dataType, ok := dataPart.Metadata["data_type"].(string)
			if !ok {
				continue
			}

			if dataType == dataTypeToolRequest && toolReqIndex == -1 {
				toolReqIndex = i
			}

			if dataType == dataTypeToolResponse && toolRespIndex == -1 {
				toolRespIndex = i
			}
		}
	}

	if toolReqIndex >= 0 && toolRespIndex >= 0 {
		assert.Less(t, toolReqIndex, toolRespIndex, "Tool request should come before tool response in history")
	}
}

// mockWeatherTool is a simple mock tool for testing.
type mockWeatherTool struct{}

type weatherInput struct {
	Location string `json:"location" jsonschema_description:"The city and state, e.g. San Francisco, CA"`
}

func (m *mockWeatherTool) Definition() llm.ToolDefinition {
	// Use google/jsonschema-go to generate schema from Go type
	schema, err := jsonschema.For[weatherInput](nil)
	if err != nil {
		return llm.ToolDefinition{} // Return empty definition on error
	}

	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return llm.ToolDefinition{} // Return empty definition on error
	}

	return llm.ToolDefinition{
		Name:        "get_weather",
		Description: "Get the current weather for a location",
		Parameters:  schemaBytes,
	}
}

func (m *mockWeatherTool) Execute(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
	return json.RawMessage(`{"temperature": "72°F", "conditions": "sunny"}`), nil
}
