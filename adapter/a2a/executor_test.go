package a2a

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
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
	ctx := context.Background()
	queueMgr := eventqueue.NewInMemoryManager(eventqueue.WithQueueBufferSize(100))
	// Create separate reader and writer queues (broadcast pattern - writer's events go to other queues)
	readerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)
	writerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)

	// Collect events in background
	events := []a2a.Event{}
	eventsDone := make(chan struct{})

	go func() {
		defer close(eventsDone)

		for {
			event, err := readerQueue.Read(ctx)
			if err != nil {
				return // Queue closed or error
			}

			events = append(events, event)

			// Exit when we see the final event to ensure we've read everything
			if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Final {
				return
			}
		}
	}()

	err = executor.Execute(ctx, reqCtx, writerQueue)
	require.NoError(t, err)

	// Wait for event reader to finish (exits when it sees Final=true)
	<-eventsDone

	// Close queues after reader is done
	writerQueue.Close()
	readerQueue.Close()

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

			// Check for per-message usage metadata on working state events with agent messages
			isWorkingAgentMessage := ev.Status.State == a2a.TaskStateWorking &&
				ev.Status.Message != nil &&
				ev.Status.Message.Role == a2a.MessageRoleAgent
			if isWorkingAgentMessage && ev.Status.Message.Metadata != nil {
				if usage, ok := ev.Status.Message.Metadata["usage"].(map[string]any); ok {
					t.Logf("  Per-message token usage: %+v", usage)

					if totalTokens, ok := usage["total_tokens"].(int); ok {
						assert.Positive(t, totalTokens, "Should have per-message token usage")
					}
				}
			}

			if ev.Status.State == a2a.TaskStateCompleted && ev.Final {
				hasCompleted = true
				// Check for aggregated usage metadata
				if ev.Metadata == nil {
					continue
				}

				usage, ok := ev.Metadata["usage"].(map[string]any)
				if !ok {
					continue
				}

				t.Logf("  Aggregated token usage: %+v", usage)

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
	ctx := context.Background()
	queueMgr := eventqueue.NewInMemoryManager(eventqueue.WithQueueBufferSize(100))
	// Create separate reader and writer queues (broadcast pattern - writer's events go to other queues)
	readerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)
	writerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)

	// Collect events
	events := []a2a.Event{}
	eventsDone := make(chan struct{})
	finalEventSeen := make(chan struct{})

	go func() {
		defer close(eventsDone)

		for {
			event, err := readerQueue.Read(ctx)
			if err != nil {
				return
			}

			events = append(events, event)

			// Check if this is a final event
			if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Final {
				close(finalEventSeen)
			}
		}
	}()

	err = executor.Execute(ctx, reqCtx, writerQueue)
	require.NoError(t, err)

	// Wait for reader to see the final event, then close both queues
	<-finalEventSeen
	writerQueue.Close()
	readerQueue.Close()
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

func TestExecutor_SessionPersistence_Mock(t *testing.T) {
	t.Parallel()

	// Create fake model with stateful responses
	model := fakellm.NewFakeModel()

	// Use a custom response builder that checks the conversation history
	model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		lastUserMsg := req.Messages[len(req.Messages)-1].TextContent()

		// Check if user is telling us their favorite color
		if strings.Contains(strings.ToLower(lastUserMsg), "favorite color is blue") {
			return &llm.Response{
				Message: llm.NewMessage(llm.RoleAssistant,
					llm.NewTextPart("Got it! I'll remember that your favorite color is blue.")),
				FinishReason: llm.FinishReasonStop,
				Usage: &llm.TokenUsage{
					InputTokens:  10,
					OutputTokens: 15,
					TotalTokens:  25,
				},
			}, nil
		}

		// Check if user is asking what their favorite color is
		if strings.Contains(strings.ToLower(lastUserMsg), "what is my favorite color") {
			// Look back in conversation history for the color
			for _, msg := range req.Messages {
				if strings.Contains(strings.ToLower(msg.TextContent()), "favorite color is blue") {
					return &llm.Response{
						Message: llm.NewMessage(llm.RoleAssistant,
							llm.NewTextPart("Your favorite color is blue.")),
						FinishReason: llm.FinishReasonStop,
						Usage: &llm.TokenUsage{
							InputTokens:  20,
							OutputTokens: 8,
							TotalTokens:  28,
						},
					}, nil
				}
			}
			// If not found in history, say we don't know
			return &llm.Response{
				Message: llm.NewMessage(llm.RoleAssistant,
					llm.NewTextPart("I don't know your favorite color yet.")),
				FinishReason: llm.FinishReasonStop,
				Usage: &llm.TokenUsage{
					InputTokens:  15,
					OutputTokens: 10,
					TotalTokens:  25,
				},
			}, nil
		}

		// Default response
		return &llm.Response{
			Message: llm.NewMessage(llm.RoleAssistant,
				llm.NewTextPart("I'm here to help!")),
			FinishReason: llm.FinishReasonStop,
			Usage: &llm.TokenUsage{
				InputTokens:  5,
				OutputTokens: 5,
				TotalTokens:  10,
			},
		}, nil
	})

	// Create agent
	agentInstance, err := llmagent.New("test-agent", "You are a helpful assistant.", model)
	require.NoError(t, err)

	// Create runner with shared session store
	sessionStore := session.NewInMemoryStore()
	runnerInstance, err := runner.New(agentInstance, sessionStore)
	require.NoError(t, err)

	// Create A2A executor
	executor := NewExecutor(agentInstance, runnerInstance, slog.Default())

	ctx := context.Background()
	contextID := "test-session-context-mock"

	// First request: Tell the agent your favorite color
	t.Log("=== First request: Setting favorite color to blue ===")

	reqCtx1 := &a2asrv.RequestContext{
		ContextID:  contextID,
		TaskID:     "test-task-1",
		StoredTask: nil,
		Message: a2a.NewMessage(
			a2a.MessageRoleUser,
			a2a.TextPart{Text: "My favorite color is blue. Please remember this."},
		),
	}

	queueMgr := eventqueue.NewInMemoryManager(eventqueue.WithQueueBufferSize(100))
	// Create separate reader and writer queues (broadcast pattern - writer's events go to other queues)
	readerQueue1, err := queueMgr.GetOrCreate(ctx, reqCtx1.TaskID)
	require.NoError(t, err)
	writerQueue1, err := queueMgr.GetOrCreate(ctx, reqCtx1.TaskID)
	require.NoError(t, err)

	events1 := []a2a.Event{}
	eventsDone1 := make(chan struct{})
	finalEventSeen1 := make(chan struct{})

	go func() {
		defer close(eventsDone1)

		for {
			event, err := readerQueue1.Read(ctx)
			if err != nil {
				return
			}

			events1 = append(events1, event)

			// Check if this is a final event
			if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Final {
				close(finalEventSeen1)
			}
		}
	}()

	err = executor.Execute(ctx, reqCtx1, writerQueue1)
	require.NoError(t, err)

	// Wait for reader to see the final event, then close both queues
	<-finalEventSeen1
	writerQueue1.Close()
	readerQueue1.Close()
	<-eventsDone1

	// Extract first response text
	var firstResponse string

	for _, event := range events1 {
		if artifactEvent, ok := event.(*a2a.TaskArtifactUpdateEvent); ok {
			for _, part := range artifactEvent.Artifact.Parts {
				if textPart, ok := part.(a2a.TextPart); ok {
					firstResponse += textPart.Text //nolint:perfsprint // Test readability over performance
				}
			}
		}
	}

	require.NotEmpty(t, firstResponse, "Should have received response from first request")
	t.Logf("First response: %s", firstResponse)

	// Check session was saved
	savedSession, err := sessionStore.Load(ctx, contextID)
	require.NoError(t, err, "Session should be saved after first request")
	require.Len(t, savedSession.Messages, 2, "Session should have 2 messages (user + assistant)")
	t.Logf("Session has %d messages after first request", len(savedSession.Messages))

	// Second request: Ask what their favorite color is
	t.Log("=== Second request: Asking for favorite color ===")

	reqCtx2 := &a2asrv.RequestContext{
		ContextID:  contextID, // Same context ID for session persistence
		TaskID:     "test-task-2",
		StoredTask: nil,
		Message: a2a.NewMessage(
			a2a.MessageRoleUser,
			a2a.TextPart{Text: "What is my favorite color?"},
		),
	}

	readerQueue2, err := queueMgr.GetOrCreate(ctx, reqCtx2.TaskID)
	require.NoError(t, err)
	writerQueue2, err := queueMgr.GetOrCreate(ctx, reqCtx2.TaskID)
	require.NoError(t, err)

	events2 := []a2a.Event{}
	eventsDone2 := make(chan struct{})
	finalEventSeen2 := make(chan struct{})

	go func() {
		defer close(eventsDone2)

		for {
			event, err := readerQueue2.Read(ctx)
			if err != nil {
				return
			}

			events2 = append(events2, event)

			// Check if this is a final event
			if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Final {
				close(finalEventSeen2)
			}
		}
	}()

	err = executor.Execute(ctx, reqCtx2, writerQueue2)
	require.NoError(t, err)

	// Wait for reader to see the final event, then close both queues
	<-finalEventSeen2
	writerQueue2.Close()
	readerQueue2.Close()
	<-eventsDone2

	// Extract second response text
	var secondResponse string

	for _, event := range events2 {
		if artifactEvent, ok := event.(*a2a.TaskArtifactUpdateEvent); ok {
			for _, part := range artifactEvent.Artifact.Parts {
				if textPart, ok := part.(a2a.TextPart); ok {
					secondResponse += textPart.Text //nolint:perfsprint // Test readability over performance
				}
			}
		}
	}

	require.NotEmpty(t, secondResponse, "Should have received response from second request")
	t.Logf("Second response: %s", secondResponse)

	// Verify the agent remembers the color is blue
	secondResponseLower := strings.ToLower(secondResponse)
	assert.Contains(t, secondResponseLower, "blue", "Agent should remember the favorite color is blue from the previous message")

	// Verify final session has all 4 messages (2 user + 2 assistant)
	finalSession, err := sessionStore.Load(ctx, contextID)
	require.NoError(t, err)
	assert.Len(t, finalSession.Messages, 4, "Session should have 4 messages total")
}

func TestExecutor_SessionPersistence(t *testing.T) {
	t.Parallel()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	// Create OpenAI provider
	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err)

	// Create model
	model, err := provider.NewModel("gpt-5")
	require.NoError(t, err)

	// Create agent
	agentInstance, err := llmagent.New("test-agent", "You are a helpful assistant. Remember information from previous messages in the conversation.", model)
	require.NoError(t, err)

	// Create runner with shared session store
	sessionStore := session.NewInMemoryStore()
	runnerInstance, err := runner.New(agentInstance, sessionStore)
	require.NoError(t, err)

	// Create A2A executor
	executor := NewExecutor(agentInstance, runnerInstance, slog.Default())

	ctx := context.Background()
	contextID := "test-session-context"

	// First request: Tell the agent your favorite color
	t.Log("=== First request: Setting favorite color to blue ===")

	reqCtx1 := &a2asrv.RequestContext{
		ContextID:  contextID,
		TaskID:     "test-task-1",
		StoredTask: nil,
		Message: a2a.NewMessage(
			a2a.MessageRoleUser,
			a2a.TextPart{Text: "My favorite color is blue. Please remember this."},
		),
	}

	queueMgr := eventqueue.NewInMemoryManager(eventqueue.WithQueueBufferSize(100))
	// Create separate reader and writer queues (broadcast pattern - writer's events go to other queues)
	readerQueue1, err := queueMgr.GetOrCreate(ctx, reqCtx1.TaskID)
	require.NoError(t, err)
	writerQueue1, err := queueMgr.GetOrCreate(ctx, reqCtx1.TaskID)
	require.NoError(t, err)

	events1 := []a2a.Event{}
	eventsDone1 := make(chan struct{})
	finalEventSeen1 := make(chan struct{})

	go func() {
		defer close(eventsDone1)

		for {
			event, err := readerQueue1.Read(ctx)
			if err != nil {
				return
			}

			events1 = append(events1, event)

			// Check if this is a final event
			if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Final {
				close(finalEventSeen1)
			}
		}
	}()

	err = executor.Execute(ctx, reqCtx1, writerQueue1)
	require.NoError(t, err)

	// Wait for reader to see the final event, then close both queues
	<-finalEventSeen1
	writerQueue1.Close()
	readerQueue1.Close()
	<-eventsDone1

	t.Logf("First request completed, waiting for events to finish processing")

	// Extract first response text
	var firstResponse string

	for _, event := range events1 {
		if artifactEvent, ok := event.(*a2a.TaskArtifactUpdateEvent); ok {
			for _, part := range artifactEvent.Artifact.Parts {
				if textPart, ok := part.(a2a.TextPart); ok {
					firstResponse += textPart.Text //nolint:perfsprint // Test readability over performance
				}
			}
		}
	}

	require.NotEmpty(t, firstResponse, "Should have received response from first request")
	t.Logf("First response: %s", firstResponse)

	// Check session was saved
	savedSession, err := sessionStore.Load(ctx, contextID)
	require.NoError(t, err, "Session should be saved after first request")
	t.Logf("Session has %d messages after first request", len(savedSession.Messages))

	for i, msg := range savedSession.Messages {
		t.Logf("  Message %d: role=%s, content parts=%d", i, msg.Role, len(msg.Content))
	}

	// Second request: Ask what their favorite color is
	t.Log("=== Second request: Asking for favorite color ===")

	reqCtx2 := &a2asrv.RequestContext{
		ContextID:  contextID, // Same context ID for session persistence
		TaskID:     "test-task-2",
		StoredTask: nil,
		Message: a2a.NewMessage(
			a2a.MessageRoleUser,
			a2a.TextPart{Text: "What is my favorite color?"},
		),
	}

	readerQueue2, err := queueMgr.GetOrCreate(ctx, reqCtx2.TaskID)
	require.NoError(t, err)
	writerQueue2, err := queueMgr.GetOrCreate(ctx, reqCtx2.TaskID)
	require.NoError(t, err)

	events2 := []a2a.Event{}
	eventsDone2 := make(chan struct{})
	finalEventSeen2 := make(chan struct{})

	go func() {
		defer close(eventsDone2)

		for {
			event, err := readerQueue2.Read(ctx)
			if err != nil {
				return
			}

			events2 = append(events2, event)

			// Check if this is a final event
			if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Final {
				close(finalEventSeen2)
			}
		}
	}()

	err = executor.Execute(ctx, reqCtx2, writerQueue2)
	require.NoError(t, err)

	// Wait for reader to see the final event, then close both queues
	<-finalEventSeen2
	writerQueue2.Close()
	readerQueue2.Close()
	<-eventsDone2

	// Extract second response text
	var secondResponse string

	for _, event := range events2 {
		if artifactEvent, ok := event.(*a2a.TaskArtifactUpdateEvent); ok {
			for _, part := range artifactEvent.Artifact.Parts {
				if textPart, ok := part.(a2a.TextPart); ok {
					secondResponse += textPart.Text //nolint:perfsprint // Test readability over performance
				}
			}
		}
	}

	require.NotEmpty(t, secondResponse, "Should have received response from second request")
	t.Logf("Second response: %s", secondResponse)

	// Verify the agent remembers the color is blue
	// The response should contain "blue" (case-insensitive)
	secondResponseLower := strings.ToLower(secondResponse)
	assert.Contains(t, secondResponseLower, "blue", "Agent should remember the favorite color is blue from the previous message")

	// Verify both requests completed successfully
	for i, events := range [][]a2a.Event{events1, events2} {
		hasCompleted := false

		for _, event := range events {
			if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok {
				if statusEvent.Status.State == a2a.TaskStateCompleted && statusEvent.Final {
					hasCompleted = true
					break
				}
			}
		}

		assert.True(t, hasCompleted, "Request %d should have completed successfully", i+1)
	}
}

func TestExecutor_SessionPersistence_Cancelled(t *testing.T) {
	t.Parallel()

	// Create fake model that responds
	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondText("This is a response.")

	// Create agent
	agentInstance, err := llmagent.New("test-agent", "You are a helpful assistant.", model)
	require.NoError(t, err)

	// Create runner with shared session store
	sessionStore := session.NewInMemoryStore()
	runnerInstance, err := runner.New(agentInstance, sessionStore)
	require.NoError(t, err)

	// Create A2A executor
	executor := NewExecutor(agentInstance, runnerInstance, slog.Default())

	// Create cancellable context
	ctx, cancel := context.WithCancel(context.Background())
	contextID := "test-session-context-cancelled"

	reqCtx := &a2asrv.RequestContext{
		ContextID:  contextID,
		TaskID:     "test-task-1",
		StoredTask: nil,
		Message: a2a.NewMessage(
			a2a.MessageRoleUser,
			a2a.TextPart{Text: "Tell me something."},
		),
	}

	queueMgr := eventqueue.NewInMemoryManager(eventqueue.WithQueueBufferSize(100))
	// Create separate reader and writer queues (broadcast pattern - writer's events go to other queues)
	readerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)
	writerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)

	events := []a2a.Event{}
	eventsDone := make(chan struct{})

	// Use background context for reading queue so we can receive the canceled status event
	readCtx := context.Background()

	go func() {
		defer close(eventsDone)

		for {
			event, err := readerQueue.Read(readCtx)
			if err != nil {
				return
			}

			events = append(events, event)

			// Cancel after receiving the first artifact (response started)
			if _, ok := event.(*a2a.TaskArtifactUpdateEvent); ok {
				t.Log("Cancelling context after receiving artifact")
				cancel()
			}
		}
	}()

	err = executor.Execute(ctx, reqCtx, writerQueue)
	require.NoError(t, err, "Execute should not return error even for cancellation")

	writerQueue.Close()
	readerQueue.Close()
	<-eventsDone

	t.Logf("Received %d events", len(events))

	// Log all events
	for i, event := range events {
		t.Logf("Event %d: %T", i, event)

		if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok {
			t.Logf("  Status: state=%s, final=%v", statusEvent.Status.State, statusEvent.Final)
		}
	}

	// Find the canceled status event
	var canceledEvent *a2a.TaskStatusUpdateEvent

	for _, event := range events {
		if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok {
			if statusEvent.Status.State == a2a.TaskStateCanceled {
				canceledEvent = statusEvent
				break
			}
		}
	}

	// Assert we got a canceled status event with error details
	require.NotNil(t, canceledEvent, "Should have a TaskStateCanceled event")
	assert.True(t, canceledEvent.Final, "Canceled event should be marked as final")
	require.NotNil(t, canceledEvent.Status.Message, "Canceled status should include error message")

	// Extract error text
	var errorText string

	for _, part := range canceledEvent.Status.Message.Parts {
		if textPart, ok := part.(a2a.TextPart); ok {
			errorText += textPart.Text //nolint:perfsprint // Test readability over performance
		}
	}

	t.Logf("Error message in canceled status: %s", errorText)
	assert.Contains(t, errorText, "context canceled", "Error message should indicate context was canceled")

	// The key assertion: Even though we cancelled, progress should be saved
	savedSession, err := sessionStore.Load(context.Background(), contextID)
	if err == nil {
		t.Logf("Session was saved with %d messages despite cancellation", len(savedSession.Messages))
		// Should have at least the user message
		assert.GreaterOrEqual(t, len(savedSession.Messages), 1, "Should have saved at least the user message")

		if len(savedSession.Messages) >= 1 {
			assert.Equal(t, llm.RoleUser, savedSession.Messages[0].Role)
			assert.Contains(t, savedSession.Messages[0].TextContent(), "Tell me something")
		}
	} else {
		t.Logf("Session not found after cancellation: %v", err)
	}
}

func TestExecutor_ErrorHandling(t *testing.T) {
	t.Parallel()

	// Create fake model that returns an error simulating an API error
	model := fakellm.NewFakeModel()
	apiError := errors.New("API call failed: received error while streaming: {\"type\":\"invalid_request_error\",\"code\":\"context_length_exceeded\",\"message\":\"Your input exceeds the context window of this model. Please adjust your input and try again.\",\"param\":\"input\"}")
	model.When(fakellm.Any()).ThenError(apiError)

	// Create agent
	agentInstance, err := llmagent.New("test-agent", "You are a helpful assistant.", model)
	require.NoError(t, err)

	// Create runner
	sessionStore := session.NewInMemoryStore()
	runnerInstance, err := runner.New(agentInstance, sessionStore)
	require.NoError(t, err)

	// Create A2A executor
	executor := NewExecutor(agentInstance, runnerInstance, slog.Default())

	// Create request context
	reqCtx := &a2asrv.RequestContext{
		ContextID:  "test-context-error",
		TaskID:     "test-task-error",
		StoredTask: nil,
		Message: a2a.NewMessage(
			a2a.MessageRoleUser,
			a2a.TextPart{Text: "Tell me something."},
		),
	}

	// Create queue
	ctx := context.Background()
	queueMgr := eventqueue.NewInMemoryManager(eventqueue.WithQueueBufferSize(100))
	// Create separate reader and writer queues (broadcast pattern - writer's events go to other queues)
	readerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)
	writerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)

	events := []a2a.Event{}
	eventsDone := make(chan struct{})
	finalEventSeen := make(chan struct{})

	go func() {
		defer close(eventsDone)

		for {
			event, err := readerQueue.Read(ctx)
			if err != nil {
				return
			}

			events = append(events, event)

			// Check if this is a final event
			if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Final {
				close(finalEventSeen)
			}
		}
	}()

	// Execute should NOT return an error for agent failures - those should be communicated via events
	// Only queue write failures should return errors
	err = executor.Execute(ctx, reqCtx, writerQueue)
	require.NoError(t, err, "Execute should NOT return error for agent failures - they should be communicated via task status events")

	// Wait for reader to see the final event, then close both queues
	<-finalEventSeen
	writerQueue.Close()
	readerQueue.Close()
	<-eventsDone

	// Verify we got the expected events
	require.NotEmpty(t, events, "Should have written events to queue")

	t.Logf("Total events: %d", len(events))

	for i, event := range events {
		t.Logf("Event %d: %T", i, event)
	}

	// Find the failed status event
	var failedEvent *a2a.TaskStatusUpdateEvent

	for _, event := range events {
		if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok {
			t.Logf("Status event: state=%s, final=%v, message=%v", statusEvent.Status.State, statusEvent.Final, statusEvent.Status.Message)

			if statusEvent.Status.State == a2a.TaskStateFailed {
				failedEvent = statusEvent
				break
			}
		}
	}

	// Assert we got a failed status event
	require.NotNil(t, failedEvent, "Should have a TaskStateFailed event")
	assert.True(t, failedEvent.Final, "Failed event should be marked as final")

	// The key assertion: the error message should be included in the status update
	require.NotNil(t, failedEvent.Status.Message, "Failed status should include a message with error details")
	assert.Equal(t, a2a.MessageRoleAgent, failedEvent.Status.Message.Role, "Error message should be from the agent")

	// Extract the error text from the message
	var errorText string

	for _, part := range failedEvent.Status.Message.Parts {
		if textPart, ok := part.(a2a.TextPart); ok {
			errorText += textPart.Text //nolint:perfsprint // Test readability over performance
		}
	}

	t.Logf("Error message in failed status: %s", errorText)

	// The error message should contain details about what went wrong, not just "internal error"
	assert.Contains(t, errorText, "context_length_exceeded", "Error message should contain API error details")
	assert.Contains(t, errorText, "context window", "Error message should contain human-readable error explanation")
}
