package a2a

import (
	"context"
	"log/slog"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

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
			}

		case *a2a.TaskArtifactUpdateEvent:
			hasArtifact = true

			t.Logf("  Artifact ID: %s, Append: %v, LastChunk: %v", ev.Artifact.ID, ev.Append, ev.LastChunk)

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
