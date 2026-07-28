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

package a2a

import (
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// runExecutorOnce drives the executor for a single user message against the
// given fake model and returns every A2A event written to the queue.
func runExecutorOnce(t *testing.T, model *fakellm.FakeModel, userText string) []a2a.Event {
	t.Helper()

	ag, err := llmagent.New("test-agent", "You are a helpful assistant.", model)
	require.NoError(t, err)

	runnerInstance, err := runner.New(ag, session.NewInMemoryStore())
	require.NoError(t, err)

	executor := NewExecutor(ag, runnerInstance, slog.Default())

	reqCtx := &a2asrv.RequestContext{
		ContextID: "test-context",
		TaskID:    "test-task",
		Message:   a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: userText}),
	}

	ctx := context.Background()
	queueMgr := eventqueue.NewInMemoryManager(eventqueue.WithQueueBufferSize(100))
	readerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)
	writerQueue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)

	var events []a2a.Event

	eventsDone := make(chan struct{})
	finalEventSeen := make(chan struct{})

	go func() {
		defer close(eventsDone)

		for {
			event, _, readErr := readerQueue.Read(ctx)
			if readErr != nil {
				return
			}

			events = append(events, event)

			if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Final {
				close(finalEventSeen)
			}
		}
	}()

	require.NoError(t, executor.Execute(ctx, reqCtx, writerQueue))

	<-finalEventSeen
	writerQueue.Close()
	readerQueue.Close()
	<-eventsDone

	return events
}

func finalStatusEvent(t *testing.T, events []a2a.Event) *a2a.TaskStatusUpdateEvent {
	t.Helper()

	for _, event := range events {
		if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Final {
			return statusEvent
		}
	}

	require.FailNow(t, "no final status event found")

	return nil
}

func statusText(status *a2a.TaskStatusUpdateEvent) string {
	if status.Status.Message == nil {
		return ""
	}

	var text strings.Builder

	for _, part := range status.Status.Message.Parts {
		if textPart, ok := part.(a2a.TextPart); ok {
			text.WriteString(textPart.Text)
		}
	}

	return text.String()
}

// TestExecutor_OutputTruncationIsNonFatal is the regression guard: an output
// truncation (FinishReasonLength) must NOT fail the run. It completes the task,
// delivers the partial content, and marks the turn truncated so the surface can
// offer a Continue action instead of a destructive error card.
func TestExecutor_OutputTruncationIsNonFatal(t *testing.T) {
	t.Parallel()

	const partial = "Here is the first half of the answer"

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(partial)),
				FinishReason: llm.FinishReasonLength,
				Usage:        &llm.TokenUsage{InputTokens: 10, OutputTokens: 16},
			}, nil
		})

	events := runExecutorOnce(t, model, "Write a long essay.")

	final := finalStatusEvent(t, events)
	assert.Equal(t, a2a.TaskStateCompleted, final.Status.State,
		"output truncation must complete the task, not fail it")

	require.NotNil(t, final.Status.Message, "truncated completion should carry a notice message")
	require.NotNil(t, final.Status.Message.Metadata, "truncation marker must live on the message metadata")
	assert.Equal(t, true, final.Status.Message.Metadata["truncated"],
		"consumer reads the truncated marker off the message metadata")

	// A truncation must not carry the misleading context-length message.
	assert.NotContains(t, statusText(final), "context length limit exceeded")

	// The partial content must still have been delivered to history.
	var delivered bool

	for _, event := range events {
		if statusEvent, ok := event.(*a2a.TaskStatusUpdateEvent); ok && statusEvent.Status.Message != nil {
			if strings.Contains(statusText(statusEvent), partial) {
				delivered = true
			}
		}
	}

	assert.True(t, delivered, "partial assistant content must be delivered")
}

// TestExecutor_ContextOverflowStaysFatalButTruthful verifies the other half of
// the split: a genuine context-window overflow remains a terminal failure, but
// with a truthful, actionable message rather than a misleading "context length
// limit exceeded".
func TestExecutor_ContextOverflowStaysFatalButTruthful(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			return &llm.Response{
				Message:      llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{}},
				FinishReason: llm.FinishReasonContextOverflow,
			}, nil
		})

	events := runExecutorOnce(t, model, "A very long conversation.")

	final := finalStatusEvent(t, events)
	assert.Equal(t, a2a.TaskStateFailed, final.Status.State,
		"context overflow is terminal")

	text := statusText(final)
	assert.Contains(t, strings.ToLower(text), "context window",
		"overflow message must name the real cause")
	assert.NotContains(t, text, "context length limit exceeded",
		"must not reuse the misleading legacy message")
}
