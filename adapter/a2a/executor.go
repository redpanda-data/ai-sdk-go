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
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/runner"
)

// Executor implements the a2asrv.AgentExecutor interface, bridging AI SDK agents with A2A protocol.
type Executor struct {
	log    *slog.Logger
	agent  agent.Agent
	runner *runner.Runner
}

// NewExecutor creates a new A2A executor.
func NewExecutor(
	agent agent.Agent,
	runner *runner.Runner,
	logger *slog.Logger,
) *Executor {
	if logger == nil {
		logger = slog.Default()
	}

	return &Executor{
		log:    logger,
		agent:  agent,
		runner: runner,
	}
}

// Execute implements a2asrv.AgentExecutor.
// This is called for each message/send or message/stream request.
func (e *Executor) Execute(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	e.log.InfoContext(ctx, "Executor.Execute called",
		"task_id", reqCtx.TaskID,
		"context_id", reqCtx.ContextID,
		"has_stored_task", reqCtx.StoredTask != nil,
		"related_tasks_count", len(reqCtx.RelatedTasks),
		"has_message", reqCtx.Message != nil,
	)
	// Helper closure to write events to queue with error logging
	write := func(event a2a.Event) {
		if err := queue.Write(ctx, event); err != nil {
			e.log.ErrorContext(ctx, "Failed to write to queue", "error", err)
		}
	}

	// Create new task if necessary. Otherwise, StoredTask will provide it.
	if reqCtx.StoredTask == nil {
		event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateSubmitted, nil)
		write(event)
	}

	// Emit working status before starting runner
	workingEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, nil)
	write(workingEvent)

	// Run the agent and process events
	events := e.runner.Run(ctx, "", reqCtx.ContextID, MessageToLLM(reqCtx.Message))
	e.log.InfoContext(ctx, "Runner started, processing events")

	return e.processEvents(ctx, reqCtx, queue, events)
}

// Cancel implements a2asrv.AgentExecutor.
func (e *Executor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	e.log.InfoContext(ctx, "Executor.Cancel called", "task_id", reqCtx.TaskID)

	// Write a canceled status event to the queue
	statusEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, nil)
	statusEvent.Final = true

	if err := queue.Write(ctx, statusEvent); err != nil {
		e.log.ErrorContext(ctx, "Failed to write canceled status", "error", err)

		return err
	}

	e.log.InfoContext(ctx, "Task canceled successfully", "task_id", reqCtx.TaskID)

	return nil
}

// processEvents handles the event stream from the runner and writes appropriate A2A events to the queue.
func (e *Executor) processEvents(
	ctx context.Context,
	reqCtx *a2asrv.RequestContext,
	queue eventqueue.Queue,
	events iter.Seq2[agent.Event, error],
) error {
	write := func(event a2a.Event) {
		if err := queue.Write(ctx, event); err != nil {
			e.log.ErrorContext(ctx, "Failed to write to queue", "error", err)
		}
	}

	// Rolling current artifact ID for streaming text deltas
	var currentArtifactID a2a.ArtifactID

	for event, err := range events {
		if err != nil {
			e.log.ErrorContext(ctx, "Runner returned error", "error", err)

			// Check if the error is a cancellation error
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				// Emit canceled status with error message
				// Use background context with timeout since the original context is likely canceled
				bgCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer cancel()

				errMsg := a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: err.Error()})
				statusEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, errMsg)
				statusEvent.Final = true

				//nolint:contextcheck // Must use background context since original context is canceled
				if writeErr := queue.Write(bgCtx, statusEvent); writeErr != nil {
					e.log.ErrorContext(ctx, "Failed to write canceled status", "error", writeErr)
				}
			} else {
				// Regular failure - emit failed status with error message
				errMsg := a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: err.Error()})
				statusEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateFailed, errMsg)
				statusEvent.Final = true
				write(statusEvent)
			}

			// Agent failures are communicated via task status events, not Execute errors.
			// Only return errors for queue write failures (per AgentExecutor interface contract).
			return nil
		}

		e.log.DebugContext(ctx, "Processing event", "type", fmt.Sprintf("%T", event))

		switch ev := event.(type) {
		case agent.StatusEvent:
			e.log.DebugContext(ctx, "Status event", "stage", ev.Stage)
			// When we receive a "model_call" status, it marks the start of a new LLM response
			// Reset artifact ID so next delta/message creates a distinct artifact
			if ev.Stage == agent.StatusStageModelCall {
				currentArtifactID = ""
			}
		case agent.ToolRequestEvent:
			// Tool request is already in MessageEvent, no separate handling needed
		case agent.ToolResponseEvent:
			e.log.DebugContext(ctx, "Tool response event", "tool", ev.Response.Name)

			// Add tool response to history as a user message
			llmMsg := llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(&ev.Response))
			a2amsg := MessageFromLLM(llmMsg)
			historyStatus := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, a2amsg)
			write(historyStatus)
		case agent.MessageEvent:
			// Mark the streaming artifact as complete if we were streaming
			if currentArtifactID != "" {
				finalArtifact := a2a.NewArtifactUpdateEvent(reqCtx, currentArtifactID)
				finalArtifact.LastChunk = true
				write(finalArtifact)
			}

			// Add agent's message to history via a status update
			// Convert LLM response to A2A message format
			a2amsg := MessageFromLLM(ev.Response.Message)

			// Attach token usage to the message itself if available
			if ev.Response.Usage != nil {
				a2amsg.Metadata = map[string]any{
					"usage": map[string]any{
						"input_tokens":     ev.Response.Usage.InputTokens,
						"output_tokens":    ev.Response.Usage.OutputTokens,
						"total_tokens":     ev.Response.Usage.TotalTokens,
						"cached_tokens":    ev.Response.Usage.CachedTokens,
						"reasoning_tokens": ev.Response.Usage.ReasoningTokens,
						"max_input_tokens": ev.Response.Usage.MaxInputTokens,
					},
				}
			}

			historyStatus := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, a2amsg)
			write(historyStatus)
			// Reset artifactID so next model_call creates a new one
			currentArtifactID = ""
		case agent.StreamResetEvent:
			// Stream is being retried — abandon current streaming artifact
			if currentArtifactID != "" {
				finalArtifact := a2a.NewArtifactUpdateEvent(reqCtx, currentArtifactID)
				finalArtifact.LastChunk = true
				write(finalArtifact)

				currentArtifactID = ""
			}
		case agent.AssistantDeltaEvent:
			// Stream delta updates as incremental artifact chunks
			if ev.Delta.Part != nil && ev.Delta.Part.IsText() {
				var artifact *a2a.TaskArtifactUpdateEvent
				if currentArtifactID == "" {
					// Create new artifact for streaming
					artifact = a2a.NewArtifactEvent(reqCtx, a2a.TextPart{Text: ev.Delta.Part.Text})
					currentArtifactID = artifact.Artifact.ID
				} else {
					// Append to existing artifact
					artifact = a2a.NewArtifactUpdateEvent(reqCtx, currentArtifactID, a2a.TextPart{Text: ev.Delta.Part.Text})
					artifact.Append = true
				}

				write(artifact)
			}
		case agent.InvocationEndEvent:
			e.log.DebugContext(ctx, "Invocation end event", "finish_reason", ev.FinishReason)

			// Map finish reason to appropriate A2A task state
			var taskState a2a.TaskState
			var statusMsg *a2a.Message

			switch ev.FinishReason {
			case agent.FinishReasonStop, agent.FinishReasonTransfer:
				taskState = a2a.TaskStateCompleted
			case agent.FinishReasonMaxTurns:
				taskState = a2a.TaskStateFailed
				statusMsg = a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{
					Text: "Agent stopped: maximum iterations reached",
				})
			case agent.FinishReasonLength:
				taskState = a2a.TaskStateFailed
				statusMsg = a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{
					Text: "Agent stopped: context length limit exceeded",
				})
			case agent.FinishReasonError:
				taskState = a2a.TaskStateFailed
			case agent.FinishReasonInterrupted:
				taskState = a2a.TaskStateCanceled
			case agent.FinishReasonInputRequired:
				taskState = a2a.TaskStateInputRequired
			default:
				e.log.ErrorContext(ctx, "Unknown finish reason", "finish_reason", ev.FinishReason)

				taskState = a2a.TaskStateFailed
				statusMsg = a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{
					Text: fmt.Sprintf("Agent stopped: unknown finish reason %q", ev.FinishReason),
				})
			}

			statusEvent := a2a.NewStatusUpdateEvent(reqCtx, taskState, statusMsg)
			statusEvent.Final = true

			// Add token usage and finish reason to metadata
			metadata := map[string]any{
				"finish_reason": string(ev.FinishReason),
			}

			if ev.Usage != nil {
				metadata["usage"] = map[string]any{
					"input_tokens":     ev.Usage.InputTokens,
					"output_tokens":    ev.Usage.OutputTokens,
					"total_tokens":     ev.Usage.TotalTokens,
					"cached_tokens":    ev.Usage.CachedTokens,
					"reasoning_tokens": ev.Usage.ReasoningTokens,
					"max_input_tokens": ev.Usage.MaxInputTokens,
				}
			}

			statusEvent.Metadata = metadata

			write(statusEvent)

			return nil
		default:
			e.log.DebugContext(ctx, "Received unhandled event", "type", fmt.Sprintf("%T", event))
		}
	}

	// If we exit the loop without receiving InvocationEndEvent, write a completion status anyway
	e.log.WarnContext(ctx, "Event loop ended without InvocationEndEvent")

	statusEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateFailed, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "internal error: incomplete agent call: missing InvocationEndEvent"}))
	statusEvent.Final = true
	write(statusEvent)

	return errors.New("incomplete agent call: missing InvocationEndEvent")
}
