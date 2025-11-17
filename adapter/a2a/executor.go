package a2a

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"

	"github.com/redpanda-data/ai-sdk-go/agent"
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
		e.log.InfoContext(ctx, "Wrote submitted status")
	}

	// Emit working status before starting runner
	workingEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, nil)
	write(workingEvent)
	e.log.InfoContext(ctx, "Wrote working status")

	// Run the agent and process events
	events := e.runner.Run(ctx, "", reqCtx.ContextID, MessageToLLM(reqCtx.Message))
	e.log.InfoContext(ctx, "Runner started, processing events")

	return e.processEvents(ctx, reqCtx, queue, events)
}

// Cancel implements a2asrv.AgentExecutor.
func (e *Executor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, _ eventqueue.Queue) error {
	e.log.InfoContext(ctx, "Executor.Cancel called", "task_id", reqCtx.TaskID)

	// TODO: Implement cancellation logic here

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
			// Write a failed status event and return
			statusEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateFailed, nil)
			statusEvent.Final = true
			write(statusEvent)

			return err
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
			// Already handled by generic MessageEvent
		case agent.ToolResponseEvent:
			// Already handled by generic MessageEvent
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
			historyStatus := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, a2amsg)

			// Attach token usage if available
			if ev.Response.Usage != nil {
				historyStatus.Metadata = map[string]any{
					"usage": map[string]any{
						"input_tokens":     ev.Response.Usage.InputTokens,
						"output_tokens":    ev.Response.Usage.OutputTokens,
						"total_tokens":     ev.Response.Usage.TotalTokens,
						"cached_tokens":    ev.Response.Usage.CachedTokens,
						"reasoning_tokens": ev.Response.Usage.ReasoningTokens,
					},
				}
			}
			write(historyStatus)

			// Reset artifactID so next model_call creates a new one
			currentArtifactID = ""
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
			e.log.InfoContext(ctx, "Invocation end event", "finish_reason", ev.FinishReason)
			// Write final completion status - this is required to signal task completion
			statusEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCompleted, nil)
			statusEvent.Final = true

			// Add token usage to metadata if available
			if ev.Usage != nil {
				statusEvent.Metadata = map[string]any{
					"usage": map[string]any{
						"input_tokens":     ev.Usage.InputTokens,
						"output_tokens":    ev.Usage.OutputTokens,
						"total_tokens":     ev.Usage.TotalTokens,
						"cached_tokens":    ev.Usage.CachedTokens,
						"reasoning_tokens": ev.Usage.ReasoningTokens,
					},
				}
			}
			write(statusEvent)
			e.log.InfoContext(ctx, "Returning from InvocationEndEvent")
			return nil
		default:
			e.log.DebugContext(ctx, "Received unhandled event", "type", fmt.Sprintf("%T", event))
		}
	}

	// If we exit the loop without receiving InvocationEndEvent, write a completion status anyway
	e.log.WarnContext(ctx, "Event loop ended without InvocationEndEvent")

	statusEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCompleted, nil)
	statusEvent.Final = true
	write(statusEvent)

	return errors.New("incomplete agent call: missing InvocationEndEvent")
}
