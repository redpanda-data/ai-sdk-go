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
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/runner"
)

// Executor implements the a2asrv.AgentExecutor interface, bridging AI SDK agents with A2A protocol.
type Executor struct {
	log     *slog.Logger
	agent   agent.Agent
	runner  *runner.Runner
	pricing *pricing.Catalog
}

// Option configures optional Executor behavior.
type Option func(*Executor)

// WithPricing enables server-side cost reporting. Each model response's usage
// metadata gains a "cost_microcents" entry computed against the given catalog,
// and the final status event carries the invocation's cumulative cost.
//
// Pricing on the server is deliberate: the serving side knows the rate card
// dimensions a UI client cannot see (cache-write vs cache-read rates, service
// tier, inference region, custom per-deployment pricing). Clients should render
// the reported cost rather than re-deriving it from token counts and a public
// price table.
func WithPricing(catalog *pricing.Catalog) Option {
	return func(e *Executor) { e.pricing = catalog }
}

// NewExecutor creates a new A2A executor.
func NewExecutor(
	agent agent.Agent,
	runner *runner.Runner,
	logger *slog.Logger,
	opts ...Option,
) *Executor {
	if logger == nil {
		logger = slog.Default()
	}

	e := &Executor{
		log:    logger,
		agent:  agent,
		runner: runner,
	}
	for _, opt := range opts {
		opt(e)
	}

	return e
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

// usageMetadata converts token usage into the A2A metadata "usage" map.
//
// All counters are disjoint buckets (see llm.TokenUsage): total_tokens is the
// billed sum of every bucket, so consumers can recover the full input side as
// total - output - reasoning. cache_write_tokens aggregates the per-TTL
// cache-creation counters; per-TTL granularity is a pricing concern and stays
// server-side.
func usageMetadata(u *llm.TokenUsage) map[string]any {
	return map[string]any{
		"input_tokens":       u.InputTokens,
		"output_tokens":      u.OutputTokens,
		"total_tokens":       u.TotalBilledTokens(),
		"cached_tokens":      u.CachedInputTokens,
		"reasoning_tokens":   u.ReasoningTokens,
		"cache_write_tokens": u.CacheCreation5mTokens + u.CacheCreation1hTokens + u.CacheCreationUnknownTTLTokens,
		"tool_use_tokens":    u.ToolUseInputTokens,
	}
}

// priceResponse computes the microcent cost of one model call, resolving the
// rate card from the response's own metadata (invoked model, service tier,
// speed, region). Returns false when the model is not in the catalog; the
// response then carries token counts without a cost rather than a guess.
func (e *Executor) priceResponse(ctx context.Context, resp *llm.Response) (int64, bool) {
	modelID := resp.InvokedModelID
	if modelID == "" {
		modelID = e.agent.Info().ModelName
	}

	cost, err := e.pricing.Calculate(modelID, resp.Usage, pricing.CalcRequest{Selector: pricing.SelectorFromResponse(resp)})
	if err != nil {
		e.log.DebugContext(ctx, "Skipping response cost", "model", modelID, "error", err)

		return 0, false
	}

	return cost.Total, true
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

	// Cumulative cost across all model calls in this invocation. Summing
	// per-response costs (rather than pricing the summed usage once at the
	// end) keeps context-bracket rate resolution correct per call. The
	// cumulative value is only reported when every response was priced, so
	// a partially-priced invocation never understates its cost.
	var costMicrocents int64

	costComplete := e.pricing != nil

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
			resp := ev.Response
			llmMsg := llm.NewMessage(llm.RoleUser, &resp)
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
				usage := usageMetadata(ev.Response.Usage)

				if e.pricing != nil {
					if cost, ok := e.priceResponse(ctx, &ev.Response); ok {
						usage["cost_microcents"] = cost
						costMicrocents += cost
					} else {
						costComplete = false
					}
				}

				a2amsg.Metadata = map[string]any{"usage": usage}
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
			if tp, ok := ev.Delta.Part.(*llm.TextPart); ok && tp != nil {
				var artifact *a2a.TaskArtifactUpdateEvent
				if currentArtifactID == "" {
					// Create new artifact for streaming
					artifact = a2a.NewArtifactEvent(reqCtx, a2a.TextPart{Text: tp.Text})
					currentArtifactID = artifact.Artifact.ID
				} else {
					// Append to existing artifact
					artifact = a2a.NewArtifactUpdateEvent(reqCtx, currentArtifactID, a2a.TextPart{Text: tp.Text})
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
				usage := usageMetadata(ev.Usage)

				if e.pricing != nil && costComplete {
					usage["cost_microcents"] = costMicrocents
				}

				metadata["usage"] = usage
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
