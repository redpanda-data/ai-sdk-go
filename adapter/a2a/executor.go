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
	"sync"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/runner"
)

// The two ways a context-window limit surfaces need different advice, so they get
// different messages.
//
// contextOverflowMessage covers the 200-response FinishReasonContextOverflow: the
// conversation itself outgrew the window while generating, and only shortening or
// restarting it helps.
const contextOverflowMessage = "Agent stopped: the conversation exceeds the model's context window. Start a new conversation or shorten the input."

// requestTooLargeMessage covers the pre-generation rejection
// (llm.ErrContextOverflow), which also fires when the input alone fits but
// input plus the reserved response tokens does not — so it names the provider's
// second remedy, lowering the response limit, which keeps the conversation intact.
const requestTooLargeMessage = "Agent stopped: the request does not fit the model's context window. Shorten the input or start a new conversation, or lower the response token limit."

// Executor implements the a2asrv.AgentExecutor interface, bridging AI SDK agents with A2A protocol.
type Executor struct {
	log            *slog.Logger
	agent          agent.Agent
	runner         *runner.Runner
	attributesFunc func(context.Context) map[string]string
	coalesce       DeltaCoalescing

	// activeWriters holds the deltaWriter for every task currently running
	// in processEvents with coalescing on (a2a.TaskID -> *deltaWriter), so
	// Cancel can flush a task's buffered text before it writes the
	// canceled status. Coalescing off never registers here, so Cancel
	// behaves exactly as it always has for every other SDK user.
	//
	// If two processEvents calls somehow run for the same TaskID at once,
	// the later Store overwrites the entry, so Cancel reaches whichever
	// one is currently registered. Each call's own deferred cleanup uses
	// CompareAndDelete, so it only ever removes its own writer, never a
	// newer one that replaced it.
	activeWriters sync.Map
}

// Option configures an Executor.
type Option func(*Executor)

// WithAttributesFunc sets the function that derives a request's
// caller-asserted attributes, such as the end user (agent.AttrUserID) or a
// tenant, which the executor passes to Runner.Run. It is a function because
// one Executor serves every request, so the values can only come from the
// request context, typically set by authenticating middleware. Empty entries
// assert nothing.
func WithAttributesFunc(fn func(context.Context) map[string]string) Option {
	return func(e *Executor) { e.attributesFunc = fn }
}

// WithDeltaCoalescing turns on leading-edge coalescing of streamed text
// deltas, bounded by c.Interval and c.MaxBytes. The zero value keeps the
// executor's default: one artifact event per delta, byte for byte. See
// [DeltaCoalescing] for the send and flush rules.
func WithDeltaCoalescing(c DeltaCoalescing) Option {
	return func(e *Executor) { e.coalesce = c }
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

	for _, o := range opts {
		o(e)
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
	events := e.runner.Run(ctx, reqCtx.ContextID, MessageToLLM(reqCtx.Message),
		runner.WithAttributes(e.attributes(ctx)))
	e.log.InfoContext(ctx, "Runner started, processing events")

	return e.processEvents(ctx, reqCtx, queue, events)
}

// Cancel implements a2asrv.AgentExecutor.
func (e *Executor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	e.log.InfoContext(ctx, "Executor.Cancel called", "task_id", reqCtx.TaskID)

	// Write a canceled status event to the queue
	statusEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, nil)
	statusEvent.Final = true

	// If this instance is running the task's processEvents loop AND Cancel
	// was handed the exact same queue that loop is writing to, go through
	// its deltaWriter so any buffered text is flushed before the canceled
	// status. Otherwise, write the status directly, as before. The queue
	// check matters: a2a-go's distributed (cluster) mode hands Execute and
	// Cancel different queues for the same TaskID (work_queue_handler.go),
	// so writing the canceled status to the Execute-side queue would leave
	// Cancel's own consumer waiting forever and leak the work item.
	writeCanceled := func() error { return queue.Write(ctx, statusEvent) }

	if v, ok := e.activeWriters.Load(reqCtx.TaskID); ok {
		if dw, ok := v.(*deltaWriter); ok && dw.queue == queue {
			writeCanceled = func() error { return dw.cancel(ctx, statusEvent) }
		}
	}

	if err := writeCanceled(); err != nil {
		e.log.ErrorContext(ctx, "Failed to write canceled status", "error", err)

		return err
	}

	e.log.InfoContext(ctx, "Task canceled successfully", "task_id", reqCtx.TaskID)

	return nil
}

func (e *Executor) attributes(ctx context.Context) map[string]string {
	if e.attributesFunc == nil {
		return nil
	}

	return e.attributesFunc(ctx)
}

// processEvents handles the event stream from the runner and writes appropriate A2A events to the queue.
func (e *Executor) processEvents(
	ctx context.Context,
	reqCtx *a2asrv.RequestContext,
	queue eventqueue.Queue,
	events iter.Seq2[agent.Event, error],
) error {
	dw := newDeltaWriter(reqCtx, queue, e.log, e.coalesce)

	// Only register with activeWriters when coalescing can actually buffer
	// something; off, Cancel must behave exactly as it always has, with no
	// writer to find.
	if dw.cfg.Interval > 0 {
		e.activeWriters.Store(reqCtx.TaskID, dw)
		defer e.activeWriters.CompareAndDelete(reqCtx.TaskID, dw)
	}

	defer dw.close()

	// write logs a failed queue write the same way the executor's write
	// closure always has. Every non-delta write in this loop goes through
	// it, except the context-canceled branch below, which needs its own,
	// more specific log message.
	write := func(ev a2a.Event) {
		if err := dw.write(ctx, ev); err != nil {
			e.log.ErrorContext(ctx, "Failed to write to queue", "error", err)
		}
	}

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
				if writeErr := dw.write(bgCtx, statusEvent); writeErr != nil {
					e.log.ErrorContext(ctx, "Failed to write canceled status", "error", writeErr)
				}
			} else if errors.Is(err, llm.ErrContextOverflow) {
				// The provider rejected the request before generating because it does not
				// fit the context window. Terminal, but give the user the truthful,
				// actionable message rather than the raw provider error.
				errMsg := a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: requestTooLargeMessage})
				statusEvent := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateFailed, errMsg)
				statusEvent.Final = true
				write(statusEvent)
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

		if _, isDelta := event.(agent.AssistantDeltaEvent); !isDelta {
			e.log.DebugContext(ctx, "Processing event", "type", fmt.Sprintf("%T", event))
		}

		switch ev := event.(type) {
		case agent.StatusEvent:
			e.log.DebugContext(ctx, "Status event", "stage", ev.Stage)
			// When we receive a "model_call" status, it marks the start of a new LLM response.
			// Flush anything still buffered for the old artifact, then forget its ID so the
			// next delta/message creates a distinct artifact.
			if ev.Stage == agent.StatusStageModelCall {
				dw.resetArtifact(ctx)
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
			dw.endArtifact(ctx)

			// Add agent's message to history via a status update
			// Convert LLM response to A2A message format
			a2amsg := MessageFromLLM(ev.Response.Message)

			// Attach token usage to the message itself if available
			if ev.Response.Usage != nil {
				a2amsg.Metadata = map[string]any{
					"usage": map[string]any{
						"input_tokens":     ev.Response.Usage.InputTokens,
						"output_tokens":    ev.Response.Usage.OutputTokens,
						"total_tokens":     ev.Response.Usage.TotalBilledTokens(),
						"cached_tokens":    ev.Response.Usage.CachedInputTokens,
						"reasoning_tokens": ev.Response.Usage.ReasoningTokens,
					},
				}
			}

			historyStatus := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, a2amsg)
			write(historyStatus)
		case agent.StreamResetEvent:
			// Stream is being retried — abandon current streaming artifact
			dw.endArtifact(ctx)
		case agent.AssistantDeltaEvent:
			// Stream delta updates as incremental artifact chunks
			if tp, ok := ev.Delta.Part.(*llm.TextPart); ok && tp != nil {
				dw.delta(ctx, tp.Text)
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
				// Output truncation is non-fatal: the model produced a partial
				// response and stopped at the output-token cap. Complete the task,
				// deliver what we have (already streamed to history), and mark the
				// turn truncated so the surface can offer a Continue action instead
				// of a destructive failure card.
				taskState = a2a.TaskStateCompleted
				statusMsg = a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{
					Text: "Response was truncated — the maximum output token limit was reached. Continue to get the rest.",
				})
				// The `truncated` marker is the load-bearing contract: it lives on
				// the message metadata (where consumers read usage), NOT the event
				// metadata. This notice is a UI affordance, not model output — a
				// consumer that reconstructs conversation history from agent status
				// messages MUST skip messages carrying `truncated: true`, otherwise a
				// "Continue" would resume from a history where the assistant appears
				// to have said "Continue to get the rest." The partial answer was
				// already delivered as a separate Working-status MessageEvent.
				statusMsg.Metadata = map[string]any{"truncated": true}
			case agent.FinishReasonContextOverflow:
				// Genuinely too long: the input exceeded the model's context window,
				// so nothing could be generated. Terminal, but say so truthfully.
				taskState = a2a.TaskStateFailed
				statusMsg = a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{
					Text: contextOverflowMessage,
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
					"total_tokens":     ev.Usage.TotalBilledTokens(),
					"cached_tokens":    ev.Usage.CachedInputTokens,
					"reasoning_tokens": ev.Usage.ReasoningTokens,
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
