// Package llmagent provides an LLM-based agent implementation with tool calling support.
package llmagent

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Compile-time check that LLMAgent implements agent.Agent.
var _ agent.Agent = (*LLMAgent)(nil)

// LLMAgent is an agent implementation that uses an LLM for execution.
//
// It implements the agent.Agent interface and executes a turn loop:
//   - Generate response from LLM
//   - Execute any requested tools
//   - Add results to conversation
//   - Repeat until completion
//
// Events are yielded during execution to provide real-time progress updates.
type LLMAgent struct {
	config *config
}

// New creates a new LLM agent with the given name, system prompt, and model.
//
// All three parameters are required. The system prompt defines the agent's
// behavior and purpose. Optional configuration can be provided via Option functions.
//
// # Example
//
//	agent, err := llmagent.New(
//	    "assistant",
//	    "You are a helpful assistant.",
//	    openaiModel,
//	    llmagent.WithTools(toolRegistry),
//	    llmagent.WithMaxTurns(10),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
func New(name string, systemPrompt string, model llm.Model, opts ...Option) (*LLMAgent, error) {
	cfg := &config{
		name:            name,
		systemPrompt:    systemPrompt,
		model:           model,
		maxTurns:        25, // default
		toolConcurrency: 3,  // default
	}

	// Apply options
	for _, opt := range opts {
		opt(cfg)
	}

	// Validate configuration
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return &LLMAgent{
		config: cfg,
	}, nil
}

// Name returns the agent's identifier.
func (a *LLMAgent) Name() string {
	return a.config.name
}

// Description returns the agent's purpose and capabilities.
func (a *LLMAgent) Description() string {
	return a.config.description
}

// InputSchema returns the expected input schema.
//
// For now, this returns a simple text message schema. Future versions
// may support structured inputs.
func (a *LLMAgent) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"message": map[string]any{
				"type":        "string",
				"description": "The message to send to the agent",
			},
		},
		"required": []string{"message"},
	}
}

// Run executes the LLM agent, yielding events during execution.
//
// The agent executes a turn loop, yielding events for:
//   - Status transitions (turn started, model call, tool execution)
//   - Assistant messages
//   - Tool calls and results
//   - Completion (InvocationEndEvent)
//
// The stream always ends with InvocationEndEvent, even on error or cancellation.
func (a *LLMAgent) Run(invCtx *agent.InvocationContext) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		sess := invCtx.Session()

		// Helper: create event envelope
		makeEnvelope := func() agent.EventEnvelope {
			return agent.EventEnvelope{
				InvocationID: invCtx.InvocationID(),
				SessionID:    invCtx.Session().ID,
				Turn:         invCtx.Turn(),
				At:           time.Now().UTC(),
			}
		}

		// Helper: yield event (returns false if consumer stopped)
		yieldEvent := func(evt agent.Event) bool {
			return yield(evt, nil)
		}

		// Ensure system prompt is present
		messages := a.ensureSystemPrompt(sess.Messages)

		// Execute turn loop
		for invCtx.Turn() < a.config.maxTurns {
			// Emit turn started
			if !yieldEvent(agent.StatusEvent{
				Envelope: makeEnvelope(),
				Stage:    agent.StatusStageTurnStarted,
				Details:  fmt.Sprintf("turn %d started", invCtx.Turn()),
			}) {
				return
			}

			// Check context cancellation
			if invCtx.Err() != nil {
				yieldEvent(agent.InvocationEndEvent{
					Envelope:     makeEnvelope(),
					FinishReason: agent.FinishReasonInterrupted,
					Usage:        ptr(invCtx.TotalUsage()),
				})

				return
			}

			// Emit model call status
			if !yieldEvent(agent.StatusEvent{
				Envelope: makeEnvelope(),
				Stage:    agent.StatusStageModelCall,
				Details:  "invoking model",
			}) {
				return
			}

			// Generate response from LLM (with streaming support if available)
			resp, err := a.generate(invCtx, messages, makeEnvelope, yieldEvent)
			if err != nil {
				// Determine finish reason based on error type
				reason := mapErrorToFinishReason(err)

				// Emit error event
				yieldEvent(agent.ErrorEvent{
					Envelope: makeEnvelope(),
					Err:      err,
					Message:  err.Error(), // Error is already wrapped with ErrModelGeneration
				})
				// Emit terminal event
				yieldEvent(agent.InvocationEndEvent{
					Envelope:     makeEnvelope(),
					FinishReason: reason,
					Usage:        ptr(invCtx.TotalUsage()),
				})

				return
			}

			// Update usage tracking
			invCtx.AddUsage(resp.Usage)

			// Add assistant message to session and local history
			sess.Messages = append(sess.Messages, resp.Message)
			messages = append(messages, resp.Message)

			// Emit message event
			if !yieldEvent(agent.MessageEvent{
				Envelope: makeEnvelope(),
				Response: *resp,
			}) {
				return
			}

			// Check for terminal finish reasons from the model
			agentReason, terminalErr := mapLLMFinishReason(resp.FinishReason)
			if agentReason != "" {
				// Terminal finish reason - handle completion
				if terminalErr != nil {
					// Emit error for terminal error conditions (content filter, interrupted, unknown)
					yieldEvent(agent.ErrorEvent{
						Envelope: makeEnvelope(),
						Err:      terminalErr,
						Message:  terminalErr.Error(),
					})
				} else if agentReason == agent.FinishReasonLength {
					// Emit status event for length limit (non-error terminal case)
					yieldEvent(agent.StatusEvent{
						Envelope: makeEnvelope(),
						Stage:    agent.StatusStageTurnCompleted,
						Details:  fmt.Sprintf("turn %d completed - length limit", invCtx.Turn()),
						Usage:    resp.Usage,
					})
				}

				// Emit terminal event
				yieldEvent(agent.InvocationEndEvent{
					Envelope:     makeEnvelope(),
					FinishReason: agentReason,
					Usage:        ptr(invCtx.TotalUsage()),
				})

				return
			}
			// Non-terminal finish reason (ToolCalls or Stop) - continue below

			// Check for tool calls
			toolReqs := resp.ToolRequests()
			if len(toolReqs) == 0 {
				// No tools requested - natural completion
				// Emit turn completed
				yieldEvent(agent.StatusEvent{
					Envelope: makeEnvelope(),
					Stage:    agent.StatusStageTurnCompleted,
					Details:  fmt.Sprintf("turn %d completed", invCtx.Turn()),
					Usage:    resp.Usage,
				})
				// Emit terminal event
				yieldEvent(agent.InvocationEndEvent{
					Envelope:     makeEnvelope(),
					FinishReason: agent.FinishReasonStop,
					Usage:        ptr(invCtx.TotalUsage()),
				})

				return
			}

			// Emit tool call events
			for _, toolReq := range toolReqs {
				if !yieldEvent(agent.ToolCallEvent{
					Envelope: makeEnvelope(),
					Request:  *toolReq,
				}) {
					return
				}
			}

			// Emit tool execution status
			if !yieldEvent(agent.StatusEvent{
				Envelope: makeEnvelope(),
				Stage:    agent.StatusStageToolExec,
				Details:  fmt.Sprintf("executing %d tools", len(toolReqs)),
			}) {
				return
			}

			// Execute tools and collect results
			toolMessages, err := a.executeTools(invCtx, toolReqs, makeEnvelope, yieldEvent)
			if err != nil {
				// Determine finish reason based on error type
				reason := mapErrorToFinishReason(err)

				// Emit error event
				yieldEvent(agent.ErrorEvent{
					Envelope: makeEnvelope(),
					Err:      err,
					Message:  err.Error(), // Error is already wrapped with ErrToolExecution or ErrToolRegistry
				})
				// Emit terminal event
				yieldEvent(agent.InvocationEndEvent{
					Envelope:     makeEnvelope(),
					FinishReason: reason,
					Usage:        ptr(invCtx.TotalUsage()),
				})

				return
			}

			// Add tool results to conversation
			for _, msg := range toolMessages {
				sess.Messages = append(sess.Messages, msg)
				messages = append(messages, msg)
			}

			// Emit turn completed
			if !yieldEvent(agent.StatusEvent{
				Envelope: makeEnvelope(),
				Stage:    agent.StatusStageTurnCompleted,
				Details:  fmt.Sprintf("turn %d completed", invCtx.Turn()),
			}) {
				return
			}

			// Increment turn for next iteration
			invCtx.IncrementTurn()
		}

		// Max turns reached
		yieldEvent(agent.InvocationEndEvent{
			Envelope:     makeEnvelope(),
			FinishReason: agent.FinishReasonMaxTurns,
			Usage:        ptr(invCtx.TotalUsage()),
		})
	}
}

// ensureSystemPrompt adds the system prompt if not already present.
//
// The system prompt is never persisted - it's only added at runtime.
// This prevents duplication across invocations.
func (a *LLMAgent) ensureSystemPrompt(messages []llm.Message) []llm.Message {
	if len(messages) > 0 && messages[0].Role == llm.RoleSystem {
		// System prompt already present
		return messages
	}

	// Prepend system prompt
	systemMsg := llm.NewMessage(llm.RoleSystem, llm.NewTextPart(a.config.systemPrompt))

	return append([]llm.Message{systemMsg}, messages...)
}

// generate calls the LLM to generate a response.
//
// Tool definitions are included if a tool registry is configured.
// If the model supports streaming (implements llm.EventsGenerator),
// it will emit AssistantDeltaEvent for each content part as it arrives.
func (a *LLMAgent) generate(
	invCtx *agent.InvocationContext,
	messages []llm.Message,
	makeEnvelope func() agent.EventEnvelope,
	yieldEvent func(agent.Event) bool,
) (*llm.Response, error) {
	req := &llm.Request{
		Messages: messages,
	}

	if a.config.tools != nil {
		req.Tools = a.config.tools.List()
	}

	// Use streaming if model supports it (provides better UX with real-time updates)
	if eg, ok := a.config.model.(llm.EventsGenerator); ok {
		return a.generateWithStreaming(invCtx, eg, req, makeEnvelope, yieldEvent)
	}

	// Fall back to non-streaming generation
	return a.config.model.Generate(invCtx, req)
}

// generateWithStreaming uses the EventsGenerator interface to get token-by-token deltas.
// Each delta is emitted as an AssistantDeltaEvent for real-time streaming feedback.
func (a *LLMAgent) generateWithStreaming(
	ctx context.Context,
	eg llm.EventsGenerator,
	req *llm.Request,
	makeEnvelope func() agent.EventEnvelope,
	yieldEvent func(agent.Event) bool,
) (*llm.Response, error) {
	var response *llm.Response

	for event, err := range eg.GenerateEvents(ctx, req) {
		if err != nil {
			return nil, fmt.Errorf("%w: %w", agent.ErrModelGeneration, err)
		}

		switch evt := event.(type) {
		case llm.ContentPartEvent:
			// Emit real-time delta for streaming consumers
			if !yieldEvent(agent.AssistantDeltaEvent{
				Envelope: makeEnvelope(),
				Delta:    evt,
			}) {
				return nil, errors.New("consumer stopped iteration")
			}

		case llm.StreamEndEvent:
			// StreamEndEvent always has exactly one of Response or Error set
			if evt.Error != nil {
				return nil, fmt.Errorf("%w: %w", agent.ErrModelGeneration, evt.Error)
			}

			response = evt.Response

		case llm.ErrorEvent:
			// ErrorEvent is NON-TERMINAL - emit it and continue processing.
			// The LLM SDK may emit recoverable errors (rate limits, warnings, etc.)
			// that should be passed through to callers without terminating the stream.
			// The stream ends naturally with StreamEndEvent or a transport error from the iterator.
			if !yieldEvent(agent.ErrorEvent{
				Envelope: makeEnvelope(),
				Err:      fmt.Errorf("%w: %s", agent.ErrModelGeneration, evt.Message),
				Message:  evt.Message,
			}) {
				return nil, errors.New("consumer stopped iteration")
			}

			// Continue processing - stream may recover or end naturally
			continue
		}
	}

	// Defensive check: provider should always emit StreamEndEvent, but guard against violations
	if response == nil {
		return nil, fmt.Errorf("%w: stream ended without response", agent.ErrModelGeneration)
	}

	return response, nil
}

// executeTools runs tool calls concurrently.
//
// Tool execution is limited by toolConcurrency. Individual tool errors
// are captured and returned as error messages, allowing the LLM to
// handle failures gracefully.
//
// ToolResultEvents are yielded as tools complete.
//
// Returns tool result messages to be added to the conversation.
//
// Future: Will also return list of tool IDs requiring input for
// StatusStageInputRequired / FinishReasonInputRequired support.
func (a *LLMAgent) executeTools(
	invCtx *agent.InvocationContext,
	toolReqs []*llm.ToolRequest,
	makeEnvelope func() agent.EventEnvelope,
	yieldEvent func(agent.Event) bool,
) ([]llm.Message, error) {
	if a.config.tools == nil {
		return nil, agent.ErrToolRegistry
	}

	// Execute tools concurrently with limited parallelism
	g, ctx := errgroup.WithContext(invCtx)
	g.SetLimit(min(a.config.toolConcurrency, len(toolReqs)))

	// Results channel (buffered to avoid blocking)
	type toolResult struct {
		idx       int
		requestID string
		name      string
		response  *llm.ToolResponse
		err       error
	}

	results := make(chan toolResult, len(toolReqs))

	// Launch tool executions
	for i, req := range toolReqs {
		g.Go(func() error {
			resp, err := a.config.tools.Execute(ctx, req)
			results <- toolResult{
				idx:       i,
				requestID: req.ID,
				name:      req.Name,
				response:  resp,
				err:       err,
			}

			return nil // Never return error to errgroup (we handle errors individually)
		})
	}

	// Wait for all tools to complete
	if err := g.Wait(); err != nil {
		close(results)
		return nil, fmt.Errorf("%w: %w", agent.ErrToolExecution, err)
	}

	close(results)

	// Collect results in original request order for deterministic transcripts
	ordered := make([]toolResult, len(toolReqs))
	for result := range results {
		ordered[result.idx] = result
	}

	// Build messages and yield events in request order
	messages := make([]llm.Message, 0, len(toolReqs))

	for _, result := range ordered {
		var msg llm.Message

		if result.err != nil {
			// Tool execution failed - create error response
			errResp := &llm.ToolResponse{
				ID:    result.requestID,
				Name:  result.name,
				Error: result.err.Error(),
			}
			msg = llm.NewMessage(llm.RoleTool, llm.NewToolResponsePart(errResp))

			// Yield error tool result event
			if !yieldEvent(agent.ToolResultEvent{
				Envelope: makeEnvelope(),
				Response: *errResp,
			}) {
				return messages, nil // Consumer stopped listening
			}
		} else {
			// Tool execution succeeded
			msg = llm.NewMessage(llm.RoleTool, llm.NewToolResponsePart(result.response))

			// Yield tool result event
			if !yieldEvent(agent.ToolResultEvent{
				Envelope: makeEnvelope(),
				Response: *result.response,
			}) {
				return messages, nil // Consumer stopped listening
			}
		}

		messages = append(messages, msg)
	}

	return messages, nil
}

// mapLLMFinishReason converts an llm.FinishReason to an agent.FinishReason.
// Returns the mapped finish reason and any error that should be emitted for
// terminal error conditions (content filter, interrupted, unknown).
//
// Returns ("", nil) for non-terminal reasons like ToolCalls that should
// continue execution.
func mapLLMFinishReason(reason llm.FinishReason) (agent.FinishReason, error) {
	switch reason {
	case llm.FinishReasonStop:
		return agent.FinishReasonStop, nil

	case llm.FinishReasonLength:
		return agent.FinishReasonLength, nil

	case llm.FinishReasonToolCalls:
		// Not terminal - caller should continue to tool execution
		return "", nil

	case llm.FinishReasonContentFilter:
		return agent.FinishReasonError, llm.ErrContentPolicyViolation

	case llm.FinishReasonInterrupted:
		return agent.FinishReasonInterrupted, context.Canceled

	case llm.FinishReasonUnknown:
		return agent.FinishReasonError, errors.New("model returned unknown finish reason")

	default:
		return agent.FinishReasonError, fmt.Errorf("unhandled finish reason: %v", reason)
	}
}

// mapErrorToFinishReason determines the appropriate finish reason for an error.
// Returns FinishReasonCanceled for context cancellation/deadline, otherwise
// returns FinishReasonError.
func mapErrorToFinishReason(err error) agent.FinishReason {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return agent.FinishReasonInterrupted
	}

	return agent.FinishReasonError
}

// ptr is a helper to get a pointer to a value.
func ptr[T any](v T) *T {
	return &v
}
