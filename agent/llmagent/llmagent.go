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
//	    llmagent.WithInterceptors(myInterceptor),
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

		// Ensure system prompt is present
		messages := a.ensureSystemPrompt(sess.Messages)

		// Execute turn loop
		for invCtx.Turn() < a.config.maxTurns {
			// Emit turn started
			if !yield(agent.StatusEvent{
				Envelope: makeEnvelope(),
				Stage:    agent.StatusStageTurnStarted,
				Details:  fmt.Sprintf("turn %d started", invCtx.Turn()),
			}, nil) {
				return
			}

			// Check context cancellation
			if invCtx.Err() != nil {
				yield(agent.InvocationEndEvent{
					Envelope:     makeEnvelope(),
					FinishReason: agent.FinishReasonInterrupted,
					Usage:        ptr(invCtx.TotalUsage()),
				}, nil)

				return
			}

			// Create turn execution function that can be wrapped by interceptors
			// This encapsulates the entire turn execution logic
			executeTurn := func(ctx context.Context) (agent.FinishReason, error) {
				return a.executeSingleTurn(ctx, &messages, makeEnvelope, yield)
			}

			// Apply turn interceptors
			wrappedTurn := agent.ApplyTurnInterceptors(invCtx, a.config.interceptors, executeTurn)

			// Execute the turn (wrapped by interceptors)
			finishReason, err := wrappedTurn(invCtx)
			if err != nil {
				// Terminal error from turn execution
				yield(nil, err)
				return
			}

			// Check if interceptor or turn logic wants to end execution
			if finishReason != "" {
				// Emit terminal event
				yield(agent.InvocationEndEvent{
					Envelope:     makeEnvelope(),
					FinishReason: finishReason,
					Usage:        ptr(invCtx.TotalUsage()),
				}, nil)

				return
			}

			// Increment turn for next iteration
			invCtx.IncrementTurn()
		}

		// Max turns reached
		yield(agent.InvocationEndEvent{
			Envelope:     makeEnvelope(),
			FinishReason: agent.FinishReasonMaxTurns,
			Usage:        ptr(invCtx.TotalUsage()),
		}, nil)
	}
}

// executeSingleTurn executes a single turn of the agent loop.
//
// Returns:
//   - FinishReason: non-empty if execution should stop (terminal condition reached)
//   - error: only for terminal errors that should stop execution
//
// When FinishReason is empty, the turn completed normally and the loop should continue.
func (a *LLMAgent) executeSingleTurn(
	ctx context.Context,
	messages *[]llm.Message,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) (agent.FinishReason, error) {
	// Get invocation context (guaranteed to be InvocationContext)
	//nolint:errcheck,forcetypeassert // Type assertion guaranteed by function contract - panic is intentional for programming errors
	invCtx := ctx.(*agent.InvocationContext)
	sess := invCtx.Session()

	// Emit model call status
	if !yield(agent.StatusEvent{
		Envelope: makeEnvelope(),
		Stage:    agent.StatusStageModelCall,
		Details:  "invoking model",
	}, nil) {
		// Consumer stopped listening - return interrupted
		return agent.FinishReasonInterrupted, nil
	}

	// Prepare request
	req := &llm.Request{
		Messages: *messages,
	}
	if a.config.tools != nil {
		req.Tools = a.config.tools.List()
	}

	// Apply model interceptors for this request
	// This wraps the models Generate/GenerateEvents with interceptor logic
	//nolint:contextcheck // invCtx embeds context.Context and is designed to be used as a context
	model := agent.ApplyModelInterceptors(invCtx, req, a.config.model, a.config.interceptors)

	// Generate response from LLM (with streaming support if available)
	//nolint:contextcheck // invCtx embeds context.Context and is designed to be used as a context
	resp, err := a.generate(invCtx, model, req, makeEnvelope, yield)
	if err != nil {
		// TERMINAL ERROR: System failure (auth, connection, protocol violation)
		// Observable errors (rate limits, content filters) come through:
		// - FinishReason from model (handled in terminal finish reasons block below)
		// - ErrorEvent in stream (non-terminal, handled in generateWithStreaming)
		return "", err
	}

	// Update usage tracking
	invCtx.AddUsage(resp.Usage)

	// Add assistant message to session and local history
	sess.Messages = append(sess.Messages, resp.Message)
	*messages = append(*messages, resp.Message)

	// Emit message event
	if !yield(agent.MessageEvent{
		Envelope: makeEnvelope(),
		Response: *resp,
	}, nil) {
		// Consumer stopped listening
		return agent.FinishReasonInterrupted, nil
	}

	// Check for terminal finish reasons from the model
	agentReason, terminalErr := mapLLMFinishReason(resp.FinishReason)
	if agentReason != "" {
		// Terminal finish reason - handle completion
		if terminalErr != nil {
			// Emit error for terminal error conditions (content filter, interrupted, unknown)
			yield(agent.ErrorEvent{
				Envelope: makeEnvelope(),
				Err:      terminalErr,
				Message:  terminalErr.Error(),
			}, nil)
		} else if agentReason == agent.FinishReasonLength {
			// Emit status event for length limit (non-error terminal case)
			yield(agent.StatusEvent{
				Envelope: makeEnvelope(),
				Stage:    agent.StatusStageTurnCompleted,
				Details:  fmt.Sprintf("turn %d completed - length limit", invCtx.Turn()),
				Usage:    resp.Usage,
			}, nil)
		}

		return agentReason, nil
	}
	// Non-terminal finish reason (ToolCalls or Stop) - continue below

	// Check for tool calls
	toolReqs := resp.ToolRequests()
	if len(toolReqs) == 0 {
		// No tools requested - natural completion
		// Emit turn completed
		yield(agent.StatusEvent{
			Envelope: makeEnvelope(),
			Stage:    agent.StatusStageTurnCompleted,
			Details:  fmt.Sprintf("turn %d completed", invCtx.Turn()),
			Usage:    resp.Usage,
		}, nil)

		return agent.FinishReasonStop, nil
	}

	// Emit tool call events
	for _, toolReq := range toolReqs {
		if !yield(agent.ToolRequestEvent{
			Envelope: makeEnvelope(),
			Request:  *toolReq,
		}, nil) {
			// Consumer stopped listening
			return agent.FinishReasonInterrupted, nil
		}
	}

	// Emit tool execution status
	if !yield(agent.StatusEvent{
		Envelope: makeEnvelope(),
		Stage:    agent.StatusStageToolExec,
		Details:  fmt.Sprintf("executing %d tools", len(toolReqs)),
	}, nil) {
		// Consumer stopped listening
		return agent.FinishReasonInterrupted, nil
	}

	// Execute tools and collect results
	if a.config.tools == nil {
		return "", agent.ErrToolRegistry
	}

	//nolint:contextcheck // invCtx embeds context.Context and is designed to be used as a context
	toolMessages := a.executeTools(invCtx, toolReqs, makeEnvelope, yield)

	// Add tool results to conversation (including errors - LLM handles them gracefully)
	for _, msg := range toolMessages {
		sess.Messages = append(sess.Messages, msg)
		*messages = append(*messages, msg)
	}

	// Emit turn completed
	if !yield(agent.StatusEvent{
		Envelope: makeEnvelope(),
		Stage:    agent.StatusStageTurnCompleted,
		Details:  fmt.Sprintf("turn %d completed", invCtx.Turn()),
	}, nil) {
		// Consumer stopped listening
		return agent.FinishReasonInterrupted, nil
	}

	// Turn completed normally - continue loop
	return "", nil
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
// The model parameter is the potentially intercepted model (wrapped by hooks).
// If the model supports streaming (implements llm.EventsGenerator),
// it will emit AssistantDeltaEvent for each content part as it arrives.
func (a *LLMAgent) generate(
	invCtx *agent.InvocationContext,
	model llm.Model,
	req *llm.Request,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) (*llm.Response, error) {
	// Use streaming if model supports it (provides better UX with real-time updates)
	if eg, ok := model.(llm.EventsGenerator); ok {
		return a.generateWithStreaming(invCtx, eg, req, makeEnvelope, yield)
	}

	// Fall back to non-streaming generation
	return model.Generate(invCtx, req)
}

// generateWithStreaming uses the EventsGenerator interface to get token-by-token deltas.
// Each delta is emitted as an AssistantDeltaEvent for real-time streaming feedback.
func (a *LLMAgent) generateWithStreaming(
	ctx context.Context,
	eg llm.EventsGenerator,
	req *llm.Request,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) (*llm.Response, error) {
	var response *llm.Response

	for event, err := range eg.GenerateEvents(ctx, req) {
		if err != nil {
			return nil, fmt.Errorf("%w: %w", agent.ErrModelGeneration, err)
		}

		switch evt := event.(type) {
		case llm.ContentPartEvent:
			// Emit real-time delta for streaming consumers
			if !yield(agent.AssistantDeltaEvent{
				Envelope: makeEnvelope(),
				Delta:    evt,
			}, nil) {
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
			if !yield(agent.ErrorEvent{
				Envelope: makeEnvelope(),
				Err:      fmt.Errorf("%w: %s", agent.ErrModelGeneration, evt.Message),
				Message:  evt.Message,
			}, nil) {
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
// (including context cancellation, timeouts, etc.) are captured and sent
// to the LLM as error tool responses, allowing the LLM to handle failures
// gracefully (acknowledge, retry, use different tool, etc.).
//
// This follows the pattern from ADK and other SDKs: tool errors are NEVER
// terminal - they're always sent to the LLM as part of the conversation.
//
// ToolResponseEvents are yielded as tools complete.
//
// Returns tool result messages to be added to the conversation.
//
// Future: Will also return list of tool IDs requiring input for
// StatusStageInputRequired / FinishReasonInputRequired support.
func (a *LLMAgent) executeTools(
	invCtx *agent.InvocationContext,
	toolReqs []*llm.ToolRequest,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) []llm.Message {
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

	// Create base tool executor
	baseExecutor := func(ctx context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
		return a.config.tools.Execute(ctx, req)
	}

	// Apply tool interceptors
	executor := agent.ApplyToolInterceptors(ctx, a.config.interceptors, baseExecutor)

	// Launch tool executions
	for i, req := range toolReqs {
		g.Go(func() error {
			resp, err := executor(ctx, req)
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
	// Note: g.Wait() should never return an error since goroutines always return nil
	_ = g.Wait()

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
			msg = llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(errResp))

			// Yield error tool result event
			if !yield(agent.ToolResponseEvent{
				Envelope: makeEnvelope(),
				Response: *errResp,
			}, nil) {
				return messages // Consumer stopped listening
			}
		} else {
			// Tool execution succeeded
			msg = llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(result.response))

			// Yield tool result event
			if !yield(agent.ToolResponseEvent{
				Envelope: makeEnvelope(),
				Response: *result.response,
			}, nil) {
				return messages // Consumer stopped listening
			}
		}

		messages = append(messages, msg)
	}

	return messages
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

// ptr is a helper to get a pointer to a value.
func ptr[T any](v T) *T {
	return &v
}
