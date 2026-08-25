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

// Package llmagent provides an LLM-based agent implementation with tool calling support.
package llmagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
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

// Info returns the agent's identity snapshot.
func (a *LLMAgent) Info() agent.Info {
	return agent.Info{
		Name:         a.config.name,
		Description:  a.config.description,
		SystemPrompt: a.config.systemPrompt,
		ID:           a.config.id,
		Version:      a.config.version,
		ModelName:    a.config.model.Name(),
		ProviderName: a.config.model.Provider(),
	}
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
func (a *LLMAgent) Run(ctx context.Context, inv *agent.InvocationMetadata) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		// Helper: create event envelope
		makeEnvelope := func() agent.EventEnvelope {
			return agent.EventEnvelope{
				InvocationID: inv.InvocationID(),
				SessionID:    inv.Session().ID,
				Turn:         inv.Turn(),
				At:           time.Now().UTC(),
			}
		}

		// Recover incomplete tool calls before the first turn executes.
		// This handles sessions where the previous invocation was interrupted
		// after the assistant emitted tool requests but before tool responses
		// were added to the session.
		if err := a.recoverIncompleteToolCalls(ctx, inv, makeEnvelope, yield); err != nil {
			yield(nil, err)
			return
		}

		// Execute turn loop
		for inv.Turn() < a.config.maxTurns {
			// Emit turn started
			if !yield(agent.StatusEvent{
				Envelope: makeEnvelope(),
				Stage:    agent.StatusStageTurnStarted,
				Details:  fmt.Sprintf("turn %d started", inv.Turn()),
			}, nil) {
				return
			}

			// Check context cancellation
			if ctx.Err() != nil {
				yield(agent.InvocationEndEvent{
					Envelope:     makeEnvelope(),
					FinishReason: agent.FinishReasonInterrupted,
					Usage:        new(inv.TotalUsage()),
				}, nil)

				return
			}

			// Create turn execution function that can be wrapped by interceptors
			// This encapsulates the entire turn execution logic
			executeTurn := func(ctx context.Context, info *agent.TurnInfo) (agent.FinishReason, error) {
				return a.executeSingleTurn(ctx, info.Inv, makeEnvelope, yield)
			}

			// Apply turn interceptors
			wrappedTurn := agent.ApplyTurnInterceptors(a.config.interceptors, executeTurn)

			// Execute the turn (wrapped by interceptors)
			finishReason, err := wrappedTurn(ctx, &agent.TurnInfo{Inv: inv})
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
					Usage:        new(inv.TotalUsage()),
				}, nil)

				return
			}

			// Increment turn for next iteration
			agent.IncrementTurn(inv)
		}

		// Max turns reached
		yield(agent.InvocationEndEvent{
			Envelope:     makeEnvelope(),
			FinishReason: agent.FinishReasonMaxTurns,
			Usage:        new(inv.TotalUsage()),
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
	inv *agent.InvocationMetadata,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) (agent.FinishReason, error) {
	sess := inv.Session()

	// Emit model call status
	if !yield(agent.StatusEvent{
		Envelope: makeEnvelope(),
		Stage:    agent.StatusStageModelCall,
		Details:  "invoking model",
	}, nil) {
		// Consumer stopped listening - return interrupted
		return agent.FinishReasonInterrupted, nil
	}

	// Build working message list with system prompt (not persisted)
	// This creates a transient view for the LLM request
	reqMessages, err := a.resolveSystemPrompt(ctx, inv, sess.Messages)
	if err != nil {
		return "", fmt.Errorf("llmagent: system prompt: %w", err)
	}

	var toolDefs []llm.ToolDefinition
	if a.config.tools != nil {
		toolDefs = a.config.tools.List()
	}

	// One compaction check before every model call: this position sees an
	// oversized fresh user message and every burst of tool results. The
	// fixed cost (system prompt + schemas) rides on every request, so it is
	// part of the context budget the history must fit into. Interceptors may
	// still grow the request after this check; the reactive overflow path
	// below is the backstop, and its retry re-enters the interceptor chain
	// (interceptors transform per Generate call, so nothing is lost).
	fixedTokens := 0
	scale := tokenScale(inv)

	if a.config.compaction != nil {
		fixedTokens = estimateMessageTokens(reqMessages[0]) + estimateToolTokens(toolDefs)

		stats, fitErr := a.ensureFits(sess, fixedTokens, scale)
		if stats.changed() {
			reqMessages = append([]llm.Message{reqMessages[0]}, sess.Messages...)

			if !yield(agent.StatusEvent{
				Envelope: makeEnvelope(),
				Stage:    agent.StatusStageCompaction,
				Details:  compactionDetails(stats, scale),
			}, nil) {
				return agent.FinishReasonInterrupted, nil
			}
		}

		if fitErr != nil {
			return "", fitErr
		}
	}

	// Prepare request
	req := &llm.Request{
		Messages: reqMessages,
		Tools:    toolDefs,
	}

	// Apply model interceptors for this request
	// This wraps the models Generate/GenerateEvents with interceptor logic
	modelInfo := &agent.ModelCallInfo{
		InvocationMetadata: inv,
		Model:              a.config.model,
		Req:                req,
	}
	model := agent.ApplyModelInterceptors(ctx, modelInfo, a.config.model, a.config.interceptors)

	// Generate response from LLM (with streaming support if available)
	resp, err := a.generate(ctx, model, req, makeEnvelope, yield)
	if err != nil && a.config.compaction != nil && errors.Is(err, llm.ErrContextOverflow) {
		// Reactive path: the provider rejected the request pre-flight, so
		// nothing was emitted. Force a strictly smaller request - hard
		// floors, at least 25% below the failed size - and retry once.
		stats, reduced := a.reduceAfterOverflow(sess, fixedTokens, scale)
		if !reduced {
			return "", cannotFitError(stats.afterTokens, a.deriveContextBudget(scale), scale)
		}

		if !yield(agent.StatusEvent{
			Envelope: makeEnvelope(),
			Stage:    agent.StatusStageCompaction,
			Details:  compactionDetails(stats, scale) + " (after provider overflow)",
		}, nil) {
			return agent.FinishReasonInterrupted, nil
		}

		req.Messages = append([]llm.Message{reqMessages[0]}, sess.Messages...)

		resp, err = a.generate(ctx, model, req, makeEnvelope, yield)
		if err != nil && errors.Is(err, llm.ErrContextOverflow) {
			return "", fmt.Errorf("llmagent: request still exceeds the context window after forced compaction: %w", err)
		}
	}

	if err != nil {
		// TERMINAL ERROR: System failure (auth, connection, protocol violation)
		// Observable errors (rate limits, content filters) come through:
		// - FinishReason from model (handled in terminal finish reasons block below)
		// - ErrorEvent in stream (non-terminal, handled in generateWithStreaming)
		return "", err
	}

	// Update usage tracking
	agent.AddUsage(inv, resp.Usage)

	// Calibrate: the billed input tokens measure the estimator's error on
	// this session's content; later budget checks this run use the observed
	// ratio (calibration.go).
	if a.config.compaction != nil && resp.Usage != nil {
		estSent := fixedTokens + estimateHistoryTokens(req.Messages[1:])
		observeUsage(inv, resp.Usage.BilledInputTokens(), estSent)
		scale = tokenScale(inv)
	}

	// Add assistant message to session (single source of truth).
	//
	// Skip empty-content turns: a max_tokens cut can produce an assistant
	// message with no parts (e.g. its only block was a partial tool_use the
	// provider dropped). Persisting it poisons the session — on the next
	// request the provider replays a content-less assistant message and
	// Anthropic rejects it with "messages.N.content: Field required". The
	// FinishReason (Length) is read from resp below regardless, so the
	// truncation signal still propagates. A legitimately truncated tool-use
	// turn keeps its completed tool_use parts (len > 0) and is unaffected.
	//
	// This guard covers the persisted session store only. The MessageEvent
	// below still carries the raw resp.Message, so a consumer that rebuilds
	// history purely from the event stream (rather than from sess.Messages)
	// can still observe the empty turn — the provider request-mappers'
	// empty-content substitution is the authoritative backstop for that path.
	if len(resp.Message.Content) > 0 {
		sess.Messages = append(sess.Messages, resp.Message)
	}

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
				Details:  fmt.Sprintf("turn %d completed - length limit", inv.Turn()),
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
			Details:  fmt.Sprintf("turn %d completed", inv.Turn()),
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

	// The per-result cap for this turn is fixed before any tool runs, so a
	// parallel burst cannot assemble an unfittable frontier and runs are
	// reproducible regardless of completion order.
	countedRequest := fixedTokens + estimateHistoryTokens(sess.Messages)
	resultCap := a.effectiveResultCap(countedRequest, len(toolReqs), scale)

	toolParts := a.executeTools(ctx, inv, toolReqs, req.Tools, resultCap, makeEnvelope, yield)

	// Build single message with all tool response parts
	toolMsg := llm.NewMessage(llm.RoleUser, toolParts...)
	sess.Messages = append(sess.Messages, toolMsg)

	// Emit turn completed
	if !yield(agent.StatusEvent{
		Envelope: makeEnvelope(),
		Stage:    agent.StatusStageTurnCompleted,
		Details:  fmt.Sprintf("turn %d completed", inv.Turn()),
	}, nil) {
		// Consumer stopped listening
		return agent.FinishReasonInterrupted, nil
	}

	// Turn completed normally - continue loop
	return "", nil
}

// resolveSystemPrompt produces a transient message list with the system
// prompt prepended. The system prompt is never persisted to the session.
//
// When a [SystemPromptProvider] is configured it is called every turn,
// receiving both the request context and the invocation metadata.
// Otherwise the static systemPrompt string from the config is used.
func (a *LLMAgent) resolveSystemPrompt(ctx context.Context, inv *agent.InvocationMetadata, messages []llm.Message) ([]llm.Message, error) {
	prompt := a.config.systemPrompt
	if a.config.systemPromptProvider != nil {
		p, err := a.config.systemPromptProvider(ctx, inv)
		if err != nil {
			return nil, err
		}

		prompt = p
	}

	systemMsg := llm.NewMessage(llm.RoleSystem, llm.NewTextPart(prompt))

	// Strip an existing system message — it's always stale since the
	// session never stores one; this guards against a previous turn's
	// transient copy leaking in.
	if len(messages) > 0 && messages[0].Role == llm.RoleSystem {
		messages = messages[1:]
	}

	return append([]llm.Message{systemMsg}, messages...), nil
}

// generate calls the LLM to generate a response.
//
// The model parameter is the potentially intercepted model (wrapped by interceptors).
// If the model supports streaming (implements llm.EventsGenerator),
// it will emit AssistantDeltaEvent for each content part as it arrives.
func (a *LLMAgent) generate(
	ctx context.Context,
	model llm.Model,
	req *llm.Request,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) (*llm.Response, error) {
	// Use streaming if model supports it (provides better UX with real-time updates)
	if eg, ok := model.(llm.EventsGenerator); ok {
		return a.generateWithStreaming(ctx, eg, req, makeEnvelope, yield)
	}

	// Fall back to non-streaming generation
	return model.Generate(ctx, req)
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

		case llm.StreamResetEvent:
			// Stream is being retried — reset accumulated state and notify consumer.
			// Only response needs resetting here; provider-level state (content block
			// accumulators, aggregated parts, etc.) is implicitly reset when the retry
			// interceptor calls GenerateEvents() again, creating a fresh stream context.
			response = nil

			if !yield(agent.StreamResetEvent{
				Envelope: makeEnvelope(),
				Attempt:  evt.Attempt,
				Reason:   evt.Reason,
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
// Returns tool response parts in the order they were requested.
//
// Future: Will also return list of tool IDs requiring input for
// StatusStageInputRequired / FinishReasonInputRequired support.
func (a *LLMAgent) executeTools(
	ctx context.Context,
	inv *agent.InvocationMetadata,
	toolReqs []*llm.ToolRequestPart,
	toolDefs []llm.ToolDefinition,
	resultCap int,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) []llm.Part {
	// Execute tools concurrently with limited parallelism
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(min(a.config.toolConcurrency, len(toolReqs)))

	// Results channel (buffered to avoid blocking)
	type toolResult struct {
		idx       int
		requestID string
		name      string
		response  *llm.ToolResponsePart
		err       error
	}

	results := make(chan toolResult, len(toolReqs))

	// Create base tool executor
	baseExecutor := func(ctx context.Context, info *agent.ToolCallInfo) (*llm.ToolResponsePart, error) {
		return a.config.tools.Execute(ctx, info.Req)
	}

	// Apply tool interceptors
	executor := agent.ApplyToolInterceptors(a.config.interceptors, baseExecutor)

	// Build tool definition lookup map for interceptors from provided definitions
	toolDefMap := make(map[string]*llm.ToolDefinition, len(toolDefs))
	for i := range toolDefs {
		toolDefMap[toolDefs[i].Name] = &toolDefs[i]
	}

	// Launch tool executions
	for i, req := range toolReqs {
		g.Go(func() error {
			toolInfo := &agent.ToolCallInfo{
				Inv:        inv,
				Req:        req,
				Definition: toolDefMap[req.Name], // Add tool definition
			}

			// Expose the conversation grouping id for the entire tool call —
			// interceptors, retries, and the tool itself — so tools that spawn
			// in-process sub-agents (agenttool) group them under the calling
			// conversation. Tools that do not read it are unaffected.
			toolCtx := agent.ContextWithConversationID(gctx, session.ConversationID(inv.Session()))

			resp, err := executor(toolCtx, toolInfo)
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

	// Collect tool response parts as they arrive. Events are yielded in
	// completion order, but each part is stored at its request index so the
	// session content is deterministic regardless of scheduling (I9).
	parts := make([]llm.Part, len(toolReqs))

	for range toolReqs {
		result := <-results

		if result.err != nil {
			// Tool execution failed - encode error payload into the response.
			errPayload, mErr := json.Marshal(map[string]string{"error": result.err.Error()})
			if mErr != nil {
				errPayload = []byte(`{"error":"tool error"}`)
			}

			errResp := capToolResult(&llm.ToolResponsePart{
				ID:      result.requestID,
				Name:    result.name,
				Result:  errPayload,
				IsError: true,
			}, resultCap)
			parts[result.idx] = errResp

			// Yield error tool result event
			if !yield(agent.ToolResponseEvent{
				Envelope: makeEnvelope(),
				Response: *errResp,
			}, nil) {
				return collectedParts(parts) // Consumer stopped listening
			}
		} else {
			// Tool execution succeeded. The event carries the capped result -
			// the same bytes the model will see - so session, events and
			// prompt never diverge.
			resp := capToolResult(result.response, resultCap)
			parts[result.idx] = resp

			// Yield tool result event
			if !yield(agent.ToolResponseEvent{
				Envelope: makeEnvelope(),
				Response: *resp,
			}, nil) {
				return collectedParts(parts) // Consumer stopped listening
			}
		}
	}

	return parts
}

// collectedParts drops the unfilled slots of an index-addressed part slice.
// Only relevant when the consumer stopped mid-burst: a message must never
// carry nil parts.
func collectedParts(parts []llm.Part) []llm.Part {
	out := parts[:0]

	for _, p := range parts {
		if p != nil {
			out = append(out, p)
		}
	}

	return out
}

// recoverIncompleteToolCalls detects and executes incomplete tool calls from a
// previous interrupted invocation.
//
// An incomplete tool call occurs when:
//  1. The assistant responds with tool requests
//  2. The session is saved (runner saves after MessageEvent)
//  3. The process crashes/disconnects before tool execution completes
//  4. A new user message arrives, appended to the session by the runner
//
// The resulting session has: [..., assistant(tool_request), user(text)] with no
// tool response in between. LLMs reject this with "No tool output found for function call".
//
// This method detects the pattern, executes the incomplete tools, and inserts the
// tool response message before the new user message, repairing the session.
//
// Error handling:
//   - Tool execution errors are captured in ToolResponse.Error and become part
//     of the repaired session. The LLM can reason about these failures.
//   - Context cancellation stops the yield loop, terminating recovery gracefully.
//   - If yield returns false (consumer stopped), recovery aborts without error.
//
// Observability: A StatusEvent with stage ToolExec is emitted before executing
// incomplete tools, indicating how many are being recovered.
func (a *LLMAgent) recoverIncompleteToolCalls(
	ctx context.Context,
	inv *agent.InvocationMetadata,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) error {
	sess := inv.Session()

	incomplete := detectIncompleteToolCalls(sess.Messages)
	if len(incomplete) == 0 {
		return nil
	}

	// Emit status: recovering incomplete tools
	if !yield(agent.StatusEvent{
		Envelope: makeEnvelope(),
		Stage:    agent.StatusStageToolExec,
		Details:  fmt.Sprintf("recovering %d incomplete tool calls from interrupted session", len(incomplete)),
	}, nil) {
		return nil // Consumer stopped
	}

	// Need tool registry to execute
	if a.config.tools == nil {
		return agent.ErrToolRegistry
	}

	// Execute the incomplete tools
	toolDefs := a.config.tools.List()
	// Recovered results land in the unread frontier, which compaction can
	// never reduce - so the burst budget applies here exactly as in normal
	// execution. The system prompt is not resolved yet, so counting history
	// alone undercounts slightly; the count-high bias absorbs that.
	resultCap := a.effectiveResultCap(estimateHistoryTokens(sess.Messages), len(incomplete), tokenScale(inv))
	toolParts := a.executeTools(ctx, inv, incomplete, toolDefs, resultCap, makeEnvelope, yield)

	// Insert tool response message BEFORE the last user message.
	// Current: [..., assistant(tool_req), user(text)]
	// After:   [..., assistant(tool_req), user(tool_resp), user(text)]
	toolMsg := llm.NewMessage(llm.RoleUser, toolParts...)
	lastIdx := len(sess.Messages) - 1
	sess.Messages = append(sess.Messages[:lastIdx], toolMsg, sess.Messages[lastIdx])

	return nil
}

// detectIncompleteToolCalls checks if the session ends with incomplete tool calls.
//
// Returns the incomplete tool requests if found, nil otherwise.
//
// Pattern detected: [..., assistant(tool_requests), user(text_only)]
// The user message has text but no tool responses, indicating the previous
// invocation was interrupted after tool requests but before tool execution.
//
// Why tail-only detection is correct:
// Incomplete tool calls can only occur at the session tail. The sequence is:
//  1. Runner receives user message, appends to session, calls agent
//  2. Agent generates response with tool requests
//  3. Runner saves session after MessageEvent (assistant message persisted)
//  4. Crash/disconnect before tool execution completes
//  5. New user message arrives, runner loads session and appends it
//
// The incomplete calls are always between the last assistant message and the
// new user message. Incomplete calls earlier in the session would indicate a
// different bug (session corruption, not crash recovery).
func detectIncompleteToolCalls(msgs []llm.Message) []*llm.ToolRequestPart {
	if len(msgs) < 2 {
		return nil
	}

	lastIdx := len(msgs) - 1
	lastMsg := msgs[lastIdx]
	prevMsg := msgs[lastIdx-1]

	// Last should be user (new message from runner), prev should be assistant
	if lastMsg.Role != llm.RoleUser || prevMsg.Role != llm.RoleAssistant {
		return nil
	}

	// Previous (assistant) message must have tool requests
	toolReqs := prevMsg.ToolRequests()
	if len(toolReqs) == 0 {
		return nil
	}

	// If last (user) message has tool responses, session is valid
	if len(lastMsg.ToolResponses()) > 0 {
		return nil
	}

	// Incomplete tool calls detected
	return toolReqs
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

	case llm.FinishReasonContextOverflow:
		// Terminal but not an error condition — the executor turns this into a
		// truthful "conversation too long" failure, distinct from output
		// truncation (FinishReasonLength).
		return agent.FinishReasonContextOverflow, nil

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
