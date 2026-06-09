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
	"errors"
	"fmt"
	"iter"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
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

// Tools returns the agent's tool registry. The runner uses this to
// re-enter tools on ResumeWithReentry pauses without taking a direct
// dependency on this package.
func (a *LLMAgent) Tools() tool.Registry { return a.config.tools }

// ExecuteToolResume re-enters a paused tool call through the agent's
// tool interceptor chain. The runner calls this (via a duck-typed
// interface) when a ResumeWithReentry pending call is resumed, so that
// an interceptor that paused the call — an approval gate in particular —
// consumes the decision before the tool runs. See
// agent.ToolCallInfo.Resume for the consume contract.
func (a *LLMAgent) ExecuteToolResume(
	ctx context.Context,
	sess *session.State,
	pc session.PendingToolCall,
	payload *tool.ResumePayload,
) (tool.Execution, error) {
	if a.config.tools == nil {
		return tool.Execution{}, agent.ErrToolRegistry
	}

	inv := agent.NewInvocationMetadata(sess, a.Info())

	toolInv := tool.InvocationInfo{
		InvocationID: inv.InvocationID(),
		SessionID:    sess.ID,
		AgentName:    a.config.name,
	}

	req := &llm.ToolRequestPart{ID: pc.ID, Name: pc.Name, Arguments: pc.Arguments}

	// The base executor reads info.Resume at call time: the interceptor
	// that paused the call consumes it (sets info.Resume = nil) before
	// calling next, which makes the tool run fresh. If no interceptor
	// consumed it, the payload reaches the tool as Call.Resume.
	baseExecutor := func(ctx context.Context, info *agent.ToolCallInfo) (tool.Execution, error) {
		if info.Resume == nil {
			res := a.config.tools.Run(ctx, toolInv, info.Req)
			return res.Execution, res.Err
		}

		res := a.config.tools.Resume(ctx, toolInv, info.Req, info.Resume)

		return res.Execution, res.Err
	}

	executor := agent.ApplyToolInterceptors(a.config.interceptors, baseExecutor)

	var def *llm.ToolDefinition

	if t, err := a.config.tools.Get(pc.Name); err == nil {
		d := tool.Definition(t)
		def = &d
	}

	return executor(ctx, &agent.ToolCallInfo{
		Inv:        inv,
		Req:        req,
		Definition: def,
		Resume:     payload,
	})
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
				end := agent.InvocationEndEvent{
					Envelope:     makeEnvelope(),
					FinishReason: finishReason,
					Usage:        new(inv.TotalUsage()),
				}

				// On pause, surface the pending calls that were just
				// recorded on the session so adapters can route their
				// A2A / UI streams without a second session load.
				if finishReason == agent.FinishReasonPaused {
					if sess := inv.Session(); sess != nil {
						for _, pc := range sess.PendingToolCalls {
							end.PendingCalls = append(end.PendingCalls, pendingToCallSummary(pc))
						}
					}
				}

				yield(end, nil)

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

	// Prepare request
	req := &llm.Request{
		Messages: reqMessages,
	}
	if a.config.tools != nil {
		req.Tools = a.config.tools.List()
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
	if err != nil {
		// TERMINAL ERROR: System failure (auth, connection, protocol violation)
		// Observable errors (rate limits, content filters) come through:
		// - FinishReason from model (handled in terminal finish reasons block below)
		// - ErrorEvent in stream (non-terminal, handled in generateWithStreaming)
		return "", err
	}

	// Update usage tracking
	agent.AddUsage(inv, resp.Usage)

	// Add assistant message to session (single source of truth)
	sess.Messages = append(sess.Messages, resp.Message)

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

	toolParts, pendingCalls := a.executeTools(ctx, inv, toolReqs, req.Tools, makeEnvelope, yield)

	// Build single message with all tool response parts. Placeholders for
	// paused tools live in this message so the session is internally
	// valid (every assistant tool_request is paired with a user
	// tool_response). The runner replaces them with final results on
	// resume.
	toolMsg := llm.NewMessage(llm.RoleUser, toolParts...)
	sess.Messages = append(sess.Messages, toolMsg)

	// If any tools paused, store the typed pending calls on the session
	// and stop the invocation. Resume is driven by runner.Resume /
	// runner.Run (for ResumeWithMessage pauses). The outer agent loop
	// builds the PendingCalls slice for InvocationEndEvent from the
	// session after our return; we don't need to build it here.
	if len(pendingCalls) > 0 {
		if sess.PendingToolCalls == nil {
			sess.PendingToolCalls = make(map[string]session.PendingToolCall, len(pendingCalls))
		}

		for _, pc := range pendingCalls {
			sess.PendingToolCalls[pc.ID] = pc
		}

		_ = yield(agent.StatusEvent{
			Envelope: makeEnvelope(),
			Stage:    agent.StatusStagePaused,
			Details:  fmt.Sprintf("paused: %d pending tool call(s)", len(pendingCalls)),
		}, nil)

		return agent.FinishReasonPaused, nil
	}

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

// pendingToCallSummary projects a stored PendingToolCall onto the
// surface adapter consumers see. Keeping this conversion local avoids
// agent → session import dependencies in the event types.
func pendingToCallSummary(pc session.PendingToolCall) agent.PendingCallSummary {
	return agent.PendingCallSummary{
		CallID:        pc.ID,
		ToolName:      pc.Name,
		Reason:        tool.AwaitReason(pc.Reason),
		Resume:        tool.ResumeMode(pc.Resume),
		Message:       pc.Message,
		Prompt:        pc.Prompt,
		CorrelationID: pc.CorrelationID,
		ExpiresAt:     pc.ExpiresAt,
	}
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
// Returns tool response parts in the order they were requested, plus a
// list of any pending tool calls that paused execution. The caller
// persists the pending calls on the session and stops the invocation.
//
// Tools may pause by returning Execution.Await; the runtime then
// produces a placeholder ToolResponsePart (so the session stays
// internally valid) and a typed PendingToolCall that the runner uses
// to drive Resume/Progress/Cancel.
func (a *LLMAgent) executeTools(
	ctx context.Context,
	inv *agent.InvocationMetadata,
	toolReqs []*llm.ToolRequestPart,
	toolDefs []llm.ToolDefinition,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) ([]llm.Part, []session.PendingToolCall) {
	// Execute tools concurrently with limited parallelism
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(min(a.config.toolConcurrency, len(toolReqs)))

	// Results channel (buffered to avoid blocking)
	type toolResult struct {
		idx       int
		requestID string
		name      string
		execution tool.Execution
		err       error
	}

	results := make(chan toolResult, len(toolReqs))

	now := time.Now().UTC()

	// Build tool.InvocationInfo from agent metadata to thread through
	// the registry. The registry no longer reaches into agent types
	// directly; this keeps tool.* free of an agent-package import cycle.
	var sessionID string
	if sess := inv.Session(); sess != nil {
		sessionID = sess.ID
	}

	toolInv := tool.InvocationInfo{
		InvocationID: inv.InvocationID(),
		SessionID:    sessionID,
		Turn:         inv.Turn(),
		AgentName:    inv.Agent().Name,
	}

	// Base executor returns a tool.Execution. The interceptor chain
	// preserves that shape end-to-end so any interceptor can pause via
	// Execution.Await (approval gates, MCP elicitation, etc.).
	baseExecutor := func(ctx context.Context, info *agent.ToolCallInfo) (tool.Execution, error) {
		res := a.config.tools.Run(ctx, toolInv, info.Req)
		return res.Execution, res.Err
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

			exec, err := executor(gctx, toolInfo)
			results <- toolResult{
				idx:       i,
				requestID: req.ID,
				name:      req.Name,
				execution: exec,
				err:       err,
			}

			return nil // Never return error to errgroup (we handle errors individually)
		})
	}

	// Collect tool response parts and yield events as they arrive.
	// Reconcile Execution + Err into a wire-format ToolResponsePart via
	// the registry's ExecutionResult.Response helper so the shape stays
	// in one place. Pauses (Execution.Await != nil) produce a
	// PendingToolCall in addition to the placeholder response.
	//
	// Events stream in completion order (responsive UX), but parts are
	// placed by request index: the persisted tool-response message must
	// mirror the order of the tool_use blocks in the assistant message,
	// or providers reject the request.
	ordered := make([]llm.Part, len(toolReqs))

	var pending []session.PendingToolCall

	for range toolReqs {
		result := <-results

		req := toolReqs[result.idx]
		execResult := tool.ExecutionResult{
			Index:     result.idx,
			Request:   req,
			Execution: result.execution,
			Err:       result.err,
		}
		resp := execResult.Response()

		ordered[result.idx] = resp

		// If this call paused, record the typed pending entry and emit
		// a ToolPendingEvent before the (placeholder) ToolResponseEvent
		// so adapters can see "this is non-terminal" alongside the
		// placeholder shape they need to mirror in their own streams.
		if exec := result.execution; exec.Await != nil {
			pc := buildPendingCall(req, exec, now)
			pending = append(pending, pc)

			if !yield(agent.ToolPendingEvent{
				Envelope:    makeEnvelope(),
				PendingCall: pendingToCallSummary(pc),
				Placeholder: *resp,
			}, nil) {
				return compactParts(ordered), pending // Consumer stopped listening
			}
		}

		// Yield tool result event
		if !yield(agent.ToolResponseEvent{
			Envelope: makeEnvelope(),
			Response: *resp,
		}, nil) {
			return compactParts(ordered), pending // Consumer stopped listening
		}
	}

	return compactParts(ordered), pending
}

// compactParts drops unfilled (nil) slots from the indexed parts slice.
// Slots are only nil when the consumer stopped listening before all
// tool executions were collected.
func compactParts(ordered []llm.Part) []llm.Part {
	parts := make([]llm.Part, 0, len(ordered))

	for _, p := range ordered {
		if p != nil {
			parts = append(parts, p)
		}
	}

	return parts
}

// buildPendingCall constructs a session.PendingToolCall record from a
// paused tool execution. ExpiresAt is computed from Await.Timeout if
// the tool didn't set it explicitly.
func buildPendingCall(req *llm.ToolRequestPart, exec tool.Execution, now time.Time) session.PendingToolCall {
	a := exec.Await

	pc := session.PendingToolCall{
		SchemaVersion: session.PendingToolCallSchemaVersion,
		ID:            req.ID,
		Name:          req.Name,
		Arguments:     req.Arguments,
		Reason:        string(a.Reason),
		Resume:        string(a.Resume),
		Message:       a.Message,
		Prompt:        a.Prompt,
		State:         a.State,
		CorrelationID: a.CorrelationID,
		CreatedAt:     now,
		LastOutput:    exec.Output,
		Metadata:      a.Metadata,
	}

	if a.ExpiresAt != nil {
		t := *a.ExpiresAt
		pc.ExpiresAt = &t
	} else if a.Timeout > 0 {
		t := now.Add(a.Timeout)
		pc.ExpiresAt = &t
	}

	if hash, err := tool.ArgumentsHash(req.Arguments); err == nil {
		pc.ArgumentsHash = hash
	}

	return pc
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

	// Execute the incomplete tools. Pending calls produced during
	// recovery are discarded here — the runner does not yet expose a
	// resume entry point that re-binds a pre-existing pause to the
	// recovered placeholder. Treat recovery purely as "fill in missing
	// tool responses so the next model call doesn't error out."
	toolDefs := a.config.tools.List()
	toolParts, _ := a.executeTools(ctx, inv, incomplete, toolDefs, makeEnvelope, yield)

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
