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
	"maps"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/tokens"
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

	// loader is non-nil whenever a tool registry is configured. It owns
	// everything about deferred tools: which definitions reach the model, the
	// manifest appended to the system prompt, and the tool_search calls the
	// model makes. With no deferred tool registered it changes nothing.
	loader *toolLoader
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

	llmAgent := &LLMAgent{config: cfg}

	if cfg.tools != nil {
		llmAgent.loader = newToolLoader(cfg.tools, cfg.toolLoading)
	}

	return llmAgent, nil
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
// Defaults to a single freeform text message. WithInputSchema replaces it, which
// is what a sub-agent wants: the delegation tool's parameters come from here, so a
// structured schema gives the caller named fields instead of one string to cram
// everything into.
func (a *LLMAgent) InputSchema() map[string]any {
	if a.config.inputSchema != nil {
		return a.config.inputSchema
	}

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
		// Observers see every non-nil event before the consumer. Turn events
		// are re-wrapped below with the turn context; lifecycle events use
		// the run context.
		consumerYield := guardYield(yield)
		yield = agent.ApplyEventObservers(ctx, inv, a.config.interceptors, consumerYield)

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
				// Turn events carry the interceptor-derived turn context.
				turnYield := agent.ApplyEventObservers(ctx, info.Inv, a.config.interceptors, consumerYield)

				return a.executeSingleTurn(ctx, info.Inv, makeEnvelope, turnYield)
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

// guardYield wraps yield so calls after it first returns false are dropped
// instead of panicking. The turn loop emits closing events on several paths
// after the consumer stops.
func guardYield(yield func(agent.Event, error) bool) func(agent.Event, error) bool {
	stopped := false

	return func(ev agent.Event, err error) bool {
		if stopped {
			return false
		}

		stopped = !yield(ev, err)

		return !stopped
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

	var toolDefs []llm.ToolDefinition
	if a.config.tools != nil {
		toolDefs = a.config.tools.List()
	}

	native := a.nativeToolSearch(toolDefs)

	// Resolve schemas and group instructions together before budgeting the request.
	var promptSection string

	if a.loader != nil {
		toolDefs, promptSection = a.prepareTools(toolDefs, sess, native)
	}

	// Build working message list with system prompt (not persisted)
	// This creates a transient view for the LLM request
	reqMessages, err := a.resolveSystemPrompt(ctx, inv, sess.Messages, promptSection)
	if err != nil {
		return "", fmt.Errorf("llmagent: system prompt: %w", err)
	}

	// One compaction check before every model call: this position sees an
	// oversized fresh user message and every burst of tool results. The
	// fixed cost (system prompt + schemas) rides on every request, so it is
	// part of the context budget the history must fit into. Interceptors may
	// still grow the request after this check; the reactive overflow path
	// below is the backstop, and its retry rebuilds the request and interceptor
	// chain so request-time transformations are applied again.
	fixedTokens := 0
	sysTokens := 0
	toolDefTokens := 0

	// The fixed cost also sets the lazy-loading admission line.
	if a.config.compaction != nil || a.loader != nil {
		sysTokens = tokens.Message(reqMessages[0])
		toolDefTokens = a.toolTokens(toolDefs, sess, native)
		fixedTokens = sysTokens + toolDefTokens
	}

	if a.config.compaction != nil {
		before := measureContext(sysTokens, toolDefTokens, sess.Messages)

		stats, fitErr := a.ensureFits(sess, fixedTokens)
		if stats.changed() {
			reqMessages = append([]llm.Message{reqMessages[0]}, sess.Messages...)

			report := compactionReport(agent.CompactionPhaseProactive, stats,
				before, measureContext(sysTokens, toolDefTokens, sess.Messages))

			if !yield(agent.CompactionEvent{
				Envelope: makeEnvelope(),
				Report:   report,
			}, nil) {
				return agent.FinishReasonInterrupted, nil
			}
		}

		if fitErr != nil {
			return "", fitErr
		}
	}

	resp, sentReq, err := a.generateAttempt(ctx, inv, reqMessages, toolDefs, native, makeEnvelope, yield)
	if err != nil && a.config.compaction != nil && errors.Is(err, llm.ErrContextOverflow) {
		// Reactive path: the provider rejected the request pre-flight, so
		// nothing was emitted. Force a strictly smaller request - hard
		// floors, at least 25% below the failed size - and retry once.
		before := measureContext(sysTokens, toolDefTokens, sess.Messages)

		stats, reduced := a.reduceAfterOverflow(sess, fixedTokens)
		if !reduced {
			return "", cannotFitError(stats.afterTokens, a.deriveContextBudget())
		}

		report := compactionReport(agent.CompactionPhaseReactive, stats,
			before, measureContext(sysTokens, toolDefTokens, sess.Messages))

		if !yield(agent.CompactionEvent{
			Envelope: makeEnvelope(),
			Report:   report,
		}, nil) {
			return agent.FinishReasonInterrupted, nil
		}

		retryMessages := append([]llm.Message{reqMessages[0]}, sess.Messages...)

		resp, sentReq, err = a.generateAttempt(ctx, inv, retryMessages, toolDefs, native, makeEnvelope, yield)
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

	// Native discovery may be followed by a real tool call in the same response.
	// Commit before MessageEvent, where runners persist the session.
	a.recordNativeLoads(sess, resp.Message)

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
		if native && resp.FinishReason == llm.FinishReasonToolCalls {
			// Anthropic pause_turn: replay the hosted search and let it continue.
			return "", nil
		}
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
	if native {
		fixedTokens = sysTokens + a.toolTokens(sentReq.Tools, sess, native)
	}
	countedRequest := fixedTokens + tokens.History(sess.Messages)
	resultCap := a.effectiveResultCap(countedRequest, len(toolReqs))

	toolParts := a.executeTools(ctx, inv, toolReqs, visibleTools(sentReq.Tools, sess, native), resultCap, a.schemaRoom(fixedTokens), makeEnvelope, yield)

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

// generateAttempt builds an independent request and interceptor chain for one
// provider attempt. Interceptors may mutate ModelCallInfo.Req while the chain
// is built, so an overflow retry must not reuse either object.
func (a *LLMAgent) generateAttempt(
	ctx context.Context,
	inv *agent.InvocationMetadata,
	messages []llm.Message,
	toolDefs []llm.ToolDefinition,
	native bool,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) (*llm.Response, *llm.Request, error) {
	if native {
		// Re-evaluate after either proactive compaction or an overflow retry.
		// The native prompt depends on the full catalog, not the loaded set,
		// so changing deferral after compaction leaves the prompt unchanged.
		toolDefs, _ = a.prepareTools(toolDefs, inv.Session(), native)
	}
	req := &llm.Request{
		Messages:   cloneMessages(messages),
		Tools:      cloneToolDefinitions(toolDefs),
		ToolSearch: native,
	}
	modelInfo := &agent.ModelCallInfo{
		InvocationMetadata: inv,
		Model:              a.config.model,
		Req:                req,
	}
	model := agent.ApplyModelInterceptors(ctx, modelInfo, a.config.model, a.config.interceptors)

	resp, err := a.generate(ctx, model, req, makeEnvelope, yield)

	return resp, req, err
}

func cloneMessages(messages []llm.Message) []llm.Message {
	if messages == nil {
		return nil
	}

	cloned := make([]llm.Message, len(messages))
	for i, message := range messages {
		cloned[i] = llm.CloneMessage(message)
	}

	return cloned
}

func cloneToolDefinitions(defs []llm.ToolDefinition) []llm.ToolDefinition {
	if defs == nil {
		return nil
	}

	cloned := make([]llm.ToolDefinition, len(defs))
	for i, def := range defs {
		cloned[i] = def

		cloned[i].Parameters = append(json.RawMessage(nil), def.Parameters...)
		if def.Metadata != nil {
			cloned[i].Metadata = make(map[string]any, len(def.Metadata))
			maps.Copy(cloned[i].Metadata, def.Metadata)
		}
	}

	return cloned
}

// resolveSystemPrompt produces a transient message list with the system
// prompt prepended. The system prompt is never persisted to the session.
//
// When a [SystemPromptProvider] is configured it is called while preparing
// the request, receiving both the request context and invocation metadata.
// Otherwise the static systemPrompt string from the config is used.
func (a *LLMAgent) resolveSystemPrompt(
	ctx context.Context,
	inv *agent.InvocationMetadata,
	messages []llm.Message,
	generated string,
) ([]llm.Message, error) {
	prompt := a.config.systemPrompt
	if a.config.systemPromptProvider != nil {
		p, err := a.config.systemPromptProvider(ctx, inv)
		if err != nil {
			return nil, err
		}

		prompt = p
	}

	// Append generated tool instructions after the configured prompt.
	if generated != "" {
		prompt = prompt + "\n\n" + generated
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

// executeTools runs a response's tool calls. Loader-owned calls run
// sequentially in request order so load admission is deterministic; the rest
// run on a worker pool. Results are returned in request order.
func (a *LLMAgent) executeTools(
	ctx context.Context,
	inv *agent.InvocationMetadata,
	toolReqs []*llm.ToolRequestPart,
	toolDefs []llm.ToolDefinition,
	resultCap int,
	schemaRoom int,
	makeEnvelope func() agent.EventEnvelope,
	yield func(agent.Event, error) bool,
) []llm.Part {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ctx = agent.ContextWithConversationID(ctx, session.ConversationID(inv.Session()))

	definitions := make(map[string]*llm.ToolDefinition, len(toolDefs))
	for i := range toolDefs {
		definitions[toolDefs[i].Name] = &toolDefs[i]
	}

	var batch *loadBatch
	if a.loader != nil {
		batch = a.loader.newBatch(toolDefs, schemaRoom)
	}

	var loaderCalls, ordinary []int

	for i, req := range toolReqs {
		if batch.owns(req) {
			loaderCalls = append(loaderCalls, i)
		} else {
			ordinary = append(ordinary, i)
		}
	}

	type toolResult struct {
		idx      int
		response *llm.ToolResponsePart
		loads    []string
		err      error
	}

	// Buffered so workers can finish after the consumer stops.
	results := make(chan toolResult, len(toolReqs))

	run := func(i int, batch *loadBatch) {
		req := toolReqs[i]

		resp, loads, err := a.executeTool(ctx, inv, req, definitions[req.Name], batch)
		results <- toolResult{idx: i, response: resp, loads: loads, err: err}
	}

	if len(loaderCalls) > 0 {
		go func() {
			for _, i := range loaderCalls {
				if ctx.Err() != nil {
					return
				}

				run(i, batch)
			}
		}()
	}

	jobs := make(chan int, len(ordinary))
	for _, i := range ordinary {
		jobs <- i
	}

	close(jobs)

	for range min(a.config.toolConcurrency, len(ordinary)) {
		go func() {
			for i := range jobs {
				if ctx.Err() != nil {
					return
				}

				run(i, nil)
			}
		}()
	}

	parts := make([]llm.Part, len(toolReqs))

	for range toolReqs {
		var result toolResult

		select {
		case <-ctx.Done():
			return interruptedToolParts(parts, toolReqs)
		case result = <-results:
		}

		resp := capToolResult(toolResponse(toolReqs[result.idx], result.response, result.err), resultCap)
		parts[result.idx] = resp

		if a.loader != nil {
			a.loader.commit(inv.Session(), result.loads)
		}

		if !yield(agent.ToolResponseEvent{Envelope: makeEnvelope(), Response: *resp}, nil) {
			return interruptedToolParts(parts, toolReqs)
		}
	}

	return parts
}

// executeTool runs one call through the interceptor chain. A non-nil batch
// marks a loader-owned call, answered from the batch and returning the loads
// it caused. A call an interceptor denies loads nothing.
func (a *LLMAgent) executeTool(
	ctx context.Context,
	inv *agent.InvocationMetadata,
	req *llm.ToolRequestPart,
	definition *llm.ToolDefinition,
	batch *loadBatch,
) (*llm.ToolResponsePart, []string, error) {
	var loads []string
	validateIdentity := func(info *agent.ToolCallInfo) error {
		if info == nil || info.Req == nil || info.Req.Name != req.Name || info.Req.ID != req.ID {
			return errors.New("llmagent: tool interceptors must preserve the request name and ID")
		}

		return nil
	}

	base := func(ctx context.Context, info *agent.ToolCallInfo) (*llm.ToolResponsePart, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if err := validateIdentity(info); err != nil {
			return nil, err
		}

		if batch != nil {
			if resp, added, owned := a.loader.resolve(batch, info.Req); owned {
				loads = append(loads, added...)

				return resp, nil
			}
		}

		return a.config.tools.Execute(ctx, info.Req)
	}
	executor := agent.ApplyToolInterceptors(a.config.interceptors, base)

	// Interceptors edit a copy: the original is the assistant message already
	// stored in the session.
	copied := *req
	copied.Arguments = append(json.RawMessage(nil), req.Arguments...)
	copied.Metadata = maps.Clone(req.Metadata)

	info := &agent.ToolCallInfo{Inv: inv, Req: &copied, Definition: definition}

	resp, err := executor(ctx, info)
	if identityErr := validateIdentity(info); identityErr != nil {
		return nil, loads, identityErr
	}

	if resp != nil && (resp.Name != req.Name || resp.ID != req.ID) {
		return nil, loads, errors.New("llmagent: tool interceptors must preserve the response name and ID")
	}

	return resp, loads, err
}

// toolResponse normalizes an execution outcome into the part the model reads:
// an error payload for a failed call, a placeholder for a nil response.
func toolResponse(req *llm.ToolRequestPart, resp *llm.ToolResponsePart, err error) *llm.ToolResponsePart {
	switch {
	case err != nil:
		payload, mErr := json.Marshal(map[string]string{"error": err.Error()})
		if mErr != nil {
			payload = []byte(`{"error":"tool error"}`)
		}

		return &llm.ToolResponsePart{ID: req.ID, Name: req.Name, Result: payload, IsError: true}
	case resp == nil:
		return errorResponse(req, "tool_error", "tool returned no response")
	default:
		return resp
	}
}

// interruptedToolParts pairs every request with a result so the saved
// transcript stays valid. Uncollected calls may have run; their outcome is
// unknown.
func interruptedToolParts(parts []llm.Part, reqs []*llm.ToolRequestPart) []llm.Part {
	for i, part := range parts {
		if part == nil {
			parts[i] = errorResponse(reqs[i], "interrupted",
				"Execution was interrupted before the result was collected. The outcome is unknown.")
		}
	}

	return parts
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

	// Execute the incomplete tools.
	toolDefs := a.config.tools.List()
	native := a.nativeToolSearch(toolDefs)

	// Recovery uses the persisted loaded set and the next request's fixed cost.
	var promptSection string

	if a.loader != nil {
		toolDefs, promptSection = a.prepareTools(toolDefs, sess, native)
	}

	fixedTokens := 0

	if a.config.compaction != nil || a.loader != nil {
		reqMessages, err := a.resolveSystemPrompt(ctx, inv, sess.Messages, promptSection)
		if err != nil {
			return fmt.Errorf("llmagent: system prompt for recovery budget: %w", err)
		}

		fixedTokens = tokens.Message(reqMessages[0]) + a.toolTokens(toolDefs, sess, native)
	}

	// Recovered results land in the unread frontier, which compaction can
	// never reduce - so the burst budget applies here exactly as in normal
	// execution. Include every fixed request cost because none of it can be
	// reclaimed on the following turn.
	countedRequest := fixedTokens + tokens.History(sess.Messages)
	resultCap := a.effectiveResultCap(countedRequest, len(incomplete))
	toolParts := a.executeTools(ctx, inv, incomplete, visibleTools(toolDefs, sess, native), resultCap, a.schemaRoom(fixedTokens), makeEnvelope, yield)

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
