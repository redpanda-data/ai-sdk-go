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

// Package runner provides orchestration for agent execution with session management.
//
// Runner handles session loading/saving and forwards events from agent execution.
// This separation enables independent evolution of infrastructure concerns
// (middleware, hooks, retries) without impacting the core agent interface.
package runner

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// Runner orchestrates agent execution with session management.
//
// The Runner is the entry point for executing agents. It handles:
//   - Session loading and persistence
//   - InvocationMetadata creation
//   - Agent execution coordination
//   - Event streaming
//
// Design: The Runner:
//  1. Loads (or creates) the session from the store
//  2. Adds the user message to the session
//  3. Creates an InvocationMetadata with the session reference
//  4. Calls Agent.Run(ctx, inv) and forwards events
//  5. Saves the updated session when complete
//
// The Runner is stateless - all state lives in the Session. Multiple
// runners can operate on the same session sequentially.
//
// Event Streaming:
// The Runner forwards events from the agent without modification.
// This enables real-time progress updates and protocol adapter integration.
type Runner struct {
	config *runnerConfig
}

// New creates a new runner with the given agent and session store.
//
// The agent and session store are required. Optional configuration can be
// provided via Option functions.
//
// # Example
//
//	runner, err := runner.New(myAgent, sessionStore)
//	if err != nil {
//	    log.Fatal(err)
//	}
func New(ag agent.Agent, sessionStore session.Store, opts ...Option) (*Runner, error) {
	cfg := &runnerConfig{
		agent:        ag,
		sessionStore: sessionStore,
		logger:       slog.Default(),
	}

	// Apply options
	for _, opt := range opts {
		opt(cfg)
	}

	// Validate configuration
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return &Runner{
		config: cfg,
	}, nil
}

// Run executes the agent with a user message, yielding events.
//
// The user message is added to the session, then the agent is invoked.
// Events from the agent are forwarded directly to the caller. When
// execution completes (InvocationEndEvent), the session is saved.
//
// # Event Flow
//
// Events are forwarded from Agent.Run() without modification:
//   - StatusEvent (turn started, model call, tool exec, etc.)
//   - MessageEvent (assistant responses)
//   - AssistantDeltaEvent (streaming tokens, if supported)
//   - ToolRequestEvent, ToolResponseEvent (tool execution)
//   - ErrorEvent (recoverable errors)
//   - InvocationEndEvent (terminal event)
//
// # Error Handling
//
// Run uses iter.Seq2[Event, error] following the principle:
// "errors in events are data, errors in iterators are control flow"
//
// Terminal Errors - yield(nil, error):
//
//	Runner-level failures that prevent execution (control flow)
//	Examples: ErrSessionLoad (session store unreachable), ErrSessionSave (persistence failed)
//
// Forwarded Errors - yield(nil, error):
//
//	Terminal errors from agent are forwarded upstream for handling
//	Examples: All agent terminal errors (ErrToolRegistry, auth failures, etc.)
//
// Observable Errors - ErrorEvent (forwarded from agent):
//
//	Application-level errors visible to users (data)
//	Examples: rate limits, content filters, individual tool failures
//
// Consumer pattern:
//
//	for evt, err := range runner.Run(ctx, userID, sessionID, userMsg) {
//	    if err != nil {
//	        // CONTROL FLOW: Fatal error, system can't continue
//	        return
//	    }
//	    switch e := evt.(type) {
//	    case agent.ErrorEvent:
//	        // DATA: Observable error for logging/display
//	    case agent.InvocationEndEvent:
//	        // Completion (check FinishReason)
//	    }
//	}
//
// # Example
//
//	for evt, err := range runner.Run(ctx, "user-123", "session-123", userMsg) {
//	    if err != nil {
//	        log.Printf("Error: %v", err)
//	        return
//	    }
//
//	    switch e := evt.(type) {
//	    case agent.StatusEvent:
//	        fmt.Printf("Status: %s - %s\n", e.Stage, e.Details)
//	    case agent.MessageEvent:
//	        fmt.Printf("Assistant: %s\n", e.Response.Message.Content)
//	    case agent.InvocationEndEvent:
//	        fmt.Printf("Finished: %s (usage: %+v)\n", e.FinishReason, e.Usage)
//	    }
//	}
func (r *Runner) Run(
	ctx context.Context,
	_ string, // UserID will be used in the future.
	sessionID string,
	userMessage llm.Message,
) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		// 1. Load or create session
		sess, err := r.loadOrCreateSession(ctx, sessionID)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionLoad, err))
			return
		}

		// 2. Add user message to session
		sess.Messages = append(sess.Messages, userMessage)

		// 3. Execute agent and forward events
		r.runAgent(ctx, sess, yield)
	}
}

// Resume continues a paused invocation by providing results for pending tools.
//
// This is used after an invocation ended with FinishReasonInputRequired.
// The caller provides tool results for the tools listed in
// InvocationEndEvent.InputRequiredToolIDs.
//
// The provided tool results are added to the session as a user message
// containing ToolResponse parts, then the agent resumes from where it
// left off. The LLM sees the full history: original tool call, the initial
// pending result, and the final result provided here.
//
// # Example
//
//	// After receiving InvocationEndEvent with FinishReason "input_required"
//	// and InputRequiredToolIDs: ["call_abc123"]
//	for evt, err := range runner.Resume(ctx, userID, sessionID,
//	    []llm.ToolResponse{{
//	        ID:     "call_abc123",
//	        Name:   "deploy",
//	        Result: json.RawMessage(`{"url": "https://staging.example.com"}`),
//	    }},
//	) {
//	    // ... handle events as with Run() ...
//	}
func (r *Runner) Resume(
	ctx context.Context,
	_ string, // UserID will be used in the future.
	sessionID string,
	toolResults []llm.ToolResponse,
) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		// 1. Load session (must exist — Resume is only valid for existing sessions)
		sess, err := r.config.sessionStore.Load(ctx, sessionID)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionLoad, err))
			return
		}

		// 2. Add tool results as user message with ToolResponse parts
		parts := make([]*llm.Part, 0, len(toolResults))
		for i := range toolResults {
			parts = append(parts, llm.NewToolResponsePart(&toolResults[i]))
		}

		sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, parts...))

		// 3. Execute agent and forward events
		r.runAgent(ctx, sess, yield)
	}
}

// runAgent is the shared agent execution and event forwarding logic
// used by both Run() and Resume().
func (r *Runner) runAgent(
	ctx context.Context,
	sess *session.State,
	yield func(agent.Event, error) bool,
) {
	// Create invocation metadata with agent snapshot
	inv := agent.NewInvocationMetadata(sess, r.config.agent.Info())

	// Track whether the consumer stopped iteration (yield returned false).
	// When yield returns false, we must not call it again or Go panics with
	// "range function continued iteration after function for loop body returned false".
	consumerStopped := false

	// Save session on exit (handles normal completion, cancellation, errors)
	defer func() {
		if err := r.config.sessionStore.Save(ctx, sess); err != nil {
			if !consumerStopped {
				yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionSave, err))
			} else {
				r.config.logger.Error("session save failed after consumer stopped",
					"sessionID", sess.ID,
					"error", err)
			}
		}
	}()

	// Execute agent and forward events
	for evt, err := range r.config.agent.Run(ctx, inv) {
		if err != nil {
			if !yield(nil, err) {
				consumerStopped = true
				return
			}

			continue
		}

		// Save session after each assistant message (incremental persistence)
		if _, ok := evt.(agent.MessageEvent); ok {
			if err := r.config.sessionStore.Save(ctx, sess); err != nil {
				yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionSave, err))
				return
			}
		}

		// Forward event to caller
		if !yield(evt, nil) {
			consumerStopped = true
			return
		}

		// Exit after completion event
		if _, ok := evt.(agent.InvocationEndEvent); ok {
			return
		}
	}
}

// loadOrCreateSession loads an existing session or creates a new one.
func (r *Runner) loadOrCreateSession(ctx context.Context, sessionID string) (*session.State, error) {
	sess, err := r.config.sessionStore.Load(ctx, sessionID)
	if err == nil {
		return sess, nil
	}

	if errors.Is(err, session.ErrNotFound) {
		// Create new session
		return &session.State{
			ID:       sessionID,
			Messages: []llm.Message{},
			Metadata: make(map[string]any),
		}, nil
	}

	return nil, err
}
