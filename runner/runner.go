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
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// PendingResolution supplies the final outcome for an externally-completing pending action.
type PendingResolution struct {
	// ID identifies the pending action to resolve.
	ID string

	// Result is the final successful tool output.
	Result json.RawMessage

	// Error describes a failed external completion. When non-empty, Result is ignored.
	Error string
}

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

		pendingActions, err := agent.GetPendingActions(sess)
		if err != nil {
			yield(nil, fmt.Errorf("%w: decode pending actions: %w", agent.ErrSessionLoad, err))
			return
		}

		for _, pendingAction := range pendingActions {
			if pendingAction.Kind == "external_result" {
				yield(nil, fmt.Errorf("%w: unresolved external continuation %q in session %q",
					agent.ErrPendingResolutionRequired, pendingAction.ID, sessionID))
				return
			}
		}

		if len(pendingActions) > 0 {
			agent.SetPendingActions(sess, nil)
		}

		// 2. Add user message to session
		sess.Messages = append(sess.Messages, userMessage)

		// 3. Execute agent and forward events
		r.runAgent(ctx, sess, yield)
	}
}

// ResolvePending resolves externally-completing pending actions and continues execution.
func (r *Runner) ResolvePending(
	ctx context.Context,
	_ string, // UserID will be used in the future.
	sessionID string,
	resolutions []PendingResolution,
) iter.Seq2[agent.Event, error] {
	return func(yield func(agent.Event, error) bool) {
		sess, err := r.config.sessionStore.Load(ctx, sessionID)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionLoad, err))
			return
		}

		pendingActions, err := agent.GetPendingActions(sess)
		if err != nil {
			yield(nil, fmt.Errorf("%w: decode pending actions: %w", agent.ErrSessionLoad, err))
			return
		}

		externalActions := make(map[string]agent.PendingAction)
		for _, pendingAction := range pendingActions {
			switch pendingAction.Kind {
			case "external_result":
				externalActions[pendingAction.ID] = pendingAction
			case "user_input":
				yield(nil, fmt.Errorf("%w: session %q is waiting for user input via Run()",
					agent.ErrPendingResolutionRequired, sessionID))
				return
			}
		}

		if len(externalActions) == 0 {
			yield(nil, fmt.Errorf("%w: session %q has no external pending actions", agent.ErrPendingActionNotFound, sessionID))
			return
		}

		if len(resolutions) != len(externalActions) {
			yield(nil, fmt.Errorf("%w: expected %d pending resolutions, got %d",
				agent.ErrPendingResolutionRequired, len(externalActions), len(resolutions)))
			return
		}

		resolved := make(map[string]struct{}, len(resolutions))
		for _, resolution := range resolutions {
			pendingAction, ok := externalActions[resolution.ID]
			if !ok {
				yield(nil, fmt.Errorf("%w: %q", agent.ErrPendingActionNotFound, resolution.ID))
				return
			}

			if _, seen := resolved[resolution.ID]; seen {
				yield(nil, fmt.Errorf("%w: duplicate resolution for %q", agent.ErrPendingResolutionRequired, resolution.ID))
				return
			}
			resolved[resolution.ID] = struct{}{}

			toolResponse := llm.ToolResponse{
				ID:   pendingAction.ToolCallID,
				Name: pendingAction.ToolName,
			}
			if resolution.Error != "" {
				toolResponse.Error = resolution.Error
			} else {
				toolResponse.Result = resolution.Result
			}

			if !replaceToolResponse(sess, toolResponse) {
				yield(nil, fmt.Errorf("%w: tool response for call %q not found",
					agent.ErrPendingActionNotFound, pendingAction.ToolCallID))
				return
			}
		}

		agent.SetPendingActions(sess, nil)
		r.runAgent(ctx, sess, yield)
	}
}

// runAgent is the shared agent execution and event forwarding logic.
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
			// Only yield error if consumer hasn't explicitly stopped iteration.
			// If consumer broke out of their for loop (yield returned false),
			// calling yield again would panic.
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
			// Forward error
			if !yield(nil, err) {
				consumerStopped = true
				return
			}

			continue
		}

		// Save session after each assistant message (incremental persistence)
		// Note: Agent already appended the message to sess.Messages, we just save it
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

		// Exit after completion event - consumer is still active here,
		// defer can still yield if needed
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

func replaceToolResponse(sess *session.State, response llm.ToolResponse) bool {
	if sess == nil {
		return false
	}

	for i := len(sess.Messages) - 1; i >= 0; i-- {
		msg := &sess.Messages[i]
		if msg.Role != llm.RoleUser {
			continue
		}

		for j := range msg.Content {
			part := msg.Content[j]
			if !part.IsToolResponse() || part.ToolResponse == nil {
				continue
			}
			if part.ToolResponse.ID != response.ID {
				continue
			}

			msg.Content[j] = llm.NewToolResponsePart(&response)
			return true
		}
	}

	return false
}
