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
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/hooks"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// Runner orchestrates agent execution with session management.
//
// The Runner is the entry point for executing agents. It handles:
//   - Session loading and persistence
//   - InvocationContext creation
//   - Agent execution coordination
//   - Event streaming
//
// Design: The Runner:
//  1. Loads (or creates) the session from the store
//  2. Adds the user message to the session
//  3. Creates an InvocationContext with the session reference
//  4. Calls Agent.Run(invCtx) and forwards events
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

		// 2. Create invocation context (needed for hook context)
		invCtx := agent.NewInvocationContext(ctx, sess)

		// 3. Execute BeforeInvocation hooks
		if len(r.config.hooks) > 0 {
			//nolint:contextcheck // hookCtx wraps invCtx which properly propagates the parent context
			hookCtx := hooks.NewHookContext(
				invCtx,
				invCtx.InvocationID(),
				sess.ID,
				0, // turn starts at 0
				time.Now().UTC(),
				sess,
			)

			if err := r.executeBeforeInvocationHooks(hookCtx, userMessage); err != nil {
				yield(nil, fmt.Errorf("before invocation hook: %w", err))
				return
			}
		}

		// 4. Add user message to session
		sess.Messages = append(sess.Messages, userMessage)

		// 5. Execute agent and collect events
		var completed bool
		var finalMessage *llm.Message
		var allEvents []agent.Event
		var invocationErr error

		//nolint:contextcheck // invCtx properly wraps the parent context and propagates cancellation
		for evt, err := range r.config.agent.Run(invCtx) {
			if err != nil {
				invocationErr = err
				// Forward error
				if !yield(nil, err) {
					return
				}

				continue
			}

			// Collect events for AfterInvocation hooks
			allEvents = append(allEvents, evt)

			// Track final message
			if msgEvt, ok := evt.(agent.MessageEvent); ok {
				finalMessage = &msgEvt.Response.Message
			}

			// Check if this is the terminal event
			var endEvt agent.InvocationEndEvent
			if evt != nil {
				endEvt, completed = evt.(agent.InvocationEndEvent)
			}

			// Forward event to caller
			if !yield(evt, nil) {
				return
			}

			// If completed, execute AfterInvocation hooks
			if completed && len(r.config.hooks) > 0 {
				hookCtx := hooks.NewHookContext(
					invCtx,
					invCtx.InvocationID(),
					sess.ID,
					invCtx.Turn(),
					time.Now().UTC(),
					sess,
				)

				result := hooks.InvocationResult{
					FinishReason: endEvt.FinishReason,
					FinalMessage: finalMessage,
					TotalUsage:   invCtx.TotalUsage(),
					Events:       allEvents,
					Error:        invocationErr,
				}

				if err := r.executeAfterInvocationHooks(hookCtx, result); err != nil {
					yield(nil, fmt.Errorf("after invocation hook: %w", err))
					return
				}

				break
			}
		}

		// 6. Save session
		//nolint:contextcheck // invCtx embeds the original ctx and maintains its cancellation
		if err := r.config.sessionStore.Save(invCtx, sess); err != nil {
			yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionSave, err))
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
