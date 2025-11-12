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

	"github.com/redpanda-data/ai-sdk-go/agent"
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
//   - ToolCallEvent, ToolResultEvent (tool execution)
//   - ErrorEvent (recoverable errors)
//   - InvocationEndEvent (terminal event)
//
// # Example
//
//	for evt, err := range runner.Run(ctx, "session-123", userMsg) {
//	    if err != nil {
//	        log.Printf("Error: %v", err)
//	        continue
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

		// 3. Create invocation context
		invCtx := agent.NewInvocationContext(ctx, sess)

		// 4. Execute agent and forward events
		var completed bool

		for evt, err := range r.config.agent.Run(invCtx) {
			if err != nil {
				// Forward error
				if !yield(nil, err) {
					return
				}

				continue
			}

			// Check if this is the terminal event
			if _, ok := evt.(agent.InvocationEndEvent); ok {
				completed = true
			}

			// Forward event to caller
			if !yield(evt, nil) {
				return
			}

			// If completed, save session and exit
			if completed {
				break
			}
		}

		// 5. Save session
		//nolint:contextcheck // invCtx embeds the original ctx and maintains its cancellation
		if err := r.config.sessionStore.Save(invCtx, sess); err != nil {
			yield(nil, fmt.Errorf("%w: %w", agent.ErrSessionSave, err))
			return
		}
	}
}

// RunBlocking is a convenience wrapper that blocks until execution completes.
//
// This method consumes the event stream from Run() and returns the final
// usage and finish reason. Any errors encountered during execution are returned.
//
// Note: This discards all intermediate events (status, messages, tools).
// Use Run() directly if you need real-time progress updates.
//
// # Example
//
//	finishReason, usage, err := runner.RunBlocking(ctx, "session-123", userMsg)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Completed: %s, tokens: %d\n", finishReason, usage.TotalTokens)
func (r *Runner) RunBlocking(
	ctx context.Context,
	userID string,
	sessionID string,
	userMessage llm.Message,
) (agent.FinishReason, *llm.TokenUsage, error) {
	var finishReason agent.FinishReason
	var usage *llm.TokenUsage
	var lastErr error

	for evt, err := range r.Run(ctx, userID, sessionID, userMessage) {
		if err != nil {
			lastErr = err
			continue
		}

		// Extract final result from InvocationEndEvent
		if endEvt, ok := evt.(agent.InvocationEndEvent); ok {
			finishReason = endEvt.FinishReason
			usage = endEvt.Usage
		}
	}

	if lastErr != nil {
		return "", nil, lastErr
	}

	if finishReason == "" {
		return "", nil, errors.New("no completion event received")
	}

	return finishReason, usage, nil
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
