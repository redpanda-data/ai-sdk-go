package main

import (
	"context"
	"log"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
)

// TurnTrackerInterceptor demonstrates TurnInterceptor.
// It logs each turn in the agentic loop and can implement early stopping conditions.
//
// Use cases:
// - Per-turn logging and timing
// - Custom early stopping logic
// - Turn-based rate limiting
// - Conversation quality checks
type TurnTrackerInterceptor struct {
	maxTurns int
}

// NewTurnTrackerInterceptor creates a turn tracking interceptor with optional max turns limit.
// Set maxTurns to 0 to disable custom early stopping.
func NewTurnTrackerInterceptor(maxTurns int) *TurnTrackerInterceptor {
	return &TurnTrackerInterceptor{
		maxTurns: maxTurns,
	}
}

// InterceptTurn implements agent.TurnInterceptor.
// It logs each turn and can implement custom early stopping conditions.
func (h *TurnTrackerInterceptor) InterceptTurn(ctx context.Context, inv *agent.InvocationMetadata, next agent.TurnNext) (agent.FinishReason, error) {
	start := time.Now()
	log.Printf("[TurnTracker] Turn %d started (Session: %s)", inv.Turn(), inv.Session().ID)

	reason, err := next(ctx, inv)

	duration := time.Since(start)
	if err != nil {
		log.Printf("[TurnTracker] Turn %d failed after %v: %v", inv.Turn(), duration, err)
		return reason, err
	}

	log.Printf("[TurnTracker] Turn %d completed in %v (reason: %s)", inv.Turn(), duration, reason)

	// Example: Implement custom early stopping
	if h.maxTurns > 0 && inv.Turn() >= h.maxTurns {
		log.Printf("[TurnTracker] Stopping early: reached max turns (%d)", h.maxTurns)
		return agent.FinishReasonMaxTurns, nil
	}

	return reason, nil
}
