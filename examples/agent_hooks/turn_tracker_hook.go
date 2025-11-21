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
func (h *TurnTrackerInterceptor) InterceptTurn(ctx context.Context, next agent.TurnNext) (agent.FinishReason, error) {
	// Note: In a real implementation, you might extract turn number from ctx
	// For this example, we'll just log the turn
	start := time.Now()
	log.Printf("[TurnTracker] Turn started")

	reason, err := next(ctx)

	duration := time.Since(start)
	if err != nil {
		log.Printf("[TurnTracker] Turn failed after %v: %v", duration, err)
		return reason, err
	}

	log.Printf("[TurnTracker] Turn completed in %v (reason: %s)", duration, reason)

	// Example: Implement custom early stopping
	// In a real scenario, you might check turn count from context
	// if turnNum >= h.maxTurns {
	//     log.Printf("[TurnTracker] Stopping early: reached max turns (%d)", h.maxTurns)
	//     return agent.FinishReasonMaxTurns, nil
	// }

	return reason, nil
}
