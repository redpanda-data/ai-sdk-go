package agent

import "errors"

// Sentinel errors for common agent failure modes.
// These follow Go's standard error handling patterns and can be used
// with errors.Is() for type-safe error checking.
//
// Example usage:
//
//	if errors.Is(err, agent.ErrSessionLoad) {
//	    // Handle session loading error (potentially retry)
//	}
var (
	// ────────────────────────────────────────────────────────────────────────────────
	// Runner-level errors
	// These are returned by Runner operations (session management, configuration).
	// ────────────────────────────────────────────────────────────────────────────────.

	// ErrNoAgent is returned when attempting to create a runner without an agent.
	// This is a configuration error, not a transient error.
	ErrNoAgent = errors.New("agent: no agent provided")

	// ErrNoSessionStore is returned when attempting to create a runner without a session store.
	// This is a configuration error, not a transient error.
	ErrNoSessionStore = errors.New("agent: no session store provided")

	// ErrSessionLoad indicates a failure to load session state from storage.
	// This may be a transient error (network, database connection) that can be retried.
	ErrSessionLoad = errors.New("agent: failed to load session")

	// ErrSessionSave indicates a failure to save session state to storage.
	// This may be a transient error (network, database connection) that can be retried.
	ErrSessionSave = errors.New("agent: failed to save session")

	// ────────────────────────────────────────────────────────────────────────────────
	// Agent execution errors
	// These are returned during agent execution (model generation, tool execution).
	// ────────────────────────────────────────────────────────────────────────────────.

	// ErrModelGeneration indicates the LLM model failed to generate a response.
	// This could be due to API errors, rate limits, invalid input, or model errors.
	ErrModelGeneration = errors.New("agent: model generation failed")

	// ErrToolExecution indicates a tool execution failed systemically.
	// Individual tool errors are captured in ToolResponse.Error, but this
	// sentinel is used for systemic tool execution failures (e.g., context cancellation
	// during tool execution, or failure to communicate with tool registry).
	ErrToolExecution = errors.New("agent: tool execution failed")

	// ErrToolRegistry indicates tools were requested but no tool registry is configured.
	// This is a configuration error, not a transient error.
	ErrToolRegistry = errors.New("agent: no tool registry configured")

	// ErrValidation indicates input validation failed.
	// This is a client error, not a transient error.
	ErrValidation = errors.New("agent: validation failed")

	// ErrMaxTurnsReached is returned when an agent hits its turn limit.
	// This is not an error per se, but a normal termination condition.
	ErrMaxTurnsReached = errors.New("agent: maximum turns reached")

	// ErrCanceled is returned when execution is canceled via context.
	// This is not an error per se, but indicates the caller canceled the operation.
	ErrCanceled = errors.New("agent: execution canceled")
)
