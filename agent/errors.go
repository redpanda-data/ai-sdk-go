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

	// ErrPendingResolutionRequired indicates the session is paused on an external continuation
	// and cannot accept a plain user message until the pending action is resolved.
	ErrPendingResolutionRequired = errors.New("agent: pending continuation must be resolved before running")

	// ErrPendingActionNotFound indicates a requested pending action ID does not exist in the session.
	ErrPendingActionNotFound = errors.New("agent: pending action not found")

	// ────────────────────────────────────────────────────────────────────────────────
	// Agent execution errors
	// These are returned during agent execution (model generation, tool execution).
	// ────────────────────────────────────────────────────────────────────────────────.

	// ErrModelGeneration indicates the LLM model failed to generate a response.
	// This could be due to API errors, rate limits, invalid input, or model errors.
	ErrModelGeneration = errors.New("agent: model generation failed")

	// ErrToolRegistry indicates tools were requested but no tool registry is configured.
	// This is a configuration error, not a transient error.
	ErrToolRegistry = errors.New("agent: no tool registry configured")

	// ErrValidation indicates input validation failed.
	// This is a client error, not a transient error.
	ErrValidation = errors.New("agent: validation failed")

	// ErrMaxTurnsReached is returned when an agent hits its turn limit.
	// This is not an error per se, but a normal termination condition.
	ErrMaxTurnsReached = errors.New("agent: maximum turns reached")

	// ErrInterrupted is returned when execution is canceled via context.
	// This is not an error per se, but indicates the caller canceled the operation.
	ErrInterrupted = errors.New("agent: execution canceled")
)
