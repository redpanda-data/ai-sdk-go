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

package runner

import (
	"context"
	"log/slog"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// runnerConfig holds the internal configuration for a Runner.
type runnerConfig struct {
	agent        agent.Agent
	sessionStore session.Store
	logger       *slog.Logger
	authorize    ResumeAuthorizer
}

// ResumeAuthorizer is an optional hook applied after loading pending
// state and before mutating the session on Resume / Progress / Cancel /
// message-resume. Returning a non-nil error aborts the operation and
// leaves the session unchanged. The hook receives enough context to make
// per-operation authorization decisions (actor, session, pending call,
// submitted result).
type ResumeAuthorizer func(ctx context.Context, info ResumeInfo) error

// ResumeInfo is the input to ResumeAuthorizer.
type ResumeInfo struct {
	UserID      string
	SessionID   string
	PendingCall session.PendingToolCall
	Result      Result
	Operation   ResumeOperation
}

// ResumeOperation tells the authorizer which entry point triggered the
// callback, so a single policy can reason about "approve this" vs.
// "send progress" vs. "cancel" without three separate hooks.
type ResumeOperation string

const (
	ResumeOperationResume   ResumeOperation = "resume"
	ResumeOperationProgress ResumeOperation = "progress"
	ResumeOperationCancel   ResumeOperation = "cancel"
	ResumeOperationMessage  ResumeOperation = "message"
)

// validate checks that the runner configuration is valid.
func (c *runnerConfig) validate() error {
	if c.agent == nil {
		return agent.ErrNoAgent
	}

	if c.sessionStore == nil {
		return agent.ErrNoSessionStore
	}

	return nil
}

// Option configures a Runner.
//
// Options are applied during Runner construction via New(). They allow
// customization of runner behavior such as middleware, hooks, retries,
// and observability.
//
// # Example
//
//	runner, err := runner.New(agent, store,
//	    runner.WithMiddleware(loggingMiddleware),
//	    runner.WithMaxRetries(3),
//	)
type Option func(*runnerConfig)

// WithLogger sets a custom logger for the runner.
// Defaults to slog.Default().
func WithLogger(logger *slog.Logger) Option {
	return func(c *runnerConfig) {
		c.logger = logger
	}
}

// WithResumeAuthorizer registers a callback that runs before any
// session-mutating Resume / Progress / Cancel / message-resume.
// Returning a non-nil error aborts the operation without touching the
// session.
func WithResumeAuthorizer(authorize ResumeAuthorizer) Option {
	return func(c *runnerConfig) {
		c.authorize = authorize
	}
}
