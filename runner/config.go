package runner

import (
	"log/slog"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// runnerConfig holds the internal configuration for a Runner.
type runnerConfig struct {
	agent        agent.Agent
	sessionStore session.Store
	logger       *slog.Logger
}

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
