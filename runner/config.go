package runner

import (
	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/hooks"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// runnerConfig holds the internal configuration for a Runner.
type runnerConfig struct {
	agent        agent.Agent
	sessionStore session.Store
	hooks        []hooks.Hook
}

// validate checks that the runner configuration is valid.
func (c *runnerConfig) validate() error {
	if c.agent == nil {
		return agent.ErrNoAgent
	}

	if c.sessionStore == nil {
		return agent.ErrNoSessionStore
	}

	// Validate that all registered hooks implement at least one hook interface
	for _, h := range c.hooks {
		if !hooks.ImplementsAnyHook(h) {
			return agent.ErrInvalidHook
		}
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

// WithHook registers a hook for execution at various points during invocation.
//
// The hook will be called for any hook interfaces it implements (HookBeforeInvocation,
// HookAfterInvocation, etc.). Multiple hooks can be registered and will execute in
// registration order.
//
// # Example
//
//	type AuditHook struct {
//	    logger *slog.Logger
//	}
//
//	func (h *AuditHook) OnBeforeInvocation(ctx hooks.HookContext, msg llm.Message) error {
//	    h.logger.Info("invocation started", "session", ctx.SessionID())
//	    return nil
//	}
//
//	runner, _ := runner.New(agent, store,
//	    runner.WithHook(&AuditHook{logger: slog.Default()}))
func WithHook(hook hooks.Hook) Option {
	return func(cfg *runnerConfig) {
		cfg.hooks = append(cfg.hooks, hook)
	}
}
