package runner

import (
	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// runnerConfig holds the internal configuration for a Runner.
type runnerConfig struct {
	agent        agent.Agent
	sessionStore session.Store
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
// customization of runner behavior.
//
// # Example
//
//	runner, err := runner.New(agent, store)
type Option func(*runnerConfig)
