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

type runConfig struct {
	attributes map[string]string
}

// RunOption configures a single Run call, as opposed to Option, which
// configures the long-lived Runner itself.
type RunOption func(*runConfig)

// WithAttribute adds one caller-asserted attribute to this run, such as a
// tenant or plan tier. Attributes are fixed for the run: the otel plugin
// exports every one on every span it opens, and in-process sub-agents
// inherit them. An empty key or value is ignored.
func WithAttribute(key, value string) RunOption {
	return func(c *runConfig) {
		if key == "" || value == "" {
			return
		}

		if c.attributes == nil {
			c.attributes = make(map[string]string)
		}

		c.attributes[key] = value
	}
}

// WithUserID records the end user this run is for; it is WithAttribute for
// agent.AttrUserID. An empty id asserts nothing.
func WithUserID(id string) RunOption {
	return WithAttribute(agent.AttrUserID, id)
}

// WithAttributes adds several attributes at once. See WithAttribute.
func WithAttributes(attrs map[string]string) RunOption {
	return func(c *runConfig) {
		for k, v := range attrs {
			WithAttribute(k, v)(c)
		}
	}
}
