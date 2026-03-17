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

package retry

import (
	"log/slog"
	"time"
)

// DecisionFunc decides whether to retry an error.
// It receives the error and the attempt number (1-based, where 1 is the first retry).
// Return true to retry, false to propagate the error immediately.
type DecisionFunc func(err error, attempt int) bool

// config holds retry configuration.
type config struct {
	maxRetries    int
	initialDelay  time.Duration
	maxDelay      time.Duration
	retryDecision DecisionFunc
	logger        *slog.Logger
}

// Option configures the retry interceptor.
type Option func(*config)

// WithMaxRetries sets the maximum number of retry attempts.
// Default: 3.
func WithMaxRetries(n int) Option {
	return func(c *config) {
		if n >= 0 {
			c.maxRetries = n
		}
	}
}

// WithInitialDelay sets the base delay before the first retry.
// Subsequent retries use exponential backoff: initialDelay * 2^attempt.
// Default: 200ms.
func WithInitialDelay(d time.Duration) Option {
	return func(c *config) {
		if d > 0 {
			c.initialDelay = d
		}
	}
}

// WithMaxDelay sets the maximum delay cap.
// Default: 30s.
func WithMaxDelay(d time.Duration) Option {
	return func(c *config) {
		if d > 0 {
			c.maxDelay = d
		}
	}
}

// WithRetryDecision sets a custom function to decide whether to retry an error.
// When set, this overrides the default behavior of using llm.IsRetryable.
func WithRetryDecision(fn DecisionFunc) Option {
	return func(c *config) {
		c.retryDecision = fn
	}
}

// WithLogger sets an slog.Logger for retry events.
// When set, the interceptor logs retry attempts and outcomes.
func WithLogger(l *slog.Logger) Option {
	return func(c *config) {
		c.logger = l
	}
}

func defaultConfig() config {
	return config{
		maxRetries:   3,
		initialDelay: 200 * time.Millisecond,
		maxDelay:     30 * time.Second,
	}
}
