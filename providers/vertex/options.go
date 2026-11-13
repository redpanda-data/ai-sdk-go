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

package vertex

import "github.com/redpanda-data/ai-sdk-go/llm"

// Option configures a model built by Provider.NewModel.
type Option func(*modelConfig) error

type modelConfig struct {
	reasoningEffort *llm.ReasoningEffort
	thinkingBudget  *int64
}

// WithReasoningEffort sets the reasoning effort. The model's offering
// lists the efforts it accepts, and any other is refused.
func WithReasoningEffort(effort ReasoningEffort) Option {
	return func(c *modelConfig) error {
		c.reasoningEffort = &effort

		return nil
	}
}

// WithThinkingBudget sets an explicit thinking budget in tokens for a Claude
// model that takes a budget rather than an effort, such as Claude Haiku 4.5.
// It is refused for Gemini and for Claude models without a budget.
func WithThinkingBudget(tokens int64) Option {
	return func(c *modelConfig) error {
		c.thinkingBudget = &tokens

		return nil
	}
}
