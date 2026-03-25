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

package anthropic_test

import (
	"testing"

	"github.com/redpanda-data/ai-sdk-go/agent/conformance"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// AnthropicAgentFixture implements conformance.Fixture for Anthropic provider.
type AnthropicAgentFixture struct {
	provider *anthropic.Provider
}

// NewAnthropicAgentFixture creates a new Anthropic agent test fixture.
func NewAnthropicAgentFixture(t *testing.T) *AnthropicAgentFixture {
	t.Helper()

	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	provider, err := anthropic.NewProvider(apiKey)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return &AnthropicAgentFixture{
		provider: provider,
	}
}

func (f *AnthropicAgentFixture) Name() string {
	return "Anthropic"
}

func (f *AnthropicAgentFixture) StandardAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(anthropictest.TestModelName)
	if err != nil {
		return nil, err
	}

	return llmagent.New(
		"test-agent",
		"You are a helpful assistant. When you have tools available, you must use them to answer questions rather than answering directly.",
		model,
		llmagent.WithTools(tools),
		llmagent.WithMaxTurns(10),
	)
}

func (f *AnthropicAgentFixture) ReasoningAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(anthropictest.TestReasoningModelName,
		anthropic.WithThinking(true),
		anthropic.WithMaxTokens(8192),
	)
	if err != nil {
		return nil, err
	}

	return llmagent.New(
		"reasoning-agent",
		"You are a helpful assistant with reasoning capabilities.",
		model,
		llmagent.WithTools(tools),
		llmagent.WithMaxTurns(10),
	)
}

// TestAnthropicAgentConformance_Integration runs the agent conformance test suite for Anthropic.
func TestAnthropicAgentConformance_Integration(t *testing.T) {
	t.Parallel()

	fixture := NewAnthropicAgentFixture(t)
	testsuite.Run(t, conformance.NewSuite(fixture))
}
