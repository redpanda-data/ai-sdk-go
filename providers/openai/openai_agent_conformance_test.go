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

package openai_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/agent/conformance"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// OpenAIAgentFixture implements conformance.Fixture for OpenAI provider.
type OpenAIAgentFixture struct {
	provider *openai.Provider
}

// NewOpenAIAgentFixture creates a new OpenAI agent test fixture.
func NewOpenAIAgentFixture(t *testing.T) *OpenAIAgentFixture {
	t.Helper()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	provider, err := openai.NewProvider(apiKey)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return &OpenAIAgentFixture{
		provider: provider,
	}
}

func (f *OpenAIAgentFixture) Name() string {
	return "OpenAI"
}

func (f *OpenAIAgentFixture) StandardAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(openaitest.TestModelName)
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

func (f *OpenAIAgentFixture) ReasoningAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(openaitest.TestReasoningModelName,
		openai.WithReasoningEffort(openai.ReasoningEffortHigh),
		openai.WithReasoningSummary(openai.ReasoningSummaryDetailed),
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

// TestOpenAIAgentConformance_Integration runs the agent conformance test suite for OpenAI.
func TestOpenAIAgentConformance_Integration(t *testing.T) {
	t.Parallel()

	fixture := NewOpenAIAgentFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}
