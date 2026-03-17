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

package google_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/agent/conformance"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/google/googletest"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// GoogleAgentFixture implements conformance.Fixture for Google Gemini provider.
type GoogleAgentFixture struct {
	provider *google.Provider
}

// NewGoogleAgentFixture creates a new Google Gemini agent test fixture.
func NewGoogleAgentFixture(t *testing.T) *GoogleAgentFixture {
	t.Helper()

	apiKey := googletest.GetAPIKeyOrSkipTest(t)

	provider, err := google.NewProvider(t.Context(), apiKey)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return &GoogleAgentFixture{
		provider: provider,
	}
}

func (f *GoogleAgentFixture) Name() string {
	return "Google"
}

func (f *GoogleAgentFixture) StandardAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(googletest.TestModelName)
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

func (f *GoogleAgentFixture) ReasoningAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(googletest.TestReasoningModelName,
		google.WithThinking(true),
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

// TestGoogleAgentConformance runs the agent conformance test suite for Google Gemini.
func TestGoogleAgentConformance(t *testing.T) {
	t.Parallel()

	fixture := NewGoogleAgentFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}
