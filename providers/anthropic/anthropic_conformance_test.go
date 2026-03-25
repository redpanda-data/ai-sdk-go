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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
)

// AnthropicFixture implements the conformance.Fixture interface for Anthropic provider.
type AnthropicFixture struct {
	provider       *anthropic.Provider
	standardModel  llm.Model
	reasoningModel llm.Model
}

// NewAnthropicFixture creates a new Anthropic test fixture.
func NewAnthropicFixture(t *testing.T) *AnthropicFixture {
	t.Helper()

	// Check for API key (skips test if not set)
	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	// Create provider
	provider, err := anthropic.NewProvider(apiKey, anthropic.WithTimeout(time.Minute*2))
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Create standard model (Sonnet 4.5)
	standardModel, err := provider.NewModel(anthropictest.TestModelName)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	// Create reasoning model (Opus 4.1 with thinking enabled)
	// Set explicit MaxTokens for reasoning to allow adequate space for complex reasoning
	reasoningModel, err := provider.NewModel(anthropictest.TestReasoningModelName,
		anthropic.WithThinking(true),
		anthropic.WithMaxTokens(8192),
	)
	if err != nil {
		// Reasoning model is optional, just log but don't skip
		t.Logf("Failed to create reasoning model: %v", err)
	}

	var wrappedReasoning llm.Model
	if reasoningModel != nil {
		wrappedReasoning = retry.WrapModel(reasoningModel)
	}

	return &AnthropicFixture{
		provider:       provider,
		standardModel:  retry.WrapModel(standardModel),
		reasoningModel: wrappedReasoning,
	}
}

func (f *AnthropicFixture) Name() string {
	return "Anthropic"
}

func (f *AnthropicFixture) StandardModel() llm.Model {
	return f.standardModel
}

func (f *AnthropicFixture) ReasoningModel() llm.Model {
	return f.reasoningModel
}

func (f *AnthropicFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *AnthropicFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestAnthropicConformance_Integration runs the generic conformance test suite for the Anthropic provider.
//
//nolint:paralleltest // Test suite manages its own lifecycle
func TestAnthropicConformance_Integration(t *testing.T) {
	fixture := NewAnthropicFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}

// TestAnthropicAdaptiveThinking_Integration tests adaptive thinking with Sonnet 4.6.
func TestAnthropicAdaptiveThinking_Integration(t *testing.T) {
	t.Parallel()

	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	provider, err := anthropic.NewProvider(apiKey, anthropic.WithTimeout(time.Minute*2))
	require.NoError(t, err)

	// Create Sonnet 4.6 with adaptive thinking enabled
	model, err := provider.NewModel(anthropictest.TestAdaptiveModelName,
		anthropic.WithThinking(true),
		anthropic.WithMaxTokens(8192),
	)
	require.NoError(t, err)

	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("Explain the difference between a mutex and a semaphore in two sentences."),
				},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute*2)
	defer cancel()

	resp, err := model.Generate(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	var hasText, hasReasoning bool

	for _, part := range resp.Message.Content {
		switch {
		case part.IsText():
			hasText = true

			assert.NotEmpty(t, part.Text)
		case part.IsReasoning():
			hasReasoning = true
		}
	}

	assert.True(t, hasText, "expected text content in response")
	assert.True(t, hasReasoning, "expected reasoning content from adaptive thinking")
	require.NotNil(t, resp.Usage)
	assert.Positive(t, resp.Usage.TotalTokens, "expected non-zero token usage")
}
