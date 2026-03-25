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
	"time"

	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
)

// OpenAIFixture implements the conformance.Fixture interface for OpenAI provider.
type OpenAIFixture struct {
	provider          *openai.Provider
	reasoningProvider *openai.Provider
}

// NewOpenAIFixture creates a new OpenAI test fixture.
func NewOpenAIFixture(t *testing.T) *OpenAIFixture {
	t.Helper()

	// Check for API key (skips test if not set)
	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	// Create provider with standard timeout for regular models
	provider, err := openai.NewProvider(apiKey, openai.WithTimeout(time.Minute*2))
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Create reasoning model with extended timeout since reasoning can take longer
	reasoningProvider, err := openai.NewProvider(apiKey, openai.WithTimeout(time.Minute*5))
	if err != nil {
		t.Fatalf("Failed to create reasoning provider: %v", err)
	}

	return &OpenAIFixture{
		provider:          provider,
		reasoningProvider: reasoningProvider,
	}
}

func (f *OpenAIFixture) Name() string {
	return "OpenAI"
}

func (f *OpenAIFixture) NewStandardModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.provider.NewModel(openaitest.TestModelName)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	return retry.WrapModel(model)
}

func (f *OpenAIFixture) NewReasoningModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.reasoningProvider.NewModel(openaitest.TestReasoningModelName,
		openai.WithReasoningEffort(openai.ReasoningEffortHigh),
		openai.WithReasoningSummary(openai.ReasoningSummaryDetailed),
	)
	if err != nil {
		t.Skipf("No reasoning model available: %v", err)
		return nil
	}

	return retry.WrapModel(model)
}

func (f *OpenAIFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *OpenAIFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestOpenAIConformance_Integration runs the generic conformance test suite for the OpenAI provider.
func TestOpenAIConformance_Integration(t *testing.T) {
	t.Parallel()

	fixture := NewOpenAIFixture(t)
	testsuite.Run(t, conformance.NewSuite(fixture))
}
