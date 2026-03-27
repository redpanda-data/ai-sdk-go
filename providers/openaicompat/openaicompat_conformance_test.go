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

package openaicompat_test

import (
	"testing"

	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat/openaicompattest"
)

// OpenAICompatFixture implements the conformance.Fixture interface for OpenAI-compatible provider.
type OpenAICompatFixture struct {
	provider *openaicompat.Provider
}

// NewOpenAICompatFixture creates a new OpenAI-compatible test fixture.
func NewOpenAICompatFixture(t *testing.T) *OpenAICompatFixture {
	t.Helper()

	// Check for API key (skips test if not set)
	apiKey := openaicompattest.GetAPIKeyOrSkipTest(t)

	// Create provider
	provider, err := openaicompat.NewProvider(apiKey)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return &OpenAICompatFixture{
		provider: provider,
	}
}

func (f *OpenAICompatFixture) Name() string {
	return "OpenAICompat"
}

func (f *OpenAICompatFixture) NewStandardModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.provider.NewModel(openaicompattest.TestModelName)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	return retry.WrapModel(model)
}

func (f *OpenAICompatFixture) NewReasoningModel(t *testing.T) llm.Model {
	t.Helper()

	// OpenAI's o1 models don't expose reasoning traces in Chat Completions API.
	// The reasoning_content field is only available in:
	// 1. OpenAI's Responses API (not Chat Completions)
	// 2. Other OpenAI-compatible providers like DeepSeek-R1 via vLLM
	//
	// Since these conformance tests run against OpenAI's API, skip reasoning tests.
	// The provider correctly handles reasoning_content when present (e.g., with vLLM/DeepSeek-R1).
	t.Skip("Reasoning traces are not exposed against the OpenAI-compatible Chat Completions target")

	return nil
}

func (f *OpenAICompatFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *OpenAICompatFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestOpenAICompatConformance_Integration runs the generic conformance test suite for the OpenAI-compatible provider.
func TestOpenAICompatConformance_Integration(t *testing.T) {
	t.Parallel()

	fixture := NewOpenAICompatFixture(t)
	testsuite.Run(t, conformance.NewSuite(fixture))
}
