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
	"time"

	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/google/googletest"
)

// GoogleFixture implements the conformance.Fixture interface for Google Gemini provider.
type GoogleFixture struct {
	provider  *google.Provider
	modelName string
}

// NewGoogleFixture creates a new Google Gemini test fixture for a specific model.
func NewGoogleFixture(t *testing.T, modelName string) *GoogleFixture {
	t.Helper()

	// Check for API key (skips test if not set)
	apiKey := googletest.GetAPIKeyOrSkipTest(t)

	// Create provider with a 2-minute timeout to fail fast on hung API calls,
	// consistent with the OpenAI and Anthropic conformance test fixtures.
	provider, err := google.NewProvider(t.Context(), apiKey, google.WithTimeout(2*time.Minute))
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return &GoogleFixture{
		provider:  provider,
		modelName: modelName,
	}
}

func (f *GoogleFixture) Name() string {
	return "Google"
}

func (f *GoogleFixture) NewStandardModel(t *testing.T) llm.Model {
	t.Helper()

	baseModel, err := f.provider.NewModel(f.modelName)
	if err != nil {
		t.Fatalf("Failed to create model %s: %v", f.modelName, err)
	}

	if baseModel.Capabilities().Reasoning {
		model, err := f.provider.NewModel(f.modelName, google.WithThinking(true), google.WithThinkingBudget(4096))
		if err != nil {
			t.Fatalf("Failed to create model %s with thinking: %v", f.modelName, err)
		}

		return retry.WrapModel(model)
	}

	return retry.WrapModel(baseModel)
}

func (f *GoogleFixture) NewReasoningModel(t *testing.T) llm.Model {
	t.Helper()

	baseModel, err := f.provider.NewModel(f.modelName)
	if err != nil {
		t.Fatalf("Failed to create model %s: %v", f.modelName, err)
	}

	if !baseModel.Capabilities().Reasoning {
		t.Skip("No reasoning model available")
		return nil
	}

	model, err := f.provider.NewModel(f.modelName, google.WithThinking(true), google.WithThinkingBudget(4096))
	if err != nil {
		t.Fatalf("Failed to create model %s with thinking: %v", f.modelName, err)
	}

	return retry.WrapModel(model)
}

func (f *GoogleFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *GoogleFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestGoogleConformance_Integration runs the conformance test suite for Google Gemini models.
// Tests multiple models including Gemini 3 Pro to ensure thought signature
// preservation works correctly for multi-turn tool calling.
func TestGoogleConformance_Integration(t *testing.T) {
	t.Parallel()

	modelsToTest := []string{
		google.ModelGemini25Flash,       // gemini-2.5-flash
		google.ModelGemini31ProPreview,  // gemini-3.1-pro-preview
		google.ModelGemini3FlashPreview, // gemini-3-flash-preview
	}

	for _, modelName := range modelsToTest {
		t.Run(modelName, func(t *testing.T) {
			t.Parallel()

			fixture := NewGoogleFixture(t, modelName)
			testsuite.Run(t, conformance.NewSuite(fixture))
		})
	}
}
