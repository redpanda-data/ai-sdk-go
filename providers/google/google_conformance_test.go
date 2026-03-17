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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/google/googletest"
)

// GoogleFixture implements the conformance.Fixture interface for Google Gemini provider.
type GoogleFixture struct {
	provider *google.Provider
	model    llm.Model
	ctx      context.Context //nolint:containedctx // Context required for Google provider operations
}

// NewGoogleFixture creates a new Google Gemini test fixture for a specific model.
func NewGoogleFixture(t *testing.T, modelName string) *GoogleFixture {
	t.Helper()

	// Check for API key (skips test if not set)
	apiKey := googletest.GetAPIKeyOrSkipTest(t)

	ctx := context.Background()

	// Create provider with a 2-minute timeout to fail fast on hung API calls,
	// consistent with the OpenAI and Anthropic conformance test fixtures.
	provider, err := google.NewProvider(ctx, apiKey, google.WithTimeout(2*time.Minute))
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Create model first to check capabilities
	model, err := provider.NewModel(modelName)
	if err != nil {
		t.Fatalf("Failed to create model %s: %v", modelName, err)
	}

	// If the model supports reasoning, recreate it with thinking enabled
	if model.Capabilities().Reasoning {
		model, err = provider.NewModel(modelName, google.WithThinking(true), google.WithThinkingBudget(4096))
		if err != nil {
			t.Fatalf("Failed to create model %s with thinking: %v", modelName, err)
		}
	}

	return &GoogleFixture{
		provider: provider,
		model:    model,
		ctx:      ctx,
	}
}

func (f *GoogleFixture) Name() string {
	return "Google"
}

func (f *GoogleFixture) StandardModel() llm.Model {
	return f.model
}

func (f *GoogleFixture) ReasoningModel() llm.Model {
	// Return the same model if it supports reasoning
	// The conformance suite will check capabilities and skip tests if needed
	if f.model.Capabilities().Reasoning {
		return f.model
	}

	return nil
}

func (f *GoogleFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *GoogleFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestGoogleConformance runs the conformance test suite for Google Gemini models.
// Tests multiple models including Gemini 3 Pro to ensure thought signature
// preservation works correctly for multi-turn tool calling.
//
//nolint:paralleltest // Test suite manages its own lifecycle
func TestGoogleConformance(t *testing.T) {
	modelsToTest := []string{
		google.ModelGemini25Flash,       // gemini-2.5-flash
		google.ModelGemini31ProPreview,  // gemini-3.1-pro-preview
		google.ModelGemini3FlashPreview, // gemini-3-flash-preview
	}

	for _, modelName := range modelsToTest {
		t.Run(modelName, func(t *testing.T) {
			fixture := NewGoogleFixture(t, modelName)
			suite.Run(t, conformance.NewSuite(fixture))
		})
	}
}
