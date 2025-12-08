package gemini_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/gemini"
	"github.com/redpanda-data/ai-sdk-go/providers/gemini/geminitest"
)

// GeminiFixture implements the conformance.Fixture interface for Gemini provider.
type GeminiFixture struct {
	provider *gemini.Provider
	model    llm.Model
	ctx      context.Context //nolint:containedctx // Context required for Gemini provider operations
}

// NewGeminiFixture creates a new Gemini test fixture for a specific model.
func NewGeminiFixture(t *testing.T, modelName string) *GeminiFixture {
	t.Helper()

	// Check for API key (skips test if not set)
	apiKey := geminitest.GetAPIKeyOrSkipTest(t)

	ctx := context.Background()

	// Create provider
	provider, err := gemini.NewProvider(ctx, apiKey)
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
		model, err = provider.NewModel(modelName, gemini.WithThinking(true))
		if err != nil {
			t.Fatalf("Failed to create model %s with thinking: %v", modelName, err)
		}
	}

	return &GeminiFixture{
		provider: provider,
		model:    model,
		ctx:      ctx,
	}
}

func (f *GeminiFixture) Name() string {
	return "Gemini"
}

func (f *GeminiFixture) StandardModel() llm.Model {
	return f.model
}

func (f *GeminiFixture) ReasoningModel() llm.Model {
	// Return the same model if it supports reasoning
	// The conformance suite will check capabilities and skip tests if needed
	if f.model.Capabilities().Reasoning {
		return f.model
	}

	return nil
}

func (f *GeminiFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *GeminiFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestGeminiConformance runs the conformance test suite for Gemini models.
// Tests multiple models including Gemini 3 Pro to ensure thought signature
// preservation works correctly for multi-turn tool calling.
//
//nolint:paralleltest // Test suite manages its own lifecycle
func TestGeminiConformance(t *testing.T) {
	modelsToTest := []string{
		gemini.ModelGemini25Flash,     // gemini-2.5-flash
		gemini.ModelGemini3ProPreview, // gemini-3-pro-preview
	}

	for _, modelName := range modelsToTest {
		t.Run(modelName, func(t *testing.T) {
			fixture := NewGeminiFixture(t, modelName)
			suite.Run(t, conformance.NewSuite(fixture))
		})
	}
}
