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
	provider       *gemini.Provider
	standardModel  llm.Model
	reasoningModel llm.Model
	ctx            context.Context //nolint:containedctx // Context required for Gemini provider operations
}

// NewGeminiFixture creates a new Gemini test fixture.
func NewGeminiFixture(t *testing.T) *GeminiFixture {
	t.Helper()

	// Check for API key (skips test if not set)
	apiKey := geminitest.GetAPIKeyOrSkipTest(t)

	ctx := context.Background()

	// Create provider
	provider, err := gemini.NewProvider(ctx, apiKey)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Create standard model (Gemini 2.5 Flash)
	standardModel, err := provider.NewModel(geminitest.TestModelName)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	// Create reasoning model (Gemini 2.5 Pro with thinking enabled)
	reasoningModel, err := provider.NewModel(geminitest.TestReasoningModelName,
		gemini.WithThinking(true),
	)
	if err != nil {
		// Reasoning model is optional, just log but don't skip
		t.Logf("Failed to create reasoning model: %v", err)
	}

	return &GeminiFixture{
		provider:       provider,
		standardModel:  standardModel,
		reasoningModel: reasoningModel,
		ctx:            ctx,
	}
}

func (f *GeminiFixture) Name() string {
	return "Gemini"
}

func (f *GeminiFixture) StandardModel() llm.Model {
	return f.standardModel
}

func (f *GeminiFixture) ReasoningModel() llm.Model {
	return f.reasoningModel
}

func (f *GeminiFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *GeminiFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestGeminiConformance runs the generic conformance test suite for the Gemini provider.
//
//nolint:paralleltest // Test suite manages its own lifecycle
func TestGeminiConformance(t *testing.T) {
	fixture := NewGeminiFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}
