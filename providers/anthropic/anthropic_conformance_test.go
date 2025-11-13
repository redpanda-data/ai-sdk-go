package anthropic_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
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
	reasoningModel, err := provider.NewModel(anthropictest.TestReasoningModelName,
		anthropic.WithThinking(true),
	)
	if err != nil {
		// Reasoning model is optional, just log but don't skip
		t.Logf("Failed to create reasoning model: %v", err)
	}

	return &AnthropicFixture{
		provider:       provider,
		standardModel:  standardModel,
		reasoningModel: reasoningModel,
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

// TestAnthropicConformance runs the generic conformance test suite for the Anthropic provider.
//
//nolint:paralleltest // Test suite manages its own lifecycle
func TestAnthropicConformance(t *testing.T) {
	fixture := NewAnthropicFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}
