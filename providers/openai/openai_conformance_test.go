package openai_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
)

// OpenAIFixture implements the conformance.Fixture interface for OpenAI provider.
type OpenAIFixture struct {
	provider       *openai.Provider
	standardModel  llm.Model
	reasoningModel llm.Model
}

// NewOpenAIFixture creates a new OpenAI test fixture.
func NewOpenAIFixture(t *testing.T) *OpenAIFixture {
	t.Helper()

	// Check for API key (skips test if not set)
	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	// Create provider
	provider, err := openai.NewProvider(apiKey, openai.WithTimeout(time.Minute*2))
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Create standard model
	standardModel, err := provider.NewModel(openaitest.TestModelName)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	// Create reasoning model
	reasoningModel, err := provider.NewModel(openaitest.TestReasoningModelName,
		openai.WithReasoningEffort(openai.ReasoningEffortHigh),
		openai.WithReasoningSummary(openai.ReasoningSummaryDetailed),
	)
	if err != nil {
		// Reasoning model is optional, just log but don't skip
		t.Logf("Failed to create reasoning model: %v", err)
	}

	return &OpenAIFixture{
		provider:       provider,
		standardModel:  standardModel,
		reasoningModel: reasoningModel,
	}
}

func (f *OpenAIFixture) Name() string {
	return "OpenAI"
}

func (f *OpenAIFixture) StandardModel() llm.Model {
	return f.standardModel
}

func (f *OpenAIFixture) ReasoningModel() llm.Model {
	return f.reasoningModel
}

func (f *OpenAIFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *OpenAIFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestOpenAIConformance runs the generic conformance test suite for the OpenAI provider.
//
//nolint:paralleltest // Test suite manages its own lifecycle
func TestOpenAIConformance(t *testing.T) {
	fixture := NewOpenAIFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}
