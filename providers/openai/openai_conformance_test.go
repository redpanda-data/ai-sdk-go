package openai_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
)

// OpenAIFixture implements the conformance.Fixture interface for OpenAI provider.
type OpenAIFixture struct {
	provider       *openai.Provider
	apiKey         string
	shouldSkip     bool
	skipReason     string
	standardModel  llm.Model
	reasoningModel llm.Model
}

// NewOpenAIFixture creates a new OpenAI test fixture.
func NewOpenAIFixture(t *testing.T) *OpenAIFixture {
	t.Helper()

	fixture := &OpenAIFixture{}

	// Check for API key - use fixture skip mechanism for better reporting
	apiKey := openaitest.GetAPIKeyOrSkipTest(t)
	fixture.apiKey = apiKey

	// Create provider
	provider, err := openai.NewProvider(apiKey)
	if err != nil {
		fixture.shouldSkip = true
		fixture.skipReason = "Failed to create provider: " + err.Error()
		return fixture
	}

	fixture.provider = provider

	// Create standard model
	standardModel, err := provider.NewModel(openaitest.TestModelName)
	if err != nil {
		fixture.shouldSkip = true
		fixture.skipReason = "Failed to create standard model: " + err.Error()
		return fixture
	}

	fixture.standardModel = standardModel

	// Create reasoning model
	reasoningModel, err := provider.NewModel(openaitest.TestReasoningModelName,
		openai.WithReasoningEffort(openai.ReasoningEffortHigh),
		openai.WithReasoningSummary(openai.ReasoningSummaryDetailed),
	)
	if err != nil {
		// Reasoning model is optional, just log but don't skip
		t.Logf("Failed to create reasoning model: %v", err)
	} else {
		fixture.reasoningModel = reasoningModel
	}

	return fixture
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

func (f *OpenAIFixture) ShouldSkip() bool {
	return f.shouldSkip
}

func (f *OpenAIFixture) SkipReason() string {
	return f.skipReason
}

func (f *OpenAIFixture) Models() []llm.ModelDiscoveryInfo {
	if f.provider == nil {
		return nil
	}
	return f.provider.Models()
}

func (f *OpenAIFixture) NewModel(modelName string) (llm.Model, error) {
	if f.provider == nil {
		return nil, nil
	}
	return f.provider.NewModel(modelName)
}

// TestOpenAIConformance runs the generic conformance test suite for the OpenAI provider.
//
//nolint:paralleltest // Test suite manages its own lifecycle
func TestOpenAIConformance(t *testing.T) {
	fixture := NewOpenAIFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}
