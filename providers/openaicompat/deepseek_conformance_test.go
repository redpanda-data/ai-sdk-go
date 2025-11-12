package openaicompat_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat/openaicompattest"
)

// DeepSeekFixture implements the conformance.Fixture interface for DeepSeek API.
// This tests the openaicompat provider against DeepSeek's reasoning models.
type DeepSeekFixture struct {
	provider       *openaicompat.Provider
	standardModel  llm.Model
	reasoningModel llm.Model
}

// NewDeepSeekFixture creates a new DeepSeek test fixture.
func NewDeepSeekFixture(t *testing.T) *DeepSeekFixture {
	t.Helper()

	apiKey := openaicompattest.GetDeepSeekAPIKeyOrSkipTest(t)
	baseURL := openaicompattest.GetDeepSeekBaseURL()

	// Create provider with DeepSeek base URL and extended timeout for reasoning
	provider, err := openaicompat.NewProvider(
		apiKey,
		openaicompat.WithBaseURL(baseURL),
		openaicompat.WithTimeout(3*time.Minute),
	)
	if err != nil {
		t.Fatalf("Failed to create DeepSeek provider: %v", err)
	}

	// DeepSeek-specific capabilities
	// DeepSeek supports JSON mode (json_object) but not Structured Outputs (json_schema)
	deepseekCaps := llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true,  // Supports json_object
		StructuredOutput: false, // Does NOT support json_schema
		Vision:           true,
		Audio:            false,
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        false, // Set per-model below
	}

	// Standard model (non-reasoning)
	standardModel, err := provider.NewModel(
		openaicompattest.DeepSeekDefaultStandardModel,
		openaicompat.WithCapabilities(deepseekCaps),
	)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	// Reasoning model with reasoning capability enabled
	reasoningCaps := deepseekCaps
	reasoningCaps.Reasoning = true

	reasoningModel, err := provider.NewModel(
		openaicompattest.DeepSeekDefaultReasoningModel,
		openaicompat.WithCapabilities(reasoningCaps),
	)
	if err != nil {
		t.Fatalf("Failed to create reasoning model: %v", err)
	}

	return &DeepSeekFixture{
		provider:       provider,
		standardModel:  standardModel,
		reasoningModel: reasoningModel,
	}
}

func (f *DeepSeekFixture) Name() string {
	return "DeepSeek"
}

func (f *DeepSeekFixture) StandardModel() llm.Model {
	return f.standardModel
}

func (f *DeepSeekFixture) ReasoningModel() llm.Model {
	return f.reasoningModel
}

func (f *DeepSeekFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *DeepSeekFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestDeepSeekConformance runs the generic conformance test suite against DeepSeek API.
//
// Set DEEPSEEK_API_KEY to run these tests:
//
//	DEEPSEEK_API_KEY=sk-xxx go test -v -run TestDeepSeekConformance
//
// Optional environment variables:
//
//	DEEPSEEK_BASE_URL - API base URL (default: https://api.deepseek.com)
//
//nolint:paralleltest // Test suite manages its own lifecycle
func TestDeepSeekConformance(t *testing.T) {
	fixture := NewDeepSeekFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}
