package openaicompat_test

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
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

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		t.Skip("DEEPSEEK_API_KEY not set, skipping DeepSeek conformance tests")
	}

	baseURL := os.Getenv("DEEPSEEK_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.deepseek.com"
	}

	// Create provider with DeepSeek base URL and extended timeout for reasoning
	provider, err := openaicompat.NewProvider(
		apiKey,
		openaicompat.WithBaseURL(baseURL),
		openaicompat.WithTimeout(3*time.Minute),
	)
	if err != nil {
		t.Fatalf("Failed to create DeepSeek provider: %v", err)
	}

	// Standard model (non-reasoning)
	standardModel, err := provider.NewModel("deepseek-chat")
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	// Reasoning model with reasoning capability enabled
	reasoningModel, err := provider.NewModel(
		"deepseek-reasoner",
		openaicompat.WithReasoning(),
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
	// Unlike OpenAI, DeepSeek exposes reasoning traces in the Chat Completions API
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
