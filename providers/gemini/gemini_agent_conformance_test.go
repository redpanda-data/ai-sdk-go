package gemini_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/agent/conformance"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/providers/gemini"
	"github.com/redpanda-data/ai-sdk-go/providers/gemini/geminitest"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// GeminiAgentFixture implements conformance.Fixture for Gemini provider.
type GeminiAgentFixture struct {
	provider *gemini.Provider
}

// NewGeminiAgentFixture creates a new Gemini agent test fixture.
func NewGeminiAgentFixture(t *testing.T) *GeminiAgentFixture {
	t.Helper()

	apiKey := geminitest.GetAPIKeyOrSkipTest(t)

	provider, err := gemini.NewProvider(t.Context(), apiKey)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return &GeminiAgentFixture{
		provider: provider,
	}
}

func (f *GeminiAgentFixture) Name() string {
	return "Gemini"
}

func (f *GeminiAgentFixture) StandardAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(geminitest.TestModelName)
	if err != nil {
		return nil, err
	}

	return llmagent.New(
		"test-agent",
		"You are a helpful assistant.",
		model,
		llmagent.WithTools(tools),
		llmagent.WithMaxTurns(10),
	)
}

func (f *GeminiAgentFixture) ReasoningAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(geminitest.TestReasoningModelName,
		gemini.WithThinking(true),
	)
	if err != nil {
		return nil, err
	}

	return llmagent.New(
		"reasoning-agent",
		"You are a helpful assistant with reasoning capabilities.",
		model,
		llmagent.WithTools(tools),
		llmagent.WithMaxTurns(10),
	)
}

// TestGeminiAgentConformance runs the agent conformance test suite for Gemini.
func TestGeminiAgentConformance(t *testing.T) {
	t.Parallel()

	fixture := NewGeminiAgentFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}
