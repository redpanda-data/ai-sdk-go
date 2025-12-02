package conformance

import (
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Fixture defines the interface that provider packages must implement
// to participate in agent conformance testing.
//
// Each provider package (openai, anthropic, gemini, etc.) should create
// a fixture that demonstrates their provider works correctly with the
// agent layer, particularly for tool calling scenarios.
type Fixture interface {
	// Name returns the provider name (e.g., "OpenAI", "Anthropic", "Gemini")
	Name() string

	// StandardAgent creates an agent with a standard model and the given tool registry.
	// Returns nil if provider doesn't support agents or tools.
	StandardAgent(tools tool.Registry) (*llmagent.LLMAgent, error)

	// ReasoningAgent creates an agent with a reasoning model and the given tool registry.
	// Returns nil if provider doesn't support reasoning models.
	// This is optional - providers without reasoning models should return nil.
	ReasoningAgent(tools tool.Registry) (*llmagent.LLMAgent, error)
}
