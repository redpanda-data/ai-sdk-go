package conformance

import "github.com/redpanda-data/ai-sdk-go/llm"

// Fixture defines the interface that each provider must implement
// to participate in conformance testing. This allows the generic test suite
// to test any provider implementation against the llm.Model interface.
type Fixture interface {
	// Name returns the provider name for test reporting
	Name() string

	// StandardModel returns a model instance suitable for standard testing
	// (basic generation, streaming, tools, structured output).
	// Returns nil if provider should be skipped (e.g., missing API key).
	StandardModel() llm.Model

	// ReasoningModel returns a model instance that supports reasoning capabilities.
	// Returns nil if the provider doesn't support reasoning models or if they
	// should be skipped.
	ReasoningModel() llm.Model

	// Models returns the list of all models available from this provider
	// for discovery testing. Returns nil or empty slice if model discovery
	// testing should be skipped.
	Models() []llm.ModelDiscoveryInfo

	// NewModel creates a new model instance by name for testing.
	// Used by the TestAllSupportedModels test to verify all models work.
	// Returns an error if the model cannot be created.
	NewModel(modelName string) (llm.Model, error)

	// ShouldSkip returns true if the entire test suite should be skipped
	// for this provider (e.g., missing required API keys or credentials).
	ShouldSkip() bool

	// SkipReason returns a human-readable explanation of why tests are being skipped.
	// Only called if ShouldSkip() returns true.
	SkipReason() string
}
