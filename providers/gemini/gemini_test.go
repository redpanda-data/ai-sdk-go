package gemini

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProviderCreation(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// Valid provider creation with API key
	provider, err := NewProvider(ctx, "test-api-key-123")
	require.NoError(t, err)
	assert.NotNil(t, provider)
	assert.NotNil(t, provider.client)

	// Empty API key should fail
	_, err = NewProvider(ctx, "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "API key is required")
}

func TestProviderModels(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	provider, err := NewProvider(ctx, "test-api-key")
	require.NoError(t, err)

	models := provider.Models()
	assert.NotEmpty(t, models, "Should return available models")

	// Collect model names for verification
	modelNames := make([]string, len(models))
	for i, model := range models {
		modelNames[i] = model.Name
		assert.NotEmpty(t, model.Name, "Model name should not be empty")
		assert.NotEmpty(t, model.Label, "Model label should not be empty")
		assert.Equal(t, "gemini", model.Provider, "Provider should be 'gemini'")
	}

	// Verify expected models are present
	expectedModels := []string{
		"gemini-3-pro-preview",
		"gemini-3-flash-preview",
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-2.5-flash-lite",
		"gemini-2.0-flash",
	}
	for _, expected := range expectedModels {
		assert.Contains(t, modelNames, expected, "Should include %s", expected)
	}
}

func TestModelCreation(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	provider, err := NewProvider(ctx, "test-api-key")
	require.NoError(t, err)

	// Valid model creation
	model, err := provider.NewModel(ModelGemini25Flash)
	require.NoError(t, err)
	assert.NotNil(t, model)
	assert.Equal(t, ModelGemini25Flash, model.Name())

	// Valid model with options
	model, err = provider.NewModel(ModelGemini25Flash, WithTemperature(0.7), WithMaxTokens(100))
	require.NoError(t, err)
	assert.Equal(t, ModelGemini25Flash, model.Name())

	// Gemini 3 Pro Preview should be supported
	model, err = provider.NewModel(ModelGemini3ProPreview)
	require.NoError(t, err)
	assert.NotNil(t, model)
	assert.Equal(t, ModelGemini3ProPreview, model.Name())

	// Error cases
	_, err = provider.NewModel("nonexistent-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported Gemini model")
}

func TestModelConstraints(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	provider, err := NewProvider(ctx, "test-api-key")
	require.NoError(t, err)

	// Valid temperature within range
	_, err = provider.NewModel(ModelGemini25Pro, WithTemperature(1.0))
	require.NoError(t, err)

	// Temperature out of range (max is 2.0)
	_, err = provider.NewModel(ModelGemini25Pro, WithTemperature(2.5))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "out of range")

	// Gemini 3 Pro Preview temperature validation
	_, err = provider.NewModel(ModelGemini3ProPreview, WithTemperature(1.5))
	require.NoError(t, err)

	_, err = provider.NewModel(ModelGemini3ProPreview, WithTemperature(2.5))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "out of range")
}

func TestModelCapabilities(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	provider, err := NewProvider(ctx, "test-api-key")
	require.NoError(t, err)

	// Gemini 2.5 Flash capabilities
	model, err := provider.NewModel(ModelGemini25Flash)
	require.NoError(t, err)

	caps := model.Capabilities()
	assert.True(t, caps.Streaming)
	assert.True(t, caps.Tools)
	assert.True(t, caps.Vision)
	assert.True(t, caps.JSONMode)
	assert.True(t, caps.StructuredOutput)
	assert.True(t, caps.Reasoning)

	// Gemini 3 Pro Preview capabilities (should have reasoning support)
	model, err = provider.NewModel(ModelGemini3ProPreview)
	require.NoError(t, err)

	caps = model.Capabilities()
	assert.True(t, caps.Streaming, "Should support streaming")
	assert.True(t, caps.Tools, "Should support tools")
	assert.True(t, caps.Vision, "Should support vision")
	assert.True(t, caps.JSONMode, "Should support JSON mode")
	assert.True(t, caps.StructuredOutput, "Should support structured output")
	assert.True(t, caps.Reasoning, "Should support reasoning/thinking")
	assert.True(t, caps.MultiTurn, "Should support multi-turn")
	assert.True(t, caps.SystemPrompts, "Should support system prompts")

	// Gemini 2.0 Flash (older generation - no structured output)
	model, err = provider.NewModel(ModelGemini20Flash)
	require.NoError(t, err)

	caps = model.Capabilities()
	assert.True(t, caps.Streaming)
	assert.True(t, caps.Tools)
	assert.False(t, caps.StructuredOutput, "Older model lacks structured output")
	assert.False(t, caps.Reasoning, "Older model lacks explicit reasoning")
}

func TestModelTokenLimits(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		model          string
		expectedMaxIn  int
		expectedMaxOut int
	}{
		{
			name:           "Gemini 3 Pro Preview",
			model:          ModelGemini3ProPreview,
			expectedMaxIn:  1048576, // 1M
			expectedMaxOut: 65535,   // 65K
		},
		{
			name:           "Gemini 2.5 Pro",
			model:          ModelGemini25Pro,
			expectedMaxIn:  1048576, // 1M
			expectedMaxOut: 65535,   // 65K
		},
		{
			name:           "Gemini 2.5 Flash",
			model:          ModelGemini25Flash,
			expectedMaxIn:  1048576, // 1M
			expectedMaxOut: 65535,   // 65K
		},
		{
			name:           "Gemini 2.0 Flash",
			model:          ModelGemini20Flash,
			expectedMaxIn:  1048576, // 1M
			expectedMaxOut: 8192,    // 8K
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			modelDef, ok := supportedModels[tt.model]
			require.True(t, ok, "Model should be defined")

			assert.Equal(t, tt.expectedMaxIn, modelDef.Constraints.MaxInputTokens,
				"MaxInputTokens should match")
			assert.Equal(t, tt.expectedMaxOut, modelDef.Constraints.MaxOutputTokens,
				"MaxOutputTokens should match")
		})
	}
}
