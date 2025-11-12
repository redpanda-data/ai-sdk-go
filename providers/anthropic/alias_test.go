package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelAliases(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		alias         string
		expectedModel string
	}{
		{
			name:          "claude-sonnet-4-5 resolves to timestamped version",
			alias:         "claude-sonnet-4-5",
			expectedModel: ModelClaudeSonnet45,
		},
		{
			name:          "claude-4-sonnet resolves to latest Sonnet 4",
			alias:         "claude-4-sonnet",
			expectedModel: ModelClaudeSonnet45,
		},
		{
			name:          "claude-haiku-4-5 resolves to timestamped version",
			alias:         "claude-haiku-4-5",
			expectedModel: ModelClaudeHaiku45,
		},
		{
			name:          "claude-opus-4-1 resolves to timestamped version",
			alias:         "claude-opus-4-1",
			expectedModel: ModelClaudeOpus41,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider, err := NewProvider("test-key")
			require.NoError(t, err)

			model, err := provider.NewModel(tt.alias)
			require.NoError(t, err)
			require.NotNil(t, model)

			// Check that the resolved model name is correct
			m, ok := model.(*Model)
			require.True(t, ok)
			assert.Equal(t, tt.expectedModel, m.config.ModelName)
		})
	}
}

func TestCustomModelName(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	// Use claude-opus-4-1 as base but override with custom name
	model, err := provider.NewModel(
		"claude-opus-4-1",
		WithCustomModelName("claude-opus-4-2-beta"),
	)
	require.NoError(t, err)
	require.NotNil(t, model)

	m, ok := model.(*Model)
	require.True(t, ok)

	// Base model should be opus-4-1
	assert.Equal(t, ModelClaudeOpus41, m.config.ModelName)

	// Custom name should be set
	assert.Equal(t, "claude-opus-4-2-beta", m.config.CustomModelName)

	// Check that constraints are inherited from base model
	assert.Equal(t, int(32000), m.config.Constraints.MaxTokensLimit)
}

func TestUnsupportedModelRejected(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	_, err = provider.NewModel("claude-nonexistent-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported Anthropic model")
}

func TestCustomModelNameValidation(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	// Empty custom name should be rejected
	_, err = provider.NewModel(
		"claude-opus-4-1",
		WithCustomModelName(""),
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "custom model name cannot be empty")
}
