package agentpack

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewModel_MissingProvider(t *testing.T) {
	t.Setenv("AI_PROVIDER", "")
	t.Setenv("AI_MODEL", "test")

	_, err := newModel(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "AI_PROVIDER env var is required")
}

func TestNewModel_MissingModel(t *testing.T) {
	t.Setenv("AI_PROVIDER", "anthropic")
	t.Setenv("AI_MODEL", "")

	_, err := newModel(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "AI_MODEL env var is required")
}

func TestNewModel_UnknownProvider(t *testing.T) {
	t.Setenv("AI_PROVIDER", "unknown")
	t.Setenv("AI_MODEL", "test")

	_, err := newModel(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown AI_PROVIDER: unknown")
}

func TestNewModel_AnthropicMissingKey(t *testing.T) {
	t.Setenv("AI_PROVIDER", "anthropic")
	t.Setenv("AI_MODEL", "claude-sonnet-4-6")
	t.Setenv("AI_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	_, err := newModel(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "API_KEY")
}

func TestNewModel_OpenAIMissingKey(t *testing.T) {
	t.Setenv("AI_PROVIDER", "openai")
	t.Setenv("AI_MODEL", "gpt-4")
	t.Setenv("AI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	_, err := newModel(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "API_KEY")
}

func TestNewModel_GoogleMissingKey(t *testing.T) {
	t.Setenv("AI_PROVIDER", "google")
	t.Setenv("AI_MODEL", "gemini-pro")
	t.Setenv("AI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")

	_, err := newModel(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "API_KEY")
}

func TestNewModel_OllamaNoKeyRequired(t *testing.T) {
	t.Setenv("AI_PROVIDER", "ollama")
	t.Setenv("AI_MODEL", "llama3.3")
	t.Setenv("AI_API_KEY", "")
	// Ollama does not require an API key — only validates model creation succeeds.
	model, err := newModel(context.Background())
	require.NoError(t, err)
	assert.NotNil(t, model)
}

func TestNewModel_Bedrock(t *testing.T) {
	t.Setenv("AI_PROVIDER", "bedrock")
	t.Setenv("AI_MODEL", "eu.anthropic.claude-opus-4-6-v1")
	t.Setenv("AWS_REGION", "eu-central-1")

	model, err := newModel(context.Background())
	require.NoError(t, err)
	assert.NotNil(t, model)
	assert.Equal(t, "eu.anthropic.claude-opus-4-6-v1", model.Name())
}

func TestNewModel_OllamaCustomBaseURL(t *testing.T) {
	t.Setenv("AI_PROVIDER", "ollama")
	t.Setenv("AI_MODEL", "llama3.3")
	t.Setenv("AI_BASE_URL", "http://custom:11434")

	model, err := newModel(context.Background())
	require.NoError(t, err)
	assert.NotNil(t, model)
}

func TestResolveAPIKey_Fallbacks(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		envVars  map[string]string
		expected string
	}{
		{
			name:     "AI_API_KEY takes precedence",
			provider: "anthropic",
			envVars:  map[string]string{"AI_API_KEY": "primary", "ANTHROPIC_API_KEY": "fallback"},
			expected: "primary",
		},
		{
			name:     "falls back to ANTHROPIC_API_KEY",
			provider: "anthropic",
			envVars:  map[string]string{"AI_API_KEY": "", "ANTHROPIC_API_KEY": "fallback"},
			expected: "fallback",
		},
		{
			name:     "falls back to OPENAI_API_KEY",
			provider: "openai",
			envVars:  map[string]string{"AI_API_KEY": "", "OPENAI_API_KEY": "openai-key"},
			expected: "openai-key",
		},
		{
			name:     "falls back to GOOGLE_API_KEY",
			provider: "google",
			envVars:  map[string]string{"AI_API_KEY": "", "GOOGLE_API_KEY": "google-key"},
			expected: "google-key",
		},
		{
			name:     "no fallback for unknown provider",
			provider: "ollama",
			envVars:  map[string]string{"AI_API_KEY": ""},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for k, v := range tt.envVars {
				t.Setenv(k, v)
			}
			// Clear env vars not set in this test case.
			for _, k := range []string{"AI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"} {
				if _, ok := tt.envVars[k]; !ok {
					t.Setenv(k, "")
				}
			}
			assert.Equal(t, tt.expected, resolveAPIKey(tt.provider))
		})
	}
}
