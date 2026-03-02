package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveModelFamily(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		model    string
		expected string
	}{
		{"direct family", "claude-sonnet-4-6", "claude-sonnet-4-6"},
		{"timestamped", "claude-sonnet-4-5-20250929", "claude-sonnet-4-5"},
		{"bedrock cross-region", "eu.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"},
		{"bedrock standard", "anthropic.claude-sonnet-4-6-v2:0", "claude-sonnet-4-6"},
		{"bedrock haiku", "us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"},
		{"unknown model", "some-unknown-model", "some-unknown-model"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.expected, resolveModelFamily(tt.model))
		})
	}
}

func TestNormalizeBaseURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "URL without /v1",
			input:    "https://api.anthropic.com",
			expected: "https://api.anthropic.com",
		},
		{
			name:     "URL with /v1",
			input:    "https://api.anthropic.com/v1",
			expected: "https://api.anthropic.com",
		},
		{
			name:     "URL with trailing slash",
			input:    "https://api.anthropic.com/",
			expected: "https://api.anthropic.com",
		},
		{
			name:     "URL with /v1 and trailing slash",
			input:    "https://api.anthropic.com/v1/",
			expected: "https://api.anthropic.com",
		},
		{
			name:     "custom URL without /v1",
			input:    "https://custom-api.example.com",
			expected: "https://custom-api.example.com",
		},
		{
			name:     "custom URL with /v1",
			input:    "https://custom-api.example.com/v1",
			expected: "https://custom-api.example.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := normalizeBaseURL(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestWithBaseURLNormalization(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		inputURL    string
		expectedURL string
	}{
		{
			name:        "URL without /v1 stays unchanged",
			inputURL:    "https://api.anthropic.com",
			expectedURL: "https://api.anthropic.com",
		},
		{
			name:        "URL with /v1 gets stripped",
			inputURL:    "https://api.anthropic.com/v1",
			expectedURL: "https://api.anthropic.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider, err := NewProvider("test-key", WithBaseURL(tt.inputURL))
			require.NoError(t, err)
			assert.Equal(t, tt.expectedURL, provider.BaseURL)
		})
	}
}
