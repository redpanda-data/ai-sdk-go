package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
