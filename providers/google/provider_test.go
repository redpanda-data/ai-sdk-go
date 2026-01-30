package google

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWithBaseURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		inputURL    string
		expectedURL string
	}{
		{
			name:        "custom gateway URL",
			inputURL:    "https://gateway.example.com",
			expectedURL: "https://gateway.example.com",
		},
		{
			name:        "URL with path",
			inputURL:    "https://api.example.com/v1",
			expectedURL: "https://api.example.com/v1",
		},
		{
			name:        "URL with trailing slash",
			inputURL:    "https://api.example.com/",
			expectedURL: "https://api.example.com/",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			provider, err := NewProvider(ctx, "test-key", WithBaseURL(tt.inputURL))
			require.NoError(t, err)
			assert.Equal(t, tt.expectedURL, provider.BaseURL)
		})
	}
}

func TestWithBaseURLValidation(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// Empty URL should fail
	_, err := NewProvider(ctx, "test-key", WithBaseURL(""))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "base URL cannot be empty")
}

func TestWithHTTPClient(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// Valid HTTP client
	customClient := &http.Client{Timeout: 30 * time.Second}
	provider, err := NewProvider(ctx, "test-key", WithHTTPClient(customClient))
	require.NoError(t, err)
	assert.NotNil(t, provider.HTTPClient)
	assert.Equal(t, 30*time.Second, provider.HTTPClient.Timeout)

	// Nil HTTP client should fail
	_, err = NewProvider(ctx, "test-key", WithHTTPClient(nil))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "HTTP client cannot be nil")
}

func TestWithTimeout(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// Valid timeout
	provider, err := NewProvider(ctx, "test-key", WithTimeout(5*time.Minute))
	require.NoError(t, err)
	assert.Equal(t, 5*time.Minute, provider.Timeout)
	assert.Equal(t, 5*time.Minute, provider.HTTPClient.Timeout)

	// Invalid timeout should fail
	_, err = NewProvider(ctx, "test-key", WithTimeout(-1*time.Second))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "timeout must be positive")

	// Zero timeout should fail
	_, err = NewProvider(ctx, "test-key", WithTimeout(0))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "timeout must be positive")
}

func TestProviderOptionsDefaults(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// Provider without options should have defaults
	provider, err := NewProvider(ctx, "test-key")
	require.NoError(t, err)
	assert.NotNil(t, provider.HTTPClient)
	assert.Equal(t, 10*time.Minute, provider.Timeout)
	assert.Equal(t, 10*time.Minute, provider.HTTPClient.Timeout)
	assert.Empty(t, provider.BaseURL)
}

func TestProviderOptionsCombination(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// Combine multiple options
	customClient := &http.Client{Timeout: 1 * time.Second}
	provider, err := NewProvider(
		ctx,
		"test-key",
		WithBaseURL("https://gateway.example.com"),
		WithHTTPClient(customClient),
		WithTimeout(5*time.Minute),
	)
	require.NoError(t, err)
	assert.Equal(t, "https://gateway.example.com", provider.BaseURL)
	assert.NotNil(t, provider.HTTPClient)
	assert.Equal(t, 5*time.Minute, provider.Timeout)
	assert.Equal(t, 5*time.Minute, provider.HTTPClient.Timeout)
}
