// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

func TestNewModel_DefaultMaxTokens(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	model, err := provider.NewModel(ModelClaudeSonnet5)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "expected *Model")

	// The default output budget must be generous but bounded — large enough not to
	// truncate ordinary agent turns, yet not the model max, which would reserve so
	// much context window that long conversations 400 before generating.
	assert.Equal(t, defaultMaxTokens, m.config.MaxTokens)
	assert.Greater(t, m.config.MaxTokens, 4096,
		"default output budget must be large enough for ordinary agent turns")
}

func TestModelsDiscoveryConstraints(t *testing.T) {
	t.Parallel()

	for _, m := range Catalog().All() {
		assert.Positive(t, m.Constraints.MaxInputTokens,
			"model %s missing MaxInputTokens — set Constraints on its catalog entry", m.ID)
		assert.Positive(t, m.Constraints.MaxOutputTokens,
			"model %s missing MaxOutputTokens — set Constraints on its catalog entry", m.ID)
	}
}
