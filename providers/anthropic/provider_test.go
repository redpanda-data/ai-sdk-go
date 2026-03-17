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
