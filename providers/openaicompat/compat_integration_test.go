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

package openaicompat_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat/openaicompattest"
)

// TestOpenAICompatProviders verifies that the openaicompat provider works
// against third-party OpenAI-compatible endpoints whose base URLs do not
// end in /v1 (Anthropic uses /v1, Google uses /v1beta/openai/). This
// confirms that WithBaseURL passes the URL through without mangling it.
//
// Set ANTHROPIC_API_KEY / GOOGLE_API_KEY to run.
func TestOpenAICompatProviders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		apiKey  func(t *testing.T) string
		baseURL string
		model   string
	}{
		{
			name:    "Anthropic",
			apiKey:  openaicompattest.GetAnthropicAPIKeyOrSkipTest,
			baseURL: openaicompattest.AnthropicDefaultBaseURL,
			model:   openaicompattest.AnthropicDefaultModel,
		},
		{
			name:    "Google",
			apiKey:  openaicompattest.GetGoogleAPIKeyOrSkipTest,
			baseURL: openaicompattest.GoogleDefaultBaseURL,
			model:   openaicompattest.GoogleDefaultModel,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			apiKey := tt.apiKey(t)

			provider, err := openaicompat.NewProvider(
				apiKey,
				openaicompat.WithBaseURL(tt.baseURL),
				openaicompat.WithTimeout(2*time.Minute),
			)
			require.NoError(t, err)

			model, err := provider.NewModel(tt.model)
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			t.Cleanup(cancel)

			resp, err := model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("Say hello in one sentence.")}},
				},
			})
			require.NoError(t, err)
			require.NotEmpty(t, resp.Message.Content)

			var b strings.Builder

			for _, part := range resp.Message.Content {
				if part.IsText() {
					b.WriteString(part.Text)
				}
			}

			assert.NotEmpty(t, b.String(), "expected non-empty text response from %s compat endpoint", tt.name)
		})
	}
}
