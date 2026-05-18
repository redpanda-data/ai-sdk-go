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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat/openaicompattest"
)

// TestGoogleOpenAICompat verifies that the openaicompat provider works against
// Google's Gemini OpenAI-compatible endpoint
// (https://generativelanguage.googleapis.com/v1beta/openai/).
//
// The base URL is /v1beta/openai/ — not /v1. This test confirms the provider
// passes the URL through without appending /v1.
//
// Set GOOGLE_API_KEY to run:
//
//	GOOGLE_API_KEY=xxx go test -v -run TestGoogleOpenAICompat
func TestGoogleOpenAICompat(t *testing.T) {
	t.Parallel()

	apiKey := openaicompattest.GetGoogleAPIKeyOrSkipTest(t)

	provider, err := openaicompat.NewProvider(
		apiKey,
		openaicompat.WithBaseURL(openaicompattest.GoogleDefaultBaseURL),
		openaicompat.WithTimeout(2*time.Minute),
	)
	require.NoError(t, err)

	model, err := provider.NewModel(openaicompattest.GoogleDefaultModel)
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

	var text string
	for _, part := range resp.Message.Content {
		if part.IsText() {
			text += part.Text
		}
	}

	assert.NotEmpty(t, text, "expected non-empty text response from Google compat endpoint")
}
