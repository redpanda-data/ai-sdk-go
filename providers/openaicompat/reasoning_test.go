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
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
)

// TestDeepSeekReasoningContent tests that reasoning_content is correctly extracted
// from DeepSeek's API responses.
func TestDeepSeekReasoningContent(t *testing.T) {
	t.Parallel()

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		t.Skip("DEEPSEEK_API_KEY not set, skipping integration test")
	}

	// Create provider pointing to DeepSeek's API
	provider, err := openaicompat.NewProvider(
		apiKey,
		openaicompat.WithBaseURL("https://api.deepseek.com"),
		openaicompat.WithTimeout(2*time.Minute),
	)
	require.NoError(t, err)

	// Create a reasoning model
	model, err := provider.NewModel("deepseek-reasoner", openaicompat.WithReasoning())
	require.NoError(t, err)

	ctx := context.Background()

	t.Run("non-streaming with reasoning", func(t *testing.T) {
		t.Parallel()

		request := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("What is 9.11 vs 9.8, which is greater?"),
					},
				},
			},
		}

		response, err := model.Generate(ctx, request)
		require.NoError(t, err)

		// Check that we have at least 2 parts: reasoning + text
		require.GreaterOrEqual(t, len(response.Message.Content), 2, "Should have reasoning and text content")

		// First part should be reasoning
		var (
			hasReasoning  = false
			hasText       = false
			reasoningText string
		)

		for _, part := range response.Message.Content {
			if part.Kind == llm.PartReasoning {
				hasReasoning = true
				reasoningText = part.ReasoningTrace.Text
			}

			if part.Kind == llm.PartText {
				hasText = true
			}
		}

		assert.True(t, hasReasoning, "Should have reasoning content")
		assert.True(t, hasText, "Should have text content")
		assert.NotEmpty(t, reasoningText, "Reasoning content should not be empty")

		if response.Usage.ReasoningTokens > 0 {
			assert.Positive(t, response.Usage.ReasoningTokens, "Should have reasoning tokens")
		}
	})

	t.Run("streaming with reasoning", func(t *testing.T) {
		t.Parallel()

		request := &llm.Request{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("How many Rs are in the word 'strawberry'?"),
					},
				},
			},
		}

		var (
			reasoningParts []string
			textParts      []string
			finalResponse  *llm.Response
		)

		for event, err := range model.GenerateEvents(ctx, request) {
			require.NoError(t, err)

			switch e := event.(type) {
			case llm.ContentPartEvent:
				switch e.Part.Kind {
				case llm.PartReasoning:
					reasoningParts = append(reasoningParts, e.Part.ReasoningTrace.Text)
				case llm.PartText:
					textParts = append(textParts, e.Part.Text)
				case llm.PartToolRequest, llm.PartToolResponse:
					// Not expected in this test
				}
			case llm.StreamEndEvent:
				finalResponse = e.Response
			}
		}

		assert.NotEmpty(t, reasoningParts, "Should receive reasoning content in stream")
		assert.NotEmpty(t, textParts, "Should receive text content in stream")
		require.NotNil(t, finalResponse)

		// Concatenate reasoning parts
		var reasoningBuilder strings.Builder

		for _, part := range reasoningParts {
			reasoningBuilder.WriteString(part)
		}

		fullReasoning := reasoningBuilder.String()

		assert.NotEmpty(t, fullReasoning, "Full reasoning should not be empty")

		if finalResponse.Usage.ReasoningTokens > 0 {
			assert.Positive(t, finalResponse.Usage.ReasoningTokens, "Should have reasoning tokens")
		}
	})
}
