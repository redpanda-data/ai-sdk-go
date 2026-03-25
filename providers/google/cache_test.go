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

package google_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/google/googletest"
	"github.com/redpanda-data/ai-sdk-go/providers/testutil"
)

func TestGeminiCachedTokens_Integration(t *testing.T) {
	t.Parallel()

	apiKey := googletest.GetAPIKeyOrSkipTest(t)

	ctx := context.Background()

	provider, err := google.NewProvider(ctx, apiKey)
	require.NoError(t, err)

	// Use gemini-2.5-flash for caching tests: lowest token threshold (1024) for implicit caching
	model, err := provider.NewModel(google.ModelGemini25Flash)
	require.NoError(t, err)

	// Gemini implicit caching requires 1024+ tokens for Flash models.
	// Use ~3000 tokens (well above the minimum) to maximize cache hit probability.
	longContext := testutil.GenerateLargePrompt(3000)

	messages := []llm.Message{
		{
			Role: llm.RoleUser,
			Content: []*llm.Part{
				llm.NewTextPart(longContext + "\n\nQuestion 1: Say 'yes' if you understand."),
			},
		},
	}

	// First request - establishes implicit cache
	// Note: Gemini's implicit caching may have a warm-up period
	response1, err := model.Generate(ctx, &llm.Request{Messages: messages})
	require.NoError(t, err)
	require.NotNil(t, response1)
	require.NotNil(t, response1.Usage)

	t.Logf("Request 1 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response1.Usage.InputTokens, response1.Usage.OutputTokens, response1.Usage.CachedTokens)

	// Continue the conversation
	messages = append(messages, llm.Message{
		Role:    llm.RoleAssistant,
		Content: response1.Message.Content,
	})
	messages = append(messages, llm.Message{
		Role: llm.RoleUser,
		Content: []*llm.Part{
			llm.NewTextPart("Question 2: Say 'ok'."),
		},
	})

	response2, err := model.Generate(ctx, &llm.Request{Messages: messages})
	require.NoError(t, err)
	require.NotNil(t, response2)
	require.NotNil(t, response2.Usage)

	t.Logf("Request 2 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response2.Usage.InputTokens, response2.Usage.OutputTokens, response2.Usage.CachedTokens)

	// Continue further
	messages = append(messages, llm.Message{
		Role:    llm.RoleAssistant,
		Content: response2.Message.Content,
	})
	messages = append(messages, llm.Message{
		Role: llm.RoleUser,
		Content: []*llm.Part{
			llm.NewTextPart("Question 3: Say 'ok' again."),
		},
	})

	response3, err := model.Generate(ctx, &llm.Request{Messages: messages})
	require.NoError(t, err)
	require.NotNil(t, response3)
	require.NotNil(t, response3.Usage)

	t.Logf("Request 3 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response3.Usage.InputTokens, response3.Usage.OutputTokens, response3.Usage.CachedTokens)

	// Continue further to give cache more time to warm up
	messages = append(messages, llm.Message{
		Role:    llm.RoleAssistant,
		Content: response3.Message.Content,
	})
	messages = append(messages, llm.Message{
		Role: llm.RoleUser,
		Content: []*llm.Part{
			llm.NewTextPart("Question 4: Say 'yes'."),
		},
	})

	response4, err := model.Generate(ctx, &llm.Request{Messages: messages})
	require.NoError(t, err)
	require.NotNil(t, response4)
	require.NotNil(t, response4.Usage)

	t.Logf("Request 4 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response4.Usage.InputTokens, response4.Usage.OutputTokens, response4.Usage.CachedTokens)

	// One more request
	messages = append(messages, llm.Message{
		Role:    llm.RoleAssistant,
		Content: response4.Message.Content,
	})
	messages = append(messages, llm.Message{
		Role: llm.RoleUser,
		Content: []*llm.Part{
			llm.NewTextPart("Question 5: Say 'ok' one last time."),
		},
	})

	response5, err := model.Generate(ctx, &llm.Request{Messages: messages})
	require.NoError(t, err)
	require.NotNil(t, response5)
	require.NotNil(t, response5.Usage)

	t.Logf("Request 5 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response5.Usage.InputTokens, response5.Usage.OutputTokens, response5.Usage.CachedTokens)

	// Log cache statistics. Google's implicit caching is opportunistic — cache hits
	// are not guaranteed by the API, so we don't assert on them. This test verifies
	// that multi-turn conversations work and CachedTokens is correctly parsed.
	// Deterministic CachedTokens mapping is verified by TestResponseMapper_CachedTokens.
	responses := []*llm.Response{response1, response2, response3, response4, response5}
	totalCached := 0

	for i, resp := range responses {
		assert.GreaterOrEqual(t, resp.Usage.CachedTokens, 0, "CachedTokens should be non-negative")

		totalCached += resp.Usage.CachedTokens
		if resp.Usage.CachedTokens > 0 {
			t.Logf("Cache hit on request %d with %d cached tokens", i+1, resp.Usage.CachedTokens)
		}
	}

	t.Logf("Total cached tokens: %d across 5 requests", totalCached)
}
