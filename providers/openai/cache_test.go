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

package openai_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/providers/testutil"
)

func TestOpenAICachedTokens_Integration(t *testing.T) {
	t.Parallel()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	provider, err := openai.NewProvider(apiKey, openai.WithTimeout(time.Minute*2))
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName)
	require.NoError(t, err)

	ctx := context.Background()

	// OpenAI caching requires 1024+ tokens to trigger.
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

	// First request - establishes conversation and cache
	response1, err := model.Generate(ctx, &llm.Request{Messages: messages})
	require.NoError(t, err)
	require.NotNil(t, response1)
	require.NotNil(t, response1.Usage)

	t.Logf("Request 1 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response1.Usage.InputTokens, response1.Usage.OutputTokens, response1.Usage.CachedTokens)

	// Run multiple iterations to increase chances of seeing cached tokens
	// OpenAI's automatic caching behavior is not deterministic
	var responses []*llm.Response
	responses = append(responses, response1)

	for i := 2; i <= 10; i++ {
		messages = append(messages, llm.Message{
			Role:    llm.RoleAssistant,
			Content: responses[len(responses)-1].Message.Content,
		})
		messages = append(messages, llm.Message{
			Role: llm.RoleUser,
			Content: []*llm.Part{
				llm.NewTextPart("Question: Say 'ok'."),
			},
		})

		resp, err := model.Generate(ctx, &llm.Request{Messages: messages})
		require.NoError(t, err)
		require.NotNil(t, resp)
		require.NotNil(t, resp.Usage)

		t.Logf("Request %d - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
			i, resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.CachedTokens)

		responses = append(responses, resp)
	}

	// Log cache statistics. OpenAI's automatic caching is server-side and
	// non-deterministic, so we don't assert on cache hits. This test verifies
	// that multi-turn conversations work and CachedTokens is correctly parsed.
	// Deterministic CachedTokens mapping is verified by TestResponseMapper_CachedTokens.
	totalCached := 0

	for i, resp := range responses {
		assert.GreaterOrEqual(t, resp.Usage.CachedTokens, 0, "CachedTokens should be non-negative")

		totalCached += resp.Usage.CachedTokens
		if resp.Usage.CachedTokens > 0 {
			t.Logf("Cache hit on request %d with %d cached tokens", i+1, resp.Usage.CachedTokens)
		}
	}

	t.Logf("Total cached tokens: %d across %d requests", totalCached, len(responses))
}
