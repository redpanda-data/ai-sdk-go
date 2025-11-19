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

func TestOpenAICachedTokens(t *testing.T) {
	t.Parallel()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	provider, err := openai.NewProvider(apiKey, openai.WithTimeout(time.Minute*2))
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName)
	require.NoError(t, err)

	ctx := context.Background()

	// OpenAI caching requires 1024+ tokens to trigger
	// Generate a prompt with ~1200 tokens to ensure we exceed the threshold
	longContext := testutil.GenerateLargePrompt(1200)

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

	// Check if any request showed cached tokens
	totalCached := 0

	for i, resp := range responses {
		assert.GreaterOrEqual(t, resp.Usage.CachedTokens, 0)

		totalCached += resp.Usage.CachedTokens
		if resp.Usage.CachedTokens > 0 {
			t.Logf("Cache hit on request %d with %d cached tokens", i+1, resp.Usage.CachedTokens)
		}
	}

	// OpenAI should show caching on at least one subsequent request
	require.Positive(t, totalCached, "Expected cached tokens with OpenAI automatic caching after %d requests", len(responses))

	t.Logf("SUCCESS: Detected %d total cached tokens across %d requests", totalCached, len(responses))
}
