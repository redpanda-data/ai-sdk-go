package gemini_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/gemini"
	"github.com/redpanda-data/ai-sdk-go/providers/gemini/geminitest"
	"github.com/redpanda-data/ai-sdk-go/providers/testutil"
)

func TestGeminiCachedTokens(t *testing.T) {
	t.Parallel()

	apiKey := geminitest.GetAPIKeyOrSkipTest(t)

	ctx := context.Background()

	provider, err := gemini.NewProvider(ctx, apiKey)
	require.NoError(t, err)

	model, err := provider.NewModel(geminitest.TestModelName)
	require.NoError(t, err)

	// Gemini 2.5 Flash requires 2048+ tokens for implicit caching
	// Generate a large prompt to trigger automatic caching
	longContext := testutil.GenerateLargePrompt(2500)

	messages := []llm.Message{
		{
			Role: llm.RoleUser,
			Content: []*llm.Part{
				llm.NewTextPart(longContext + "\n\nQuestion 1: Say 'yes' if you understand."),
			},
		},
	}

	// First request - establishes implicit cache
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

	// Verify all responses have the CachedTokens field populated
	assert.GreaterOrEqual(t, response1.Usage.CachedTokens, 0)
	assert.GreaterOrEqual(t, response2.Usage.CachedTokens, 0)
	assert.GreaterOrEqual(t, response3.Usage.CachedTokens, 0)

	// Check if any requests show cached tokens (Gemini implicit caching is automatic)
	totalCached := response2.Usage.CachedTokens + response3.Usage.CachedTokens

	// Gemini 2.5 should show implicit caching on subsequent requests
	require.Positive(t, totalCached, "Expected cached tokens with Gemini 2.5 implicit caching")

	t.Logf("SUCCESS: Detected %d total cached tokens across requests", totalCached)
}
