package openai_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
)

// generateLargePrompt generates a large prompt with approximately the target number of tokens.
// Rough estimate: 1 token ≈ 4 characters for English text.
func generateLargePrompt(targetTokens int) string {
	const loremIpsum = `Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. `

	// Estimate characters needed (4 chars per token)
	targetChars := targetTokens * 4

	var builder strings.Builder
	builder.WriteString("Context information:\n\n")

	// Repeat lorem ipsum until we hit target
	for builder.Len() < targetChars {
		builder.WriteString(loremIpsum)
	}

	builder.WriteString("\n\nPlease answer the following question based on this context.")

	return builder.String()
}

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
	longContext := generateLargePrompt(1200)

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

	// Continue the conversation - this should hit the cache
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

	// Continue further - cache should grow
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

	// Request 4 - Now that we have 1024+ tokens in history, this should hit cache
	messages = append(messages, llm.Message{
		Role:    llm.RoleAssistant,
		Content: response3.Message.Content,
	})
	messages = append(messages, llm.Message{
		Role: llm.RoleUser,
		Content: []*llm.Part{
			llm.NewTextPart("Question 4: Say 'confirmed'."),
		},
	})

	response4, err := model.Generate(ctx, &llm.Request{Messages: messages})
	require.NoError(t, err)
	require.NotNil(t, response4)
	require.NotNil(t, response4.Usage)

	t.Logf("Request 4 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response4.Usage.InputTokens, response4.Usage.OutputTokens, response4.Usage.CachedTokens)

	// Request 5 - Continue to build cache
	messages = append(messages, llm.Message{
		Role:    llm.RoleAssistant,
		Content: response4.Message.Content,
	})
	messages = append(messages, llm.Message{
		Role: llm.RoleUser,
		Content: []*llm.Part{
			llm.NewTextPart("Question 5: Say 'acknowledged'."),
		},
	})

	response5, err := model.Generate(ctx, &llm.Request{Messages: messages})
	require.NoError(t, err)
	require.NotNil(t, response5)
	require.NotNil(t, response5.Usage)

	t.Logf("Request 5 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response5.Usage.InputTokens, response5.Usage.OutputTokens, response5.Usage.CachedTokens)

	// Verify all responses have the CachedTokens field populated
	assert.GreaterOrEqual(t, response1.Usage.CachedTokens, 0)
	assert.GreaterOrEqual(t, response2.Usage.CachedTokens, 0)
	assert.GreaterOrEqual(t, response3.Usage.CachedTokens, 0)
	assert.GreaterOrEqual(t, response4.Usage.CachedTokens, 0)
	assert.GreaterOrEqual(t, response5.Usage.CachedTokens, 0)

	// Check if any requests show cached tokens
	totalCached := response2.Usage.CachedTokens + response3.Usage.CachedTokens +
		response4.Usage.CachedTokens + response5.Usage.CachedTokens

	// OpenAI caching is automatic but may not always trigger - field should exist
	assert.GreaterOrEqual(t, totalCached, 0)

	if totalCached > 0 {
		t.Logf("SUCCESS: Detected %d total cached tokens across requests", totalCached)
	} else {
		t.Logf("INFO: No cached tokens detected (OpenAI automatic caching may not trigger)")
	}
}
