package anthropic_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
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

func TestAnthropicCachedTokens(t *testing.T) {
	t.Parallel()

	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	provider, err := anthropic.NewProvider(
		apiKey,
		anthropic.WithCaching(),
		anthropic.WithTimeout(time.Minute*3),
	)
	require.NoError(t, err)

	model, err := provider.NewModel(anthropictest.TestModelName)
	require.NoError(t, err)

	ctx := context.Background()

	// Anthropic requires large prompts for caching (minimum 1024 tokens for Sonnet)
	// Generate a large system prompt - need ~1400 tokens worth of characters to get 1024+ tokens
	longSystemPrompt := generateLargePrompt(1800)

	// System message with cache breakpoint - will be reused across all requests
	systemMessage := llm.Message{
		Role: llm.RoleSystem,
		Content: []*llm.Part{
			llm.NewTextPart(longSystemPrompt),
		},
	}

	// Request 1 - Creates cache
	request1 := &llm.Request{
		Messages: []llm.Message{
			systemMessage,
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("Question 1: Say 'yes' if you understand."),
				},
			},
		},
	}

	response1, err := model.Generate(ctx, request1)
	require.NoError(t, err)
	require.NotNil(t, response1)
	require.NotNil(t, response1.Usage)

	t.Logf("Request 1 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response1.Usage.InputTokens, response1.Usage.OutputTokens, response1.Usage.CachedTokens)

	// Request 2 - Should hit cache on system message
	request2 := &llm.Request{
		Messages: []llm.Message{
			systemMessage,
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("Question 2: Say 'ok'."),
				},
			},
		},
	}

	response2, err := model.Generate(ctx, request2)
	require.NoError(t, err)
	require.NotNil(t, response2)
	require.NotNil(t, response2.Usage)

	t.Logf("Request 2 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response2.Usage.InputTokens, response2.Usage.OutputTokens, response2.Usage.CachedTokens)

	// Request 3 - Should hit cache on system message
	request3 := &llm.Request{
		Messages: []llm.Message{
			systemMessage,
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("Question 3: Say 'confirmed'."),
				},
			},
		},
	}

	response3, err := model.Generate(ctx, request3)
	require.NoError(t, err)
	require.NotNil(t, response3)
	require.NotNil(t, response3.Usage)

	t.Logf("Request 3 - InputTokens: %d, OutputTokens: %d, CachedTokens: %d",
		response3.Usage.InputTokens, response3.Usage.OutputTokens, response3.Usage.CachedTokens)

	// Verify all responses have the CachedTokens field populated
	assert.GreaterOrEqual(t, response1.Usage.CachedTokens, 0)
	assert.GreaterOrEqual(t, response2.Usage.CachedTokens, 0)
	assert.GreaterOrEqual(t, response3.Usage.CachedTokens, 0)

	// Check if any requests show cached tokens (requests 2-3 should hit the cached system prompt)
	totalCached := response2.Usage.CachedTokens + response3.Usage.CachedTokens

	// Anthropic should show caching on subsequent requests that reuse the system prompt
	require.Positive(t, totalCached, "Expected cached tokens when reusing system prompt with cache_control marker")

	t.Logf("SUCCESS: Detected %d total cached tokens across requests 2-3", totalCached)
}
