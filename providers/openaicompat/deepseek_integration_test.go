package openaicompat_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
)

// TestDeepSeekMultiTurnReasoning verifies that reasoning traces work correctly
// across multiple conversation turns.
//
// Set DEEPSEEK_API_KEY to run this test:
//
//	DEEPSEEK_API_KEY=sk-xxx go test -v -run TestDeepSeekMultiTurnReasoning
//
// Optional environment variables:
//
//	DEEPSEEK_BASE_URL - API base URL (default: https://api.deepseek.com)
//	DEEPSEEK_MODEL - Model name (default: deepseek-reasoner)
func TestDeepSeekMultiTurnReasoning(t *testing.T) {
	t.Parallel()

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		t.Skip("DEEPSEEK_API_KEY not set, skipping integration test")
	}

	baseURL := os.Getenv("DEEPSEEK_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.deepseek.com"
	}

	modelName := os.Getenv("DEEPSEEK_MODEL")
	if modelName == "" {
		modelName = "deepseek-reasoner"
	}

	// Create provider
	provider, err := openaicompat.NewProvider(
		apiKey,
		openaicompat.WithBaseURL(baseURL),
		openaicompat.WithTimeout(3*time.Minute),
	)
	require.NoError(t, err, "Failed to create provider")

	// Create reasoning model
	model, err := provider.NewModel(modelName, openaicompat.WithReasoning())
	require.NoError(t, err, "Failed to create model")

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	t.Cleanup(cancel)

	// Define reusable message parts
	userMsg1 := llm.Message{
		Role: llm.RoleUser,
		Content: []*llm.Part{
			llm.NewTextPart("What is 15 * 23?"),
		},
	}

	// First turn with reasoning
	request1 := &llm.Request{
		Messages: []llm.Message{userMsg1},
	}

	response1, err := model.Generate(ctx, request1)
	require.NoError(t, err, "First turn should succeed")

	// Verify first response has reasoning and text
	var hasReasoning1, hasText1 bool

	for _, part := range response1.Message.Content {
		if part.IsReasoning() {
			hasReasoning1 = true
		}

		if part.IsText() {
			hasText1 = true
		}
	}

	assert.True(t, hasReasoning1, "First turn should have reasoning content")
	assert.True(t, hasText1, "First turn should have text content")

	// Second turn building on first
	userMsg2 := llm.Message{
		Role: llm.RoleUser,
		Content: []*llm.Part{
			llm.NewTextPart("Now add 100 to that result."),
		},
	}

	request2 := &llm.Request{
		Messages: []llm.Message{
			userMsg1,
			{
				Role:    llm.RoleAssistant,
				Content: response1.Message.Content, // Include full response with reasoning
			},
			userMsg2,
		},
	}

	response2, err := model.Generate(ctx, request2)
	require.NoError(t, err, "Second turn should succeed")

	// Verify second response has reasoning and text
	var hasReasoning2, hasText2 bool

	for _, part := range response2.Message.Content {
		if part.IsReasoning() {
			hasReasoning2 = true
		}

		if part.IsText() {
			hasText2 = true
		}
	}

	assert.True(t, hasReasoning2, "Second turn should have reasoning content")
	assert.True(t, hasText2, "Second turn should have text content")
}
