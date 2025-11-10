package openaicompat

import (
	"testing"

	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestReasoningContentParsing verifies that reasoning_content is correctly
// extracted from API responses when present in ExtraFields.
func TestReasoningContentParsing(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper()

	t.Run("parse reasoning_content from non-streaming response", func(t *testing.T) {
		t.Parallel()

		// Simulate a response with reasoning_content in extra fields
		// This is what DeepSeek-R1 or vLLM with reasoning parser returns
		rawJSON := `{
			"id": "test-123",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "deepseek-reasoner",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "9.11 is greater than 9.8",
					"reasoning_content": "Let me think about this step by step:\n1. Compare 9.11 and 9.8\n2. 9.11 has value 9.11\n3. 9.8 has value 9.8\n4. Therefore 9.11 > 9.8"
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 20,
				"total_tokens": 30,
				"completion_tokens_details": {
					"reasoning_tokens": 15
				}
			}
		}`

		var apiResp openai.ChatCompletion

		err := apiResp.UnmarshalJSON([]byte(rawJSON))
		require.NoError(t, err)

		// Convert to our format
		response, err := mapper.FromProvider(&apiResp)
		require.NoError(t, err)

		// Should have 2 content parts: reasoning + text
		require.Len(t, response.Message.Content, 2, "Expected reasoning + text content")

		// First part should be reasoning
		assert.Equal(t, llm.PartReasoning, response.Message.Content[0].Kind)
		assert.Contains(t, response.Message.Content[0].ReasoningTrace.Text, "Let me think")
		assert.Contains(t, response.Message.Content[0].ReasoningTrace.Text, "step by step")

		// Second part should be text
		assert.Equal(t, llm.PartText, response.Message.Content[1].Kind)
		assert.Equal(t, "9.11 is greater than 9.8", response.Message.Content[1].Text)

		// Check reasoning tokens
		assert.Equal(t, 15, response.Usage.ReasoningTokens)
	})

	t.Run("response without reasoning_content", func(t *testing.T) {
		t.Parallel()

		// Standard response without reasoning_content
		rawJSON := `{
			"id": "test-456",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "gpt-4o-mini",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "Hello, how can I help?"
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 5,
				"completion_tokens": 10,
				"total_tokens": 15
			}
		}`

		var apiResp openai.ChatCompletion

		err := apiResp.UnmarshalJSON([]byte(rawJSON))
		require.NoError(t, err)

		response, err := mapper.FromProvider(&apiResp)
		require.NoError(t, err)

		// Should have only 1 content part: text
		require.Len(t, response.Message.Content, 1)
		assert.Equal(t, llm.PartText, response.Message.Content[0].Kind)
		assert.Equal(t, "Hello, how can I help?", response.Message.Content[0].Text)

		// No reasoning tokens
		assert.Equal(t, 0, response.Usage.ReasoningTokens)
	})

	t.Run("empty reasoning_content is skipped", func(t *testing.T) {
		t.Parallel()

		rawJSON := `{
			"id": "test-789",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "test-model",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "Answer",
					"reasoning_content": ""
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 5,
				"completion_tokens": 5,
				"total_tokens": 10
			}
		}`

		var apiResp openai.ChatCompletion

		err := apiResp.UnmarshalJSON([]byte(rawJSON))
		require.NoError(t, err)

		response, err := mapper.FromProvider(&apiResp)
		require.NoError(t, err)

		// Should have only text part, empty reasoning is skipped
		require.Len(t, response.Message.Content, 1)
		assert.Equal(t, llm.PartText, response.Message.Content[0].Kind)
	})
}
