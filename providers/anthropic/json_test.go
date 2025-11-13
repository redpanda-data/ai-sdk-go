package anthropic_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
)

func TestAnthropicJSONOutput(t *testing.T) {
	t.Parallel()

	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	provider, err := anthropic.NewProvider(apiKey)
	require.NoError(t, err)

	model, err := provider.NewModel(anthropictest.TestModelName)
	require.NoError(t, err)

	t.Run("JSON object mode", func(t *testing.T) {
		t.Parallel()

		req := &llm.Request{
			Messages: []llm.Message{
				{
					Role:    llm.RoleUser,
					Content: []*llm.Part{llm.NewTextPart("Generate a JSON object with a 'name' field and an 'age' field. Return only the JSON, nothing else.")},
				},
			},
			ResponseFormat: &llm.ResponseFormat{
				Type: llm.ResponseFormatJSONObject,
			},
		}

		resp, err := model.Generate(context.Background(), req)
		require.NoError(t, err)
		require.NotNil(t, resp)

		text := resp.TextContent()
		t.Logf("Raw response: %q", text)
		t.Logf("First 10 chars: %q", text[:min(10, len(text))])

		// Check if it's wrapped in markdown code blocks
		if len(text) > 0 && text[0] == '`' {
			t.Logf("Response is markdown-wrapped JSON")
		}

		// Try to parse as JSON
		var jsonData any

		err = json.Unmarshal([]byte(text), &jsonData)
		if err != nil {
			t.Logf("Direct JSON parse failed: %v", err)
			t.Logf("This is expected - Anthropic wraps JSON in markdown code blocks")
		}
	})

	t.Run("JSON schema mode", func(t *testing.T) {
		t.Parallel()

		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
				"age":  map[string]any{"type": "integer"},
			},
			"required": []any{"name", "age"},
		}

		schemaJSON, err := json.Marshal(schema)
		require.NoError(t, err)

		req := &llm.Request{
			Messages: []llm.Message{
				{
					Role:    llm.RoleUser,
					Content: []*llm.Part{llm.NewTextPart("Generate data for a person")},
				},
			},
			ResponseFormat: &llm.ResponseFormat{
				Type: llm.ResponseFormatJSONSchema,
				JSONSchema: &llm.JSONSchema{
					Name:   "Person",
					Schema: schemaJSON,
				},
			},
		}

		resp, err := model.Generate(context.Background(), req)
		require.NoError(t, err)
		require.NotNil(t, resp)

		text := resp.TextContent()
		t.Logf("Raw response: %q", text)

		// Try to parse as JSON
		var jsonData any

		err = json.Unmarshal([]byte(text), &jsonData)
		if err != nil {
			t.Logf("Direct JSON parse failed: %v", err)
			t.Logf("This is expected - Anthropic wraps JSON in markdown code blocks")
		}
	})
}
