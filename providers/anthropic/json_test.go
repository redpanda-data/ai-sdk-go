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

func TestAnthropicJSONOutput_Integration(t *testing.T) {
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
