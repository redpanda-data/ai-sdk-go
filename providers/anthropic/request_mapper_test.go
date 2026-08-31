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

package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestMapResponseFormat(t *testing.T) {
	t.Parallel()

	rm := NewRequestMapper(&Config{ModelName: ModelClaudeOpus5, MaxTokens: 1024})

	t.Run("json schema populates output_config.format", func(t *testing.T) {
		t.Parallel()

		schema := `{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`
		req := &llm.Request{
			Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("hi")}}},
			ResponseFormat: &llm.ResponseFormat{
				Type:       llm.ResponseFormatJSONSchema,
				JSONSchema: &llm.JSONSchema{Name: "person", Schema: []byte(schema)},
			},
		}

		apiReq, err := rm.ToProvider(req)
		require.NoError(t, err)
		assert.Equal(t, "object", apiReq.OutputConfig.Format.Schema["type"])
	})

	t.Run("json schema is adapted to the structured-output subset", func(t *testing.T) {
		t.Parallel()

		// Anthropic structured outputs require additionalProperties: false on
		// every object and reject constraint keywords like "minimum".
		schema := `{"type":"object","properties":{"age":{"type":"integer","minimum":0}},"required":["age"]}`
		req := &llm.Request{
			Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("hi")}}},
			ResponseFormat: &llm.ResponseFormat{
				Type:       llm.ResponseFormatJSONSchema,
				JSONSchema: &llm.JSONSchema{Name: "person", Schema: []byte(schema)},
			},
		}

		apiReq, err := rm.ToProvider(req)
		require.NoError(t, err)

		out := apiReq.OutputConfig.Format.Schema
		assert.Equal(t, false, out["additionalProperties"])

		age, ok := out["properties"].(map[string]any)["age"].(map[string]any)
		require.True(t, ok)
		assert.NotContains(t, age, "minimum")
	})

	t.Run("text format leaves the schema unset", func(t *testing.T) {
		t.Parallel()

		req := &llm.Request{
			Messages:       []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("hi")}}},
			ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatText},
		}

		apiReq, err := rm.ToProvider(req)
		require.NoError(t, err)
		assert.Nil(t, apiReq.OutputConfig.Format.Schema)
	})

	t.Run("json object mode is rejected, not silently dropped", func(t *testing.T) {
		t.Parallel()

		req := &llm.Request{
			Messages:       []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("hi")}}},
			ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject},
		}

		_, err := rm.ToProvider(req)
		require.ErrorIs(t, err, llm.ErrUnsupportedFeature)
	})

	t.Run("json schema without a schema is rejected", func(t *testing.T) {
		t.Parallel()

		req := &llm.Request{
			Messages:       []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("hi")}}},
			ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema},
		}

		_, err := rm.ToProvider(req)
		require.ErrorIs(t, err, llm.ErrRequestMapping)
	})
}
