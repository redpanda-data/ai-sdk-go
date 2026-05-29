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

package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAdaptSchemaForOpenAI_MCPShapes feeds a schema shaped like the JSON Schema
// an MCP server (protoc-gen-go-mcp) emits — base64 bytes, a dynamic Struct, a
// dynamic Value, and a map — and asserts the adapted result is OpenAI-strict
// valid: dynamic nodes collapsed to strings, bytes annotations stripped (with a
// base64 hint preserved), maps preserved, every object closed.
func TestAdaptSchemaForOpenAI_MCPShapes(t *testing.T) {
	t.Parallel()
	in := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"blob":   map[string]any{"type": "string", "format": "byte", "contentEncoding": "base64"},
			"meta":   map[string]any{"type": "object", "additionalProperties": true},                                                                                 // Struct
			"val":    map[string]any{"description": "a google.protobuf.Value"},                                                                                       // Value (typeless)
			"labels": map[string]any{"type": "object", "propertyNames": map[string]any{"pattern": "^.+$"}, "additionalProperties": map[string]any{"type": "string"}}, // map
		},
		"required": []any{"blob"},
	}
	out := NewSchemaMapper().AdaptSchemaForOpenAI(in)
	props := out["properties"].(map[string]any)

	blob := props["blob"].(map[string]any)
	_, hasFmt := blob["format"]
	_, hasEnc := blob["contentEncoding"]
	assert.False(t, hasFmt, "format:byte must be stripped")
	assert.False(t, hasEnc, "contentEncoding must be stripped")
	assert.Contains(t, blob["description"], "Base64", "base64 hint preserved in description")

	assert.Equal(t, "string", typeNoNull(props["meta"]), "Struct collapses to string")
	assert.Equal(t, "string", typeNoNull(props["val"]), "Value collapses to string")

	labels := props["labels"].(map[string]any)
	_, hasPN := labels["propertyNames"]
	assert.False(t, hasPN, "propertyNames stripped")
	// Map stays an object (OpenAI forces additionalProperties:false).
	assert.Equal(t, false, labels["additionalProperties"])

	// Root object closed and every property required.
	assert.Equal(t, false, out["additionalProperties"])
	req := map[string]bool{}
	for _, r := range out["required"].([]any) {
		req[r.(string)] = true
	}
	for name := range props {
		assert.True(t, req[name], "property %q must be required under strict", name)
	}

	// The input must not be mutated.
	origBlob := in["properties"].(map[string]any)["blob"].(map[string]any)
	require.Equal(t, "byte", origBlob["format"], "input schema must not be mutated")
}

// typeNoNull returns the non-null type of a node whose type may be "T" or
// ["T","null"].
func typeNoNull(node any) string {
	m, ok := node.(map[string]any)
	if !ok {
		return ""
	}
	switch t := m["type"].(type) {
	case string:
		return t
	case []any:
		for _, e := range t {
			if s, ok := e.(string); ok && s != "null" {
				return s
			}
		}
	}
	return ""
}
