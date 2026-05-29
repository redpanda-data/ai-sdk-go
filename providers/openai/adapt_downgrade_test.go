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

// mp / sl are checked type-assertion helpers (errcheck check-type-assertions).
func mp(v any) map[string]any { m, _ := v.(map[string]any); return m }
func sl(v any) []any          { s, _ := v.([]any); return s }

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
	props := mp(out["properties"])

	blob := mp(props["blob"])
	_, hasFmt := blob["format"]
	_, hasEnc := blob["contentEncoding"]

	assert.False(t, hasFmt, "format:byte must be stripped")
	assert.False(t, hasEnc, "contentEncoding must be stripped")
	assert.Contains(t, blob["description"], "Base64", "base64 hint preserved in description")

	assert.Equal(t, "string", typeNoNull(props["meta"]), "Struct collapses to string")
	assert.Equal(t, "string", typeNoNull(props["val"]), "Value collapses to string")

	labels := mp(props["labels"])
	_, hasPN := labels["propertyNames"]
	assert.False(t, hasPN, "propertyNames stripped")
	// Map stays an object (OpenAI forces additionalProperties:false).
	assert.Equal(t, false, labels["additionalProperties"])

	// Root object closed and every property required.
	assert.Equal(t, false, out["additionalProperties"])

	req := map[string]bool{}

	for _, r := range sl(out["required"]) {
		rs, _ := r.(string)
		req[rs] = true
	}

	for name := range props {
		assert.True(t, req[name], "property %q must be required under strict", name)
	}

	// The input must not be mutated.
	origBlob := mp(mp(in["properties"])["blob"])
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

func TestStripUnsupportedKeywords(t *testing.T) {
	t.Parallel()

	t.Run("format byte removed, base64 hint folded into description", func(t *testing.T) {
		t.Parallel()

		n := map[string]any{"type": "string", "format": "byte", "contentEncoding": "base64"}
		stripUnsupportedKeywords(n)
		_, hasFormat := n["format"]
		assert.False(t, hasFormat)

		_, hasEnc := n["contentEncoding"]
		assert.False(t, hasEnc)
		assert.Contains(t, n["description"], "Base64")
	})

	t.Run("supported formats are kept", func(t *testing.T) {
		t.Parallel()

		for _, f := range []string{"date-time", "uuid", "email"} {
			n := map[string]any{"type": "string", "format": f}
			stripUnsupportedKeywords(n)
			assert.Equal(t, f, n["format"], "format %q should survive", f)
		}
	})

	t.Run("propertyNames and patternProperties removed", func(t *testing.T) {
		t.Parallel()

		n := map[string]any{
			"type":              "object",
			"propertyNames":     map[string]any{"pattern": "^x"},
			"patternProperties": map[string]any{"^y": map[string]any{"type": "string"}},
		}
		stripUnsupportedKeywords(n)
		_, hasPN := n["propertyNames"]
		_, hasPP := n["patternProperties"]

		assert.False(t, hasPN)
		assert.False(t, hasPP)
	})

	t.Run("recurses into nested properties and items", func(t *testing.T) {
		t.Parallel()

		n := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"blob": map[string]any{"type": "string", "format": "byte"},
				"list": map[string]any{
					"type":  "array",
					"items": map[string]any{"type": "string", "contentEncoding": "base64"},
				},
			},
		}
		stripUnsupportedKeywords(n)
		props := mp(n["properties"])
		_, blobFmt := mp(props["blob"])["format"]
		assert.False(t, blobFmt)

		_, itemEnc := mp(mp(props["list"])["items"])["contentEncoding"]
		assert.False(t, itemEnc)
	})

	t.Run("does not touch unrelated constraints", func(t *testing.T) {
		t.Parallel()

		n := map[string]any{"type": "string", "minLength": 3, "maxLength": 5, "pattern": "^a"}
		stripUnsupportedKeywords(n)
		require.Equal(t, "string", n["type"])
		assert.Equal(t, 3, n["minLength"])
		assert.Equal(t, 5, n["maxLength"])
		assert.Equal(t, "^a", n["pattern"])
	})
}
