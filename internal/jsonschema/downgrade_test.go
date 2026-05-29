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

package jsonschema

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mp is a checked map type-assertion helper: it keeps errcheck's
// check-type-assertions satisfied while keeping the assertions readable.
func mp(v any) map[string]any { m, _ := v.(map[string]any); return m }

func TestCollapseDynamicNodes(t *testing.T) {
	t.Parallel()

	t.Run("typeless Value node collapses to string", func(t *testing.T) {
		// google.protobuf.Value renders typeless.
		n := map[string]any{"description": "a dynamic JSON value"}
		CollapseDynamicNodes(n)
		assert.Equal(t, "string", n["type"])
		assert.Contains(t, n["description"], "JSON-encoded string")
	})

	t.Run("open object Struct collapses to string", func(t *testing.T) {
		n := map[string]any{"type": "object", "additionalProperties": true}
		CollapseDynamicNodes(n)
		assert.Equal(t, "string", n["type"])
		assert.Nil(t, n["additionalProperties"])
	})

	t.Run("untyped array ListValue collapses to string", func(t *testing.T) {
		n := map[string]any{"type": "array", "items": map[string]any{}}
		CollapseDynamicNodes(n)
		assert.Equal(t, "string", n["type"])
		assert.Nil(t, n["items"])
	})

	t.Run("map is NOT collapsed (additionalProperties is a schema)", func(t *testing.T) {
		n := map[string]any{
			"type":                 "object",
			"additionalProperties": map[string]any{"type": "string"},
		}
		CollapseDynamicNodes(n)
		assert.Equal(t, "object", n["type"])
		assert.IsType(t, map[string]any{}, n["additionalProperties"])
	})

	t.Run("closed message object is not collapsed; recurses into properties", func(t *testing.T) {
		n := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id":     map[string]any{"type": "string"},
				"config": map[string]any{"description": "dynamic"}, // Value
			},
		}
		CollapseDynamicNodes(n)
		assert.Equal(t, "object", n["type"])
		props := mp(n["properties"])
		assert.Equal(t, "string", mp(props["id"])["type"])
		// nested Value collapsed
		assert.Equal(t, "string", mp(props["config"])["type"])
	})

	t.Run("Any keeps wrapper, collapses its typeless value", func(t *testing.T) {
		n := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"@type": map[string]any{"type": "string"},
				"value": map[string]any{}, // typeless
			},
			"required": []any{"@type"},
		}
		CollapseDynamicNodes(n)
		assert.Equal(t, "object", n["type"]) // wrapper preserved
		val := mp(mp(n["properties"])["value"])
		assert.Equal(t, "string", val["type"])
	})

	t.Run("dynamic inside array items collapses", func(t *testing.T) {
		n := map[string]any{
			"type":  "array",
			"items": map[string]any{"description": "dynamic value"}, // Value, not empty
		}
		CollapseDynamicNodes(n)
		// array stays array (items not empty), but item is collapsed
		assert.Equal(t, "array", n["type"])
		assert.Equal(t, "string", mp(n["items"])["type"])
	})

	t.Run("typed scalar untouched", func(t *testing.T) {
		n := map[string]any{"type": "string", "format": "uuid"}
		CollapseDynamicNodes(n)
		assert.Equal(t, "string", n["type"])
		assert.Equal(t, "uuid", n["format"])
	})

	t.Run("preserves original description", func(t *testing.T) {
		n := map[string]any{"description": "Custom metadata."}
		CollapseDynamicNodes(n)
		assert.Contains(t, n["description"], "Custom metadata.")
	})
}

func TestStripUnsupportedOpenAIKeywords(t *testing.T) {
	t.Parallel()

	t.Run("format byte removed, base64 hint folded into description", func(t *testing.T) {
		n := map[string]any{"type": "string", "format": "byte", "contentEncoding": "base64"}
		StripUnsupportedOpenAIKeywords(n)
		_, hasFormat := n["format"]
		assert.False(t, hasFormat)
		_, hasEnc := n["contentEncoding"]
		assert.False(t, hasEnc)
		assert.Contains(t, n["description"], "Base64")
	})

	t.Run("supported formats are kept", func(t *testing.T) {
		for _, f := range []string{"date-time", "uuid", "email"} {
			n := map[string]any{"type": "string", "format": f}
			StripUnsupportedOpenAIKeywords(n)
			assert.Equal(t, f, n["format"], "format %q should survive", f)
		}
	})

	t.Run("propertyNames and patternProperties removed", func(t *testing.T) {
		n := map[string]any{
			"type":              "object",
			"propertyNames":     map[string]any{"pattern": "^x"},
			"patternProperties": map[string]any{"^y": map[string]any{"type": "string"}},
		}
		StripUnsupportedOpenAIKeywords(n)
		_, hasPN := n["propertyNames"]
		_, hasPP := n["patternProperties"]
		assert.False(t, hasPN)
		assert.False(t, hasPP)
	})

	t.Run("recurses into nested properties and items", func(t *testing.T) {
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
		StripUnsupportedOpenAIKeywords(n)
		props := mp(n["properties"])
		_, blobFmt := mp(props["blob"])["format"]
		assert.False(t, blobFmt)
		_, itemEnc := mp(mp(props["list"])["items"])["contentEncoding"]
		assert.False(t, itemEnc)
	})

	t.Run("does not mutate beyond the listed keywords", func(t *testing.T) {
		n := map[string]any{"type": "string", "minLength": 3, "maxLength": 5, "pattern": "^a"}
		StripUnsupportedOpenAIKeywords(n)
		require.Equal(t, "string", n["type"])
		assert.Equal(t, 3, n["minLength"])
		assert.Equal(t, 5, n["maxLength"])
		assert.Equal(t, "^a", n["pattern"])
	})
}
