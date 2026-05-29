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
	"slices"
	"sort"

	"github.com/redpanda-data/ai-sdk-go/internal/jsonschema"
)

// openAISupportedStringFormats is the set of JSON Schema string "format" values
// OpenAI Structured Outputs / strict function calling accepts. Any other value
// (e.g. "byte", "uri", "uri-reference") is rejected with a 400.
// See https://platform.openai.com/docs/guides/structured-outputs.
var openAISupportedStringFormats = map[string]bool{
	"date-time": true,
	"time":      true,
	"date":      true,
	"duration":  true,
	"email":     true,
	"hostname":  true,
	"ipv4":      true,
	"ipv6":      true,
	"uuid":      true,
}

// stripUnsupportedKeywords removes JSON Schema keywords OpenAI strict mode
// rejects outright. These are annotations/constraints whose removal does not
// change a value's type, so dropping them is lossless for decoding (a base64
// "bytes" field, for example, is still a string).
func stripUnsupportedKeywords(node any) {
	jsonschema.Walk(node, func(obj map[string]any) {
		// Object/map keywords OpenAI does not permit.
		delete(obj, "propertyNames")
		delete(obj, "patternProperties")
		delete(obj, "minProperties")
		delete(obj, "maxProperties")
		delete(obj, "dependentRequired")
		delete(obj, "dependentSchemas")

		// Content keywords (e.g. base64 bytes) are not part of the subset.
		delete(obj, "contentEncoding")
		delete(obj, "contentMediaType")
		delete(obj, "contentSchema")

		// "format" survives only for the documented supported values. When a
		// non-supported format is dropped from a bytes field, fold the base64
		// hint into the description so the model still knows to base64-encode.
		if f, ok := obj["format"].(string); ok && !openAISupportedStringFormats[f] {
			if f == "byte" {
				appendBase64Hint(obj)
			}

			delete(obj, "format")
		}
	})
}

func appendBase64Hint(obj map[string]any) {
	const hint = "Base64-encoded binary data."

	if d, ok := obj["description"].(string); ok && d != "" {
		obj["description"] = d + " " + hint
		return
	}

	obj["description"] = hint
}

// SchemaMapper transforms standard JSON Schemas to OpenAI-compatible schemas.
// Notes:
// - All object properties must be listed in "required"; optionality is represented by nullability.
// - For Structured Outputs / strict tools, objects should have additionalProperties: false.
// See: https://platform.openai.com/docs/guides/structured-outputs
type SchemaMapper struct{}

// NewSchemaMapper creates a new SchemaMapper for schema transformations.
func NewSchemaMapper() *SchemaMapper { return &SchemaMapper{} }

// AdaptSchemaForOpenAI returns a transformed deep copy, never mutating the input.
func (*SchemaMapper) AdaptSchemaForOpenAI(schema map[string]any) map[string]any {
	cp, err := jsonschema.DeepCopy(schema)
	if err != nil {
		return schema
	}

	// Collapse open-ended/dynamic nodes (e.g. protobuf Struct/Value) to a
	// JSON-encoded string, then drop keywords OpenAI strict mode rejects, before
	// applying OpenAI's structural requirements.
	jsonschema.CollapseDynamicNodes(cp)
	stripUnsupportedKeywords(cp)
	transformSchemaForOpenAI(cp)

	return cp
}

// ---- helpers ----

func transformSchemaForOpenAI(node any) {
	obj, ok := node.(map[string]any)
	if !ok {
		return
	}

	// Convert "nullable": true to union-with-null early
	normalizeNullable(obj)

	// Recurse combinators first
	for _, k := range []string{"allOf", "anyOf", "oneOf"} {
		if arr, ok := obj[k].([]any); ok {
			for _, sub := range arr {
				transformSchemaForOpenAI(sub)
			}
		}
	}

	// Recurse $defs/definitions lightly (we're not resolving $ref, but we can normalize nested shapes)
	for _, k := range []string{"$defs", "definitions"} {
		if defs, ok := obj[k].(map[string]any); ok {
			for _, sub := range defs {
				transformSchemaForOpenAI(sub)
			}
		}
	}

	// Recurse items (object or tuple)
	switch it := obj["items"].(type) {
	case map[string]any:
		transformSchemaForOpenAI(it)
	case []any:
		for _, sub := range it {
			transformSchemaForOpenAI(sub)
		}
	}

	// Recurse patternProperties
	if pp, ok := obj["patternProperties"].(map[string]any); ok {
		for _, sub := range pp {
			transformSchemaForOpenAI(sub)
		}
	}

	// Detect object schemas even when "type" is omitted
	if !isObjectLike(obj) {
		return
	}

	// Force additionalProperties: false for OpenAI Structured Outputs strict mode
	obj["additionalProperties"] = false

	// Ensure properties field exists (OpenAI requires it, even if empty)
	props, ok := obj["properties"].(map[string]any)
	if !ok {
		// Add empty properties map if missing
		props = make(map[string]any)
		obj["properties"] = props
	}

	// If properties is empty, no further transformation needed
	if len(props) == 0 {
		return
	}

	// Recurse properties first
	for _, p := range props {
		transformSchemaForOpenAI(p)
	}

	// Remember original required set
	origReq := toStringSet(obj["required"])

	// For each property not originally required, make it nullable
	for name, raw := range props {
		if _, wasReq := origReq[name]; wasReq {
			continue
		}

		pm, ok := raw.(map[string]any)
		if !ok {
			continue
		}

		makeOptionalByAllowingNull(pm) // idempotent
	}

	// Now set required = all property names (preserving original order, then sorted remainder)
	obj["required"] = mergeRequiredStable(obj["required"], props)
}

func isObjectLike(m map[string]any) bool {
	if t, ok := m["type"].(string); ok && t == "object" {
		return true
	}

	_, hasProps := m["properties"].(map[string]any)
	_, hasPat := m["patternProperties"].(map[string]any)

	return hasProps || hasPat
}

func normalizeNullable(m map[string]any) {
	nb, ok := m["nullable"].(bool)
	if !ok || !nb {
		return
	}
	// Prefer union with "null"
	if t, ok := m["type"]; ok {
		switch tt := t.(type) {
		case string:
			m["type"] = []any{tt, "null"}
		case []any:
			if !slices.Contains(tt, "null") {
				m["type"] = append(tt, "null")
			}
		}
	} else if enumVals, ok := m["enum"].([]any); ok {
		// No type, but has enum: add null to enum set
		if !slices.Contains(enumVals, nil) {
			m["enum"] = append(enumVals, nil)
		}
	}

	delete(m, "nullable")
}

func makeOptionalByAllowingNull(pm map[string]any) {
	// Types: add "null" to type union
	switch t := pm["type"].(type) {
	case string:
		pm["type"] = []any{t, "null"}
	case []any:
		if !slices.Contains(t, "null") {
			pm["type"] = append(t, "null")
		}
	default:
		// If there's no type but there is an enum/const, make enum nullable
		if ev, ok := pm["enum"].([]any); ok && !slices.Contains(ev, nil) {
			pm["enum"] = append(ev, nil)
		}

		if c, ok := pm["const"]; ok {
			pm["enum"] = []any{c, nil}
			delete(pm, "const")
		}
	}

	// Remove legacy nullable flag to avoid conflict
	delete(pm, "nullable")
}

func mergeRequiredStable(existing any, props map[string]any) []any {
	// preserve original required order first
	out := []any{}
	seen := map[string]struct{}{}

	if arr, ok := existing.([]any); ok {
		for _, v := range arr {
			if s, ok := v.(string); ok {
				if _, has := props[s]; has {
					out = append(out, s)
					seen[s] = struct{}{}
				}
			}
		}
	}
	// Add the rest sorted for determinism
	var rest []string

	for name := range props {
		if _, ok := seen[name]; !ok {
			rest = append(rest, name)
		}
	}

	sort.Strings(rest)

	for _, n := range rest {
		out = append(out, n)
	}

	return out
}

func toStringSet(v any) map[string]struct{} {
	res := map[string]struct{}{}

	if arr, ok := v.([]any); ok {
		for _, x := range arr {
			if s, ok := x.(string); ok {
				res[s] = struct{}{}
			}
		}
	}

	return res
}
