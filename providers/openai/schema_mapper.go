package openai

import (
	"bytes"
	"encoding/json"
	"slices"
	"sort"
)

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
	cp, err := deepCopyMap(schema)
	if err != nil {
		return schema
	}

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

	props, ok := obj["properties"].(map[string]any)
	if !ok {
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

// deepCopyMap via JSON (simple and good enough here).
func deepCopyMap(m map[string]any) (map[string]any, error) {
	var buf bytes.Buffer

	enc := json.NewEncoder(&buf)
	dec := json.NewDecoder(&buf)

	err := enc.Encode(m)
	if err != nil {
		return nil, err
	}

	var cp map[string]any

	err = dec.Decode(&cp)
	if err != nil {
		return nil, err
	}

	return cp, nil
}
