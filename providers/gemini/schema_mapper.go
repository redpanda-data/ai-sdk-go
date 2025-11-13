package gemini

import (
	"bytes"
	"encoding/json"
)

// SchemaMapper transforms standard JSON Schemas to Gemini-compatible schemas.
// Gemini uses standard JSON Schema format similar to Anthropic.
type SchemaMapper struct{}

// NewSchemaMapper creates a new SchemaMapper for schema transformations.
func NewSchemaMapper() *SchemaMapper { return &SchemaMapper{} }

// AdaptSchemaForGemini returns a transformed deep copy, never mutating the input.
// Gemini accepts standard JSON Schema, so minimal transformation is needed.
func (*SchemaMapper) AdaptSchemaForGemini(schema map[string]any) map[string]any {
	cp, err := deepCopyMap(schema)
	if err != nil {
		return schema
	}

	// Gemini uses standard JSON Schema format
	// No special transformations needed currently
	return cp
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
