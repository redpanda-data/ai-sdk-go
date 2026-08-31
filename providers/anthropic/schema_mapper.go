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
	"github.com/redpanda-data/ai-sdk-go/internal/jsonschema"
)

// SchemaMapper transforms standard JSON Schemas to Anthropic-compatible schemas.
// Anthropic uses standard JSON Schema format, so this is simpler than OpenAI's requirements.
type SchemaMapper struct{}

// NewSchemaMapper creates a new SchemaMapper for schema transformations.
func NewSchemaMapper() *SchemaMapper { return &SchemaMapper{} }

// AdaptSchemaForAnthropic returns a transformed deep copy, never mutating the input.
// Anthropic validates tool input_schema against JSON Schema draft 2020-12 and
// rejects typeless / open-ended nodes (e.g. protobuf Struct/Value), so those are
// collapsed to a JSON-encoded string. Everything else is standard JSON Schema and
// passes through unchanged.
func (*SchemaMapper) AdaptSchemaForAnthropic(schema map[string]any) map[string]any {
	cp, err := jsonschema.DeepCopy(schema)
	if err != nil {
		return schema
	}

	jsonschema.CollapseDynamicNodes(cp)

	return cp
}

// adaptSchemaForStructuredOutput rewrites schema in place to fit the subset
// Anthropic structured outputs (output_config.format) accept, which is
// narrower than what tool input_schema allows: every object node must set
// additionalProperties to false explicitly, and numeric/string/array
// constraint keywords are rejected with a 400. The official SDKs strip those
// constraints before sending; this mirrors that. Dropping a constraint never
// changes a value's type, so decoding is unaffected.
func adaptSchemaForStructuredOutput(schema map[string]any) {
	jsonschema.Walk(schema, func(obj map[string]any) {
		for _, k := range []string{
			"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
			"minLength", "maxLength",
			"minItems", "maxItems", "uniqueItems",
		} {
			delete(obj, k)
		}

		if isObjectLike(obj) {
			obj["additionalProperties"] = false
		}
	})
}

// isObjectLike reports whether m describes an object schema, even when "type"
// is omitted.
func isObjectLike(m map[string]any) bool {
	if t, ok := m["type"].(string); ok && t == "object" {
		return true
	}

	_, hasProps := m["properties"].(map[string]any)

	return hasProps
}
