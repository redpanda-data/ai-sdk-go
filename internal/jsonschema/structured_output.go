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

// AdaptForStructuredOutput rewrites schema in place to fit the subset Claude's
// structured outputs accept (output_config.format on the Anthropic API,
// outputConfig.textFormat on Bedrock Converse), which is narrower than what a
// tool input_schema allows: every object node must set additionalProperties to
// false explicitly, and numeric/string/array constraint keywords are rejected
// with a validation error. The official Anthropic SDKs strip those constraints
// before sending; this mirrors that. Dropping a constraint never changes a
// value's type, so decoding is unaffected.
func AdaptForStructuredOutput(schema map[string]any) {
	Walk(schema, func(obj map[string]any) {
		for _, k := range []string{
			"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
			"minLength", "maxLength",
			"minItems", "maxItems", "uniqueItems",
		} {
			delete(obj, k)
		}

		if isObjectSchema(obj) {
			obj["additionalProperties"] = false
		}
	})
}

// isObjectSchema reports whether m describes an object schema, even when
// "type" is omitted.
func isObjectSchema(m map[string]any) bool {
	if t, ok := m["type"].(string); ok && t == "object" {
		return true
	}

	_, hasProps := m["properties"].(map[string]any)

	return hasProps
}
