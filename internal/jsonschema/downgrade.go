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

// Package jsonschema holds provider-agnostic helpers for downgrading standard
// JSON Schema (e.g. schemas exposed by MCP servers) into the narrower subsets
// individual LLM providers accept as tool input_schema. MCP servers legitimately
// publish full JSON Schema; per the MCP ecosystem the client is responsible for
// adapting it per provider (the same way LangChain, the OpenAI Agents SDK and
// the Vercel AI SDK do).
package jsonschema

// OpenAISupportedStringFormats is the set of JSON Schema string "format" values
// OpenAI Structured Outputs / strict function calling accepts. Any other value
// (e.g. "byte", "uri", "uri-reference") is rejected with a 400.
// See https://platform.openai.com/docs/guides/structured-outputs.
var OpenAISupportedStringFormats = map[string]bool{
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

// StripUnsupportedOpenAIKeywords walks the schema and removes keywords OpenAI
// strict mode rejects outright. These are all annotations/constraints whose
// removal does not change a value's type, so dropping them is lossless for
// decoding (a base64 "bytes" field, for example, is still a string).
func StripUnsupportedOpenAIKeywords(node any) {
	walk(node, func(obj map[string]any) {
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
		if f, ok := obj["format"].(string); ok && !OpenAISupportedStringFormats[f] {
			if f == "byte" {
				appendBase64Hint(obj)
			}
			delete(obj, "format")
		}
	})
}

func appendBase64Hint(obj map[string]any) {
	const hint = "Base64-encoded binary data."
	switch d := obj["description"].(type) {
	case string:
		if d == "" {
			obj["description"] = hint
		} else {
			obj["description"] = d + " " + hint
		}
	default:
		obj["description"] = hint
	}
}

// CollapseDynamicNodes rewrites schema nodes that describe open-ended / dynamic
// JSON (no fixed type) into a single JSON-encoded string node. OpenAI strict and
// Anthropic both reject typeless or open-object schemas as tool input_schema
// (only Gemini accepts them), so the common-denominator representation is a
// string the model fills with JSON text; the MCP server parses it back.
//
// It targets exactly the protobuf dynamic well-known types as they render in
// standard JSON Schema:
//   - google.protobuf.Value     -> typeless node (no "type")
//   - google.protobuf.Struct    -> {"type":"object","additionalProperties":true}
//   - google.protobuf.ListValue -> {"type":"array","items":{}}
//   - google.protobuf.Any.value -> typeless node
//
// Closed message objects (with "properties"), maps (additionalProperties is a
// schema), and typed scalars/arrays are left untouched.
func CollapseDynamicNodes(node any) {
	m, ok := node.(map[string]any)
	if !ok {
		if arr, ok := node.([]any); ok {
			for _, e := range arr {
				CollapseDynamicNodes(e)
			}
		}
		return
	}

	if isDynamicNode(m) {
		desc := "JSON value, provided as a JSON-encoded string."
		if d, ok := m["description"].(string); ok && d != "" {
			desc = d + " Provide it as a JSON-encoded string."
		}
		for k := range m {
			delete(m, k)
		}
		m["type"] = "string"
		m["description"] = desc
		return
	}

	// Recurse into the standard subschema-bearing keywords.
	if props, ok := m["properties"].(map[string]any); ok {
		for _, v := range props {
			CollapseDynamicNodes(v)
		}
	}
	for _, k := range []string{"$defs", "definitions", "patternProperties"} {
		if defs, ok := m[k].(map[string]any); ok {
			for _, v := range defs {
				CollapseDynamicNodes(v)
			}
		}
	}
	for _, k := range []string{"anyOf", "oneOf", "allOf", "prefixItems"} {
		if arr, ok := m[k].([]any); ok {
			for _, v := range arr {
				CollapseDynamicNodes(v)
			}
		}
	}
	switch it := m["items"].(type) {
	case map[string]any:
		CollapseDynamicNodes(it)
	case []any:
		for _, v := range it {
			CollapseDynamicNodes(v)
		}
	}
	// additionalProperties may be a schema (map value type); recurse so dynamic
	// values inside a map still collapse.
	if ap, ok := m["additionalProperties"].(map[string]any); ok {
		CollapseDynamicNodes(ap)
	}
}

// isDynamicNode reports whether m is an open-ended / dynamic JSON node with no
// fixed, closed type that a strict provider can express.
func isDynamicNode(m map[string]any) bool {
	_, hasRef := m["$ref"]
	if hasRef {
		return false
	}
	for _, k := range []string{"anyOf", "oneOf", "allOf"} {
		if _, ok := m[k]; ok {
			return false
		}
	}

	t, hasType := m["type"]
	if !hasType {
		// Typeless node: google.protobuf.Value, google.protobuf.Any.value.
		return true
	}

	_, hasProps := m["properties"]
	switch ts := t.(type) {
	case string:
		// Open object: google.protobuf.Struct (additionalProperties:true, no
		// fixed properties). Maps render additionalProperties as a schema, not
		// true, so they are not collapsed here.
		if ts == "object" && !hasProps && m["additionalProperties"] == true {
			return true
		}
		// Untyped array: google.protobuf.ListValue ({"items":{}}).
		if ts == "array" {
			if it, ok := m["items"].(map[string]any); ok && len(it) == 0 {
				return true
			}
		}
	}
	return false
}

// walk applies fn to every schema-object node reachable from node, recursing
// through the standard subschema-bearing keywords.
func walk(node any, fn func(map[string]any)) {
	switch n := node.(type) {
	case map[string]any:
		fn(n)
		if props, ok := n["properties"].(map[string]any); ok {
			for _, v := range props {
				walk(v, fn)
			}
		}
		for _, k := range []string{"$defs", "definitions", "patternProperties"} {
			if defs, ok := n[k].(map[string]any); ok {
				for _, v := range defs {
					walk(v, fn)
				}
			}
		}
		for _, k := range []string{"anyOf", "oneOf", "allOf", "prefixItems"} {
			if arr, ok := n[k].([]any); ok {
				for _, v := range arr {
					walk(v, fn)
				}
			}
		}
		switch it := n["items"].(type) {
		case map[string]any:
			walk(it, fn)
		case []any:
			for _, v := range it {
				walk(v, fn)
			}
		}
		if ap, ok := n["additionalProperties"].(map[string]any); ok {
			walk(ap, fn)
		}
	case []any:
		for _, v := range n {
			walk(v, fn)
		}
	}
}
