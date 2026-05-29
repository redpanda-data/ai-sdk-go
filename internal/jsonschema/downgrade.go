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
// individual LLM providers accept as a tool input_schema. MCP servers
// legitimately publish full JSON Schema; per the MCP ecosystem the client is
// responsible for adapting it per provider (the same way LangChain, the OpenAI
// Agents SDK and the Vercel AI SDK do).
//
// Only transforms common to more than one provider live here. Provider-specific
// rules (e.g. OpenAI's keyword allowlist) live in the respective provider
// package and use Walk for traversal.
package jsonschema

import (
	"bytes"
	"encoding/json"
)

// DeepCopy returns a JSON round-tripped deep copy of a schema, so adapters can
// mutate freely without touching the caller's input.
func DeepCopy(m map[string]any) (map[string]any, error) {
	var buf bytes.Buffer

	if err := json.NewEncoder(&buf).Encode(m); err != nil {
		return nil, err
	}

	var cp map[string]any
	if err := json.NewDecoder(&buf).Decode(&cp); err != nil {
		return nil, err
	}

	return cp, nil
}

// Walk applies fn to every schema-object node reachable from node, recursing
// through the standard subschema-bearing keywords (properties, $defs,
// definitions, patternProperties, anyOf/oneOf/allOf, prefixItems, items,
// additionalProperties). fn runs on a node before its children, so a pass that
// deletes a keyword also prunes that keyword's subtree from the walk.
func Walk(node any, fn func(map[string]any)) {
	switch n := node.(type) {
	case map[string]any:
		fn(n)

		if props, ok := n["properties"].(map[string]any); ok {
			for _, v := range props {
				Walk(v, fn)
			}
		}

		for _, k := range []string{"$defs", "definitions", "patternProperties"} {
			if defs, ok := n[k].(map[string]any); ok {
				for _, v := range defs {
					Walk(v, fn)
				}
			}
		}

		for _, k := range []string{"anyOf", "oneOf", "allOf", "prefixItems"} {
			if arr, ok := n[k].([]any); ok {
				for _, v := range arr {
					Walk(v, fn)
				}
			}
		}

		switch it := n["items"].(type) {
		case map[string]any:
			Walk(it, fn)
		case []any:
			for _, v := range it {
				Walk(v, fn)
			}
		}

		if ap, ok := n["additionalProperties"].(map[string]any); ok {
			Walk(ap, fn)
		}
	case []any:
		for _, v := range n {
			Walk(v, fn)
		}
	}
}

// CollapseDynamicNodes rewrites schema nodes that describe open-ended / dynamic
// JSON (no fixed type) into a single JSON-encoded string node. OpenAI strict and
// Anthropic both reject typeless or open-object schemas as a tool input_schema
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
	if _, hasRef := m["$ref"]; hasRef {
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

	ts, ok := t.(string)
	if !ok {
		return false
	}

	_, hasProps := m["properties"]

	// Open object: google.protobuf.Struct (additionalProperties:true, no fixed
	// properties). Maps render additionalProperties as a schema, not true, so
	// they are not collapsed here.
	if ts == "object" && !hasProps && m["additionalProperties"] == true {
		return true
	}

	// Untyped array: google.protobuf.ListValue ({"items":{}}).
	if ts == "array" {
		if it, ok := m["items"].(map[string]any); ok && len(it) == 0 {
			return true
		}
	}

	return false
}
