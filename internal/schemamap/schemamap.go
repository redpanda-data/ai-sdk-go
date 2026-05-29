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

// Package schemamap converts a typed tool parameter schema into the
// map[string]any form that provider request mappers mutate before sending.
package schemamap

import (
	"encoding/json"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ToMap renders a tool parameter schema as a JSON object map for a provider
// request body. It returns a nil map when the schema carries no object
// constraints, leaving each provider free to apply its own empty-schema
// handling (some send an explicit empty-object schema, others omit it).
//
// A nil schema yields a nil map. JSON Schema also permits the boolean schemas
// `true` (allow anything) and `false` (allow nothing), and jsonschema-go encodes
// an empty Schema{} as the bare literal `true`. Because provider tool APIs
// expect an object, a top-level boolean or null schema is treated as "no
// constraints" and also yields a nil map, rather than failing to decode into
// map[string]any (which is what a naive json.Unmarshal of `true`/`false` does).
func ToMap(s *llm.Schema) (map[string]any, error) {
	if s == nil {
		return nil, nil
	}

	b, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}

	if !isJSONObject(b) {
		return nil, nil
	}

	m := map[string]any{}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}

	return m, nil
}

// isJSONObject reports whether b is a JSON object, i.e. its first non-whitespace
// byte is '{'. Boolean, null, string, number and array schemas return false.
func isJSONObject(b []byte) bool {
	for _, c := range b {
		switch c {
		case ' ', '\t', '\n', '\r':
			continue
		case '{':
			return true
		default:
			return false
		}
	}

	return false
}
