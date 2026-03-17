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
	"bytes"
	"encoding/json"
)

// SchemaMapper transforms standard JSON Schemas to Anthropic-compatible schemas.
// Anthropic uses standard JSON Schema format, so this is simpler than OpenAI's requirements.
type SchemaMapper struct{}

// NewSchemaMapper creates a new SchemaMapper for schema transformations.
func NewSchemaMapper() *SchemaMapper { return &SchemaMapper{} }

// AdaptSchemaForAnthropic returns a transformed deep copy, never mutating the input.
// Currently, Anthropic accepts standard JSON Schema, so minimal transformation is needed.
func (*SchemaMapper) AdaptSchemaForAnthropic(schema map[string]any) map[string]any {
	cp, err := deepCopyMap(schema)
	if err != nil {
		return schema
	}

	// Anthropic uses standard JSON Schema format
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
