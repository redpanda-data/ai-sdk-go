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

package llm

import (
	"encoding/json"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
)

// Schema is a JSON Schema describing the parameters of a tool.
//
// It is a type alias for jsonschema.Schema so that SDK consumers can build and
// inspect tool parameter schemas using only the ai-sdk-go module, without
// importing github.com/google/jsonschema-go directly. The alias keeps the
// underlying type identical, so a *Schema is interchangeable with a
// *jsonschema.Schema at every call site.
type Schema = jsonschema.Schema

// SchemaFor generates a Schema from the Go type T via reflection. It is a thin
// wrapper over jsonschema.For so callers do not need to import jsonschema-go.
//
// Struct fields map to properties using their JSON tags; non-omitempty fields
// become required. A `jsonschema:"..."` field tag sets the property description.
func SchemaFor[T any]() (*Schema, error) {
	return jsonschema.For[T](nil)
}

// MustSchemaFor is like SchemaFor but panics if schema generation fails.
// Intended for package-level vars where T is a compile-time-known struct and a
// failure is a programmer error rather than a runtime condition.
func MustSchemaFor[T any]() *Schema {
	s, err := jsonschema.For[T](nil)
	if err != nil {
		var zero T
		panic(fmt.Errorf("llm: generate schema for %T: %w", zero, err)) //nolint:forbidigo // Must* helper for compile-time-known types
	}

	return s
}

// MustSchema parses a JSON Schema literal into a *Schema, panicking on invalid
// input. Intended for compile-time-constant schema strings (for example tool
// definitions declared as package-level vars) where a parse failure is a
// programmer error. For schemas built at runtime, unmarshal into a Schema and
// handle the error instead.
func MustSchema(raw string) *Schema {
	s := &Schema{}
	if err := json.Unmarshal([]byte(raw), s); err != nil {
		panic(fmt.Errorf("llm: parse schema: %w", err)) //nolint:forbidigo // Must* helper for compile-time-constant schema literals
	}

	return s
}
