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

// Package testschema provides a small helper for parsing JSON Schema
// literals in test fixtures.
package testschema

import (
	"encoding/json"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
)

// MustParse parses a JSON Schema literal. Panics on failure — inputs are
// compile-time strings in test code.
func MustParse(raw string) *jsonschema.Schema {
	s := &jsonschema.Schema{}
	if err := json.Unmarshal([]byte(raw), s); err != nil {
		panic(fmt.Errorf("testschema: parse: %w", err)) //nolint:forbidigo // test helper for compile-time-constant fixtures
	}

	return s
}
