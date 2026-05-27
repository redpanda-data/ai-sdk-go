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

package builtin

import (
	"encoding/json"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
)

// MustParseSchema parses a JSON Schema literal into *jsonschema.Schema once
// at package init time. Panics if the input is not valid JSON or not a valid
// schema — intended for compile-time-constant schema strings inside builtin
// tools where parse errors are programmer bugs.
func MustParseSchema(raw string) *jsonschema.Schema {
	var s jsonschema.Schema

	if err := json.Unmarshal([]byte(raw), &s); err != nil {
		panic(fmt.Errorf("builtin: parse schema: %w", err)) //nolint:forbidigo // compile-time-constant schema literals
	}

	return &s
}
