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
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestAdaptSchemaForAnthropic_CollapsesDynamicNodes asserts the Anthropic adapter
// collapses typeless / open-ended nodes (which Anthropic rejects as invalid
// draft-2020-12) to strings, while leaving typed fields and maps untouched.
func TestAdaptSchemaForAnthropic_CollapsesDynamicNodes(t *testing.T) {
	t.Parallel()
	in := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":   map[string]any{"type": "string"},
			"meta":   map[string]any{"type": "object", "additionalProperties": true}, // Struct
			"val":    map[string]any{"description": "a google.protobuf.Value"},       // Value (typeless)
			"labels": map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}},
		},
	}
	out := NewSchemaMapper().AdaptSchemaForAnthropic(in)
	props := out["properties"].(map[string]any)

	assert.Equal(t, "string", props["name"].(map[string]any)["type"])
	assert.Equal(t, "string", props["meta"].(map[string]any)["type"], "Struct collapses to string")
	assert.Equal(t, "string", props["val"].(map[string]any)["type"], "typeless Value collapses to string")
	// Maps are valid for Anthropic and must be preserved.
	assert.Equal(t, "object", props["labels"].(map[string]any)["type"])

	// Input not mutated.
	assert.Equal(t, true, in["properties"].(map[string]any)["meta"].(map[string]any)["additionalProperties"])
}
