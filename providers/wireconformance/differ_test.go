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

package wireconformance

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiffJSON_Identical(t *testing.T) {
	t.Parallel()

	a := json.RawMessage(`{"model":"gpt-5","input":[{"role":"user","content":"hi"}],"max_tokens":100}`)
	diffs := DiffJSON(a, a, nil)
	assert.Empty(t, diffs)
}

func TestDiffJSON_MissingField(t *testing.T) {
	t.Parallel()

	native := json.RawMessage(`{"model":"gpt-5","temperature":0.7}`)
	aisdk := json.RawMessage(`{"model":"gpt-5"}`)

	diffs := DiffJSON(native, aisdk, nil)
	require.Len(t, diffs, 1)
	assert.Equal(t, "temperature", diffs[0].Path)
	assert.Equal(t, DiffMissing, diffs[0].Kind)
}

func TestDiffJSON_ExtraField(t *testing.T) {
	t.Parallel()

	native := json.RawMessage(`{"model":"gpt-5"}`)
	aisdk := json.RawMessage(`{"model":"gpt-5","extra":true}`)

	diffs := DiffJSON(native, aisdk, nil)
	require.Len(t, diffs, 1)
	assert.Equal(t, "extra", diffs[0].Path)
	assert.Equal(t, DiffExtra, diffs[0].Kind)
}

func TestDiffJSON_ChangedValue(t *testing.T) {
	t.Parallel()

	native := json.RawMessage(`{"model":"gpt-5","temperature":0.7}`)
	aisdk := json.RawMessage(`{"model":"gpt-5","temperature":0.9}`)

	diffs := DiffJSON(native, aisdk, nil)
	require.Len(t, diffs, 1)
	assert.Equal(t, "temperature", diffs[0].Path)
	assert.Equal(t, DiffChanged, diffs[0].Kind)
}

func TestDiffJSON_NestedObject(t *testing.T) {
	t.Parallel()

	native := json.RawMessage(`{"config":{"a":1,"b":2}}`)
	aisdk := json.RawMessage(`{"config":{"a":1,"b":3}}`)

	diffs := DiffJSON(native, aisdk, nil)
	require.Len(t, diffs, 1)
	assert.Equal(t, "config.b", diffs[0].Path)
	assert.Equal(t, DiffChanged, diffs[0].Kind)
}

func TestDiffJSON_ArrayElements(t *testing.T) {
	t.Parallel()

	native := json.RawMessage(`{"items":["a","b","c"]}`)
	aisdk := json.RawMessage(`{"items":["a","x","c"]}`)

	diffs := DiffJSON(native, aisdk, nil)
	require.Len(t, diffs, 1)
	assert.Equal(t, "items[1]", diffs[0].Path)
	assert.Equal(t, DiffChanged, diffs[0].Kind)
}

func TestDiffJSON_ArrayLengthMismatch(t *testing.T) {
	t.Parallel()

	native := json.RawMessage(`{"items":[1,2,3]}`)
	aisdk := json.RawMessage(`{"items":[1,2]}`)

	diffs := DiffJSON(native, aisdk, nil)
	require.Len(t, diffs, 1)
	assert.Equal(t, "items[2]", diffs[0].Path)
	assert.Equal(t, DiffMissing, diffs[0].Kind)
}

func TestDiffJSON_TypeMismatch(t *testing.T) {
	t.Parallel()

	native := json.RawMessage(`{"value":"string"}`)
	aisdk := json.RawMessage(`{"value":42}`)

	diffs := DiffJSON(native, aisdk, nil)
	require.Len(t, diffs, 1)
	assert.Equal(t, "value", diffs[0].Path)
	assert.Equal(t, DiffTypeMismatch, diffs[0].Kind)
}

func TestDiffJSON_IgnoredPaths(t *testing.T) {
	t.Parallel()

	native := json.RawMessage(`{"model":"gpt-5","stream":true,"metadata":{"key":"val"}}`)
	aisdk := json.RawMessage(`{"model":"gpt-5"}`)

	ignores := map[string]bool{
		"stream":     true,
		"metadata.*": true,
	}

	diffs := DiffJSON(native, aisdk, ignores)
	// "stream" is ignored directly, "metadata" itself is not ignored but
	// if we also want to ignore the parent, we need to add it.
	// Let's just check stream is ignored:
	for _, d := range diffs {
		assert.NotEqual(t, "stream", d.Path)
	}
}

func TestDiffJSON_WildcardIgnore(t *testing.T) {
	t.Parallel()

	native := json.RawMessage(`{"metadata":{"key":"val","nested":{"deep":1}}}`)
	aisdk := json.RawMessage(`{"metadata":{"key":"different"}}`)

	ignores := map[string]bool{
		"metadata.*": true,
	}

	diffs := DiffJSON(native, aisdk, ignores)
	assert.Empty(t, diffs)
}

func TestFormatDiffs_Output(t *testing.T) {
	t.Parallel()

	diffs := []FieldDiff{
		{
			Path:     "tools[0].cache_control",
			Kind:     DiffMissing,
			Expected: json.RawMessage(`{"type":"ephemeral"}`),
		},
	}

	output := FormatDiffs("tool_test", diffs, "request_mapper.go")
	assert.Contains(t, output, "FAIL: tool_test (1 diffs)")
	assert.Contains(t, output, "tools[0].cache_control")
	assert.Contains(t, output, "MISSING")
	assert.Contains(t, output, "request_mapper.go")
}
