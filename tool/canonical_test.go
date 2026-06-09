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

package tool_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

func TestArgumentsHash_StableAcrossKeyOrder(t *testing.T) {
	t.Parallel()

	a, err := tool.ArgumentsHash(json.RawMessage(`{"a":1,"b":2,"c":3}`))
	require.NoError(t, err)

	b, err := tool.ArgumentsHash(json.RawMessage(`{"c":3,"b":2,"a":1}`))
	require.NoError(t, err)

	assert.Equal(t, a, b, "argument hash must be stable across key order")
}

func TestArgumentsHash_NestedObjects(t *testing.T) {
	t.Parallel()

	a, err := tool.ArgumentsHash(json.RawMessage(`{"outer":{"x":1,"y":2}}`))
	require.NoError(t, err)

	b, err := tool.ArgumentsHash(json.RawMessage(`{"outer":{"y":2,"x":1}}`))
	require.NoError(t, err)

	assert.Equal(t, a, b)
}

func TestArgumentsHash_DistinguishesDifferentInputs(t *testing.T) {
	t.Parallel()

	a, err := tool.ArgumentsHash(json.RawMessage(`{"x":1}`))
	require.NoError(t, err)

	b, err := tool.ArgumentsHash(json.RawMessage(`{"x":2}`))
	require.NoError(t, err)

	assert.NotEqual(t, a, b)
}

func TestArgumentsHash_NumberNormalization(t *testing.T) {
	t.Parallel()

	// 1 and 1.0 should hash identically: JCS normalizes integer floats
	// to integer form.
	a, err := tool.ArgumentsHash(json.RawMessage(`{"x":1}`))
	require.NoError(t, err)

	b, err := tool.ArgumentsHash(json.RawMessage(`{"x":1.0}`))
	require.NoError(t, err)

	assert.Equal(t, a, b, "1 and 1.0 must canonicalize the same way")
}

func TestArgumentsHash_NullAndEmpty(t *testing.T) {
	t.Parallel()

	hashNil, err := tool.ArgumentsHash(nil)
	require.NoError(t, err)

	hashNull, err := tool.ArgumentsHash(json.RawMessage(`null`))
	require.NoError(t, err)

	assert.Equal(t, hashNil, hashNull, "nil and null JSON should hash the same")
}

func TestArgumentsHash_Arrays(t *testing.T) {
	t.Parallel()

	a, err := tool.ArgumentsHash(json.RawMessage(`[1,2,3]`))
	require.NoError(t, err)

	// Arrays preserve order.
	b, err := tool.ArgumentsHash(json.RawMessage(`[3,2,1]`))
	require.NoError(t, err)

	assert.NotEqual(t, a, b, "array order must affect the hash")
}

func TestArgumentsHash_InvalidJSON(t *testing.T) {
	t.Parallel()

	_, err := tool.ArgumentsHash(json.RawMessage(`{not json`))
	require.Error(t, err)
}
