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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMarshalPart_InterfaceNil verifies that an interface-nil Part marshals
// to JSON null without panicking.
func TestMarshalPart_InterfaceNil(t *testing.T) {
	t.Parallel()

	data, err := MarshalPart(nil)
	require.NoError(t, err)
	assert.JSONEq(t, "null", string(data))
}

// TestMarshalPart_TypedNil verifies that a typed-nil pointer in a Part-typed
// variable marshals as JSON null rather than panicking or emitting a
// zero-value envelope. This matches the interface-nil semantics callers
// expect when they assign a nil concrete pointer into a Part.
func TestMarshalPart_TypedNil(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		part Part
	}{
		{"TextPart", (*TextPart)(nil)},
		{"ToolRequestPart", (*ToolRequestPart)(nil)},
		{"ToolResponsePart", (*ToolResponsePart)(nil)},
		{"ReasoningPart", (*ReasoningPart)(nil)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			data, err := MarshalPart(tc.part)
			require.NoError(t, err)
			assert.JSONEq(t, "null", string(data))
		})
	}
}

// TestMarshalPart_RoundTrip verifies that each concrete Part type
// round-trips through MarshalPart/UnmarshalPart preserving its content.
func TestMarshalPart_RoundTrip(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		part Part
	}{
		{
			name: "TextPart",
			part: &TextPart{Text: "hello", Metadata: map[string]any{"k": "v"}},
		},
		{
			name: "ToolRequestPart",
			part: &ToolRequestPart{
				ID:        "req-1",
				Name:      "search",
				Arguments: json.RawMessage(`{"q":"ai"}`),
			},
		},
		{
			name: "ToolResponsePart_success",
			part: &ToolResponsePart{
				ID:     "req-1",
				Name:   "search",
				Result: json.RawMessage(`{"ok":true}`),
			},
		},
		{
			name: "ToolResponsePart_error",
			part: &ToolResponsePart{
				ID:     "req-1",
				Name:   "search",
				Result: json.RawMessage(`null`),
				Error:  "timeout",
			},
		},
		{
			name: "ReasoningPart",
			part: &ReasoningPart{ID: "sig", Text: "thinking..."},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			data, err := MarshalPart(tc.part)
			require.NoError(t, err)

			decoded, err := UnmarshalPart(data)
			require.NoError(t, err)
			assert.Equal(t, tc.part, decoded)
		})
	}
}

// TestUnmarshalPart_Null verifies that null and empty input produce a nil
// Part with no error.
func TestUnmarshalPart_Null(t *testing.T) {
	t.Parallel()

	cases := []string{"null", ""}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			t.Parallel()

			p, err := UnmarshalPart([]byte(in))
			require.NoError(t, err)
			assert.Nil(t, p)
		})
	}
}

// TestClonePart_DeepCopy verifies that ClonePart returns a value that does
// not share any mutable state with the original.
func TestClonePart_DeepCopy(t *testing.T) {
	t.Parallel()

	const mutated = "mutated"

	t.Run("TextPart", func(t *testing.T) {
		t.Parallel()

		orig := &TextPart{Text: "hi", Metadata: map[string]any{"k": "v"}}

		clone, ok := ClonePart(orig).(*TextPart)
		require.True(t, ok)

		clone.Text = mutated
		clone.Metadata["k"] = mutated

		assert.Equal(t, "hi", orig.Text)
		assert.Equal(t, "v", orig.Metadata["k"])
	})

	t.Run("ToolRequestPart", func(t *testing.T) {
		t.Parallel()

		args := json.RawMessage(`{"q":"ai"}`)
		orig := &ToolRequestPart{ID: "id", Name: "n", Arguments: args, Metadata: map[string]any{"k": "v"}}

		clone, ok := ClonePart(orig).(*ToolRequestPart)
		require.True(t, ok)

		clone.Arguments[0] = '['
		clone.Metadata["k"] = mutated

		assert.Equal(t, byte('{'), orig.Arguments[0])
		assert.Equal(t, "v", orig.Metadata["k"])
	})

	t.Run("ToolResponsePart", func(t *testing.T) {
		t.Parallel()

		orig := &ToolResponsePart{
			ID:       "id",
			Name:     "n",
			Result:   json.RawMessage(`{"ok":true}`),
			Metadata: map[string]any{"k": "v"},
		}

		clone, ok := ClonePart(orig).(*ToolResponsePart)
		require.True(t, ok)

		clone.Result[0] = '['
		clone.Metadata["k"] = mutated

		assert.Equal(t, byte('{'), orig.Result[0])
		assert.Equal(t, "v", orig.Metadata["k"])
	})

	t.Run("ReasoningPart", func(t *testing.T) {
		t.Parallel()

		orig := &ReasoningPart{ID: "id", Text: "t", Metadata: map[string]any{"k": "v"}}

		clone, ok := ClonePart(orig).(*ReasoningPart)
		require.True(t, ok)

		clone.Text = mutated
		clone.Metadata["k"] = mutated

		assert.Equal(t, "t", orig.Text)
		assert.Equal(t, "v", orig.Metadata["k"])
	})
}

// TestClonePart_Nil verifies that nil and typed-nil inputs yield nil
// outputs without panicking.
func TestClonePart_Nil(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		in   Part
	}{
		{"interface-nil", nil},
		{"typed-nil TextPart", (*TextPart)(nil)},
		{"typed-nil ToolRequestPart", (*ToolRequestPart)(nil)},
		{"typed-nil ToolResponsePart", (*ToolResponsePart)(nil)},
		{"typed-nil ReasoningPart", (*ReasoningPart)(nil)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			assert.Nil(t, ClonePart(tc.in))
		})
	}
}

// TestCloneMessage_Independence verifies that mutating a cloned Message
// does not affect the original.
func TestCloneMessage_Independence(t *testing.T) {
	t.Parallel()

	orig := Message{
		Role: RoleAssistant,
		Content: []Part{
			&TextPart{Text: "a"},
			&ToolRequestPart{ID: "1", Name: "tool", Arguments: json.RawMessage(`{}`)},
		},
	}

	clone := CloneMessage(orig)

	cloneText, ok := clone.Content[0].(*TextPart)
	require.True(t, ok)

	cloneText.Text = "mutated"

	origText, ok := orig.Content[0].(*TextPart)
	require.True(t, ok)

	assert.Equal(t, "a", origText.Text)
	assert.NotSame(t, origText, cloneText)
}
