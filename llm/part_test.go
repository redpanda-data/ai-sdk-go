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

package llm_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestMarshalPart_TypedNil(t *testing.T) {
	t.Parallel()

	// A typed-nil pointer assigned to the Part interface must marshal as
	// JSON null rather than panic. Verifies the explicit nil checks inside
	// MarshalPart for each concrete type.
	cases := []struct {
		name string
		part llm.Part
	}{
		{"text", (*llm.TextPart)(nil)},
		{"tool_request", (*llm.ToolRequestPart)(nil)},
		{"tool_response", (*llm.ToolResponsePart)(nil)},
		{"reasoning", (*llm.ReasoningPart)(nil)},
		{"plain nil", nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			out, err := llm.MarshalPart(tc.part)
			require.NoError(t, err)
			assert.Equal(t, "null", string(out))
		})
	}
}

func TestMarshalPart_RoundTrip(t *testing.T) {
	t.Parallel()

	cases := []llm.Part{
		llm.NewTextPart("hello"),
		llm.NewToolRequestPart("call_1", "search", json.RawMessage(`{"q":"x"}`)),
		llm.NewToolResponsePart("call_1", "search", json.RawMessage(`{"ok":true}`)),
		&llm.ToolResponsePart{ID: "call_2", Name: "search", Result: json.RawMessage(`{"error":"boom"}`), IsError: true},
		&llm.ReasoningPart{Text: "thinking", Signature: "sig-1"},
	}

	for _, p := range cases {
		data, err := llm.MarshalPart(p)
		require.NoError(t, err)

		got, err := llm.UnmarshalPart(data)
		require.NoError(t, err)
		assert.IsType(t, p, got)
	}
}

func TestMessage_JSONRoundTrip(t *testing.T) {
	t.Parallel()

	original := llm.NewMessage(
		llm.RoleAssistant,
		llm.NewTextPart("hello"),
		llm.NewToolRequestPart("call_1", "search", json.RawMessage(`{"q":"x"}`)),
		&llm.ReasoningPart{Text: "thinking", Signature: "sig"},
	)

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded llm.Message

	require.NoError(t, json.Unmarshal(data, &decoded))
	assert.Equal(t, original.Role, decoded.Role)
	require.Len(t, decoded.Content, len(original.Content))

	for i := range original.Content {
		assert.IsType(t, original.Content[i], decoded.Content[i])
	}
}

func TestCloneMessage_IsDeep(t *testing.T) {
	t.Parallel()

	src := llm.NewMessage(
		llm.RoleUser,
		llm.NewTextPart("hi"),
		llm.NewToolRequestPart("id", "name", json.RawMessage(`{"a":1}`)),
	)

	dst := llm.CloneMessage(src)

	// Mutate src and verify dst is unaffected.
	srcText, ok := src.Content[0].(*llm.TextPart)
	require.True(t, ok)

	srcText.Text = "MUTATED"

	srcReq, ok := src.Content[1].(*llm.ToolRequestPart)
	require.True(t, ok)

	srcReq.Arguments[0] = 'X'

	dstText, ok := dst.Content[0].(*llm.TextPart)
	require.True(t, ok)
	assert.Equal(t, "hi", dstText.Text)

	dstReq, ok := dst.Content[1].(*llm.ToolRequestPart)
	require.True(t, ok)
	assert.Equal(t, byte('{'), dstReq.Arguments[0])
}

// TestPart_ValueLiteralDoesNotSatisfyInterface is a compile-time guard: the
// marker method has a pointer receiver, so passing a value-form literal where
// a Part is expected must not compile. This test exists as a doc reminder;
// uncomment the body to confirm.
//
//	func TestPart_ValueLiteralDoesNotSatisfyInterface(t *testing.T) {
//	    var _ llm.Part = llm.TextPart{} // intentionally fails to compile
//	}
