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

package genai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMarshalMessages(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		msgs []Message
		want string
	}{
		{
			name: "single text message",
			msgs: []Message{
				{Role: RoleUser, Parts: []Part{{Type: PartTypeText, Content: "hello"}}},
			},
			want: `[{"role":"user","parts":[{"type":"text","content":"hello"}]}]`,
		},
		{
			name: "empty slice",
			msgs: []Message{},
			want: "[]",
		},
		{
			name: "nil slice",
			msgs: nil,
			want: "null",
		},
		{
			name: "message with tool call",
			msgs: []Message{
				{
					Role: RoleAssistant,
					Parts: []Part{
						{Type: PartTypeToolCall, Name: "search", ID: "call_1", Arguments: json.RawMessage(`{"q":"test"}`)},
					},
					FinishReason: "tool_call",
				},
			},
			want: `[{"role":"assistant","parts":[{"type":"tool_call","name":"search","id":"call_1","arguments":{"q":"test"}}],"finish_reason":"tool_call"}]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := MarshalMessages(tt.msgs)
			assert.JSONEq(t, tt.want, got)
		})
	}
}

func TestMarshalMessage(t *testing.T) {
	t.Parallel()

	msg := Message{Role: RoleAssistant, Parts: []Part{{Type: PartTypeText, Content: "hi"}}, FinishReason: "stop"}
	got := MarshalMessage(msg)
	assert.JSONEq(t, `[{"role":"assistant","parts":[{"type":"text","content":"hi"}],"finish_reason":"stop"}]`, got)
}

func TestValidateMessages(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		msgs    []Message
		wantErr string
	}{
		{
			name: "valid messages",
			msgs: []Message{
				{Role: RoleUser, Parts: []Part{{Type: PartTypeText, Content: "hello"}}},
				{Role: RoleAssistant, Parts: []Part{{Type: PartTypeText, Content: "hi"}}},
			},
		},
		{
			name: "all valid roles",
			msgs: []Message{
				{Role: RoleSystem, Parts: []Part{{Type: PartTypeText, Content: "a"}}},
				{Role: RoleUser, Parts: []Part{{Type: PartTypeText, Content: "b"}}},
				{Role: RoleAssistant, Parts: []Part{{Type: PartTypeText, Content: "c"}}},
				{Role: RoleTool, Parts: []Part{{Type: PartTypeToolCallResponse, ID: "1", Response: json.RawMessage(`null`)}}},
			},
		},
		{
			name: "all valid part types",
			msgs: []Message{
				{Role: RoleAssistant, Parts: []Part{
					{Type: PartTypeText, Content: "thinking..."},
					{Type: PartTypeReasoning, Content: "step 1"},
					{Type: PartTypeToolCall, Name: "f", ID: "1"},
					{Type: PartTypeToolCallResponse, ID: "1", Response: json.RawMessage(`null`)},
				}},
			},
		},
		{
			name:    "invalid role",
			msgs:    []Message{{Role: "narrator", Parts: []Part{{Type: PartTypeText, Content: "x"}}}},
			wantErr: `message[0]: invalid role "narrator"`,
		},
		{
			name:    "empty parts",
			msgs:    []Message{{Role: RoleUser, Parts: []Part{}}},
			wantErr: "message[0]: parts must not be empty",
		},
		{
			name:    "unknown part type",
			msgs:    []Message{{Role: RoleUser, Parts: []Part{{Type: "image"}}}},
			wantErr: `message[0].parts[0]: invalid type "image"`,
		},
		{
			name: "error on second message",
			msgs: []Message{
				{Role: RoleUser, Parts: []Part{{Type: PartTypeText, Content: "ok"}}},
				{Role: "bogus", Parts: []Part{{Type: PartTypeText, Content: "x"}}},
			},
			wantErr: `message[1]: invalid role "bogus"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := ValidateMessages(tt.msgs)

			if tt.wantErr == "" {
				require.NoError(t, err)
			} else {
				require.EqualError(t, err, tt.wantErr)
			}
		})
	}
}
