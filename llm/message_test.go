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

func TestNewMessage(t *testing.T) {
	t.Parallel()

	toolReq := &llm.ToolRequest{
		ID:        "call_1",
		Name:      "search",
		Arguments: json.RawMessage(`{"query":"test"}`),
	}
	toolResp := &llm.ToolResponse{
		ID:     "call_1",
		Name:   "search",
		Result: json.RawMessage(`{"results":["item1"]}`),
	}

	tests := []struct {
		name      string
		role      llm.MessageRole
		parts     []*llm.Part
		wantParts int
		wantText  string
	}{
		{
			name:      "user message with text",
			role:      llm.RoleUser,
			parts:     []*llm.Part{llm.NewTextPart("Hello")},
			wantParts: 1,
			wantText:  "Hello",
		},
		{
			name:      "system message",
			role:      llm.RoleSystem,
			parts:     []*llm.Part{llm.NewTextPart("You are helpful")},
			wantParts: 1,
			wantText:  "You are helpful",
		},
		{
			name: "assistant with text and tool request",
			role: llm.RoleAssistant,
			parts: []*llm.Part{
				llm.NewTextPart("Searching..."),
				llm.NewToolRequestPart(toolReq),
			},
			wantParts: 2,
			wantText:  "Searching...",
		},
		{
			name:      "tool response message",
			role:      llm.RoleUser,
			parts:     []*llm.Part{llm.NewToolResponsePart(toolResp)},
			wantParts: 1,
			wantText:  "",
		},
		{
			name:      "empty message",
			role:      llm.RoleUser,
			parts:     []*llm.Part{},
			wantParts: 0,
			wantText:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			msg := llm.NewMessage(tt.role, tt.parts...)

			assert.Equal(t, tt.role, msg.Role)
			require.Len(t, msg.Content, tt.wantParts)
			assert.Equal(t, tt.wantText, msg.TextContent())
		})
	}
}
