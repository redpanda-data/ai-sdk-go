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

package fakellm_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

func TestValidateConversation(t *testing.T) {
	t.Parallel()

	user := func(text string) llm.Message { return llm.NewMessage(llm.RoleUser, llm.NewTextPart(text)) }
	call := func(id string) llm.Message {
		return llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart(id, "fetch", json.RawMessage(`{}`)))
	}
	result := func(id string, payload string) llm.Message {
		return llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(id, "fetch", json.RawMessage(payload), false))
	}

	tests := []struct {
		name     string
		messages []llm.Message
		wantErr  string
	}{
		{
			name:     "plain exchange",
			messages: []llm.Message{user("hi"), llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("hello"))},
		},
		{
			name:     "complete tool round trip",
			messages: []llm.Message{user("go"), call("c1"), result("c1", `{"ok":true}`), user("thanks")},
		},
		{
			name:     "pending calls on the final message are legal",
			messages: []llm.Message{user("go"), call("c1")},
		},
		{
			name:     "empty message",
			messages: []llm.Message{user("go"), {Role: llm.RoleAssistant}},
			wantErr:  "has no content",
		},
		{
			name:     "orphaned tool result",
			messages: []llm.Message{user("go"), result("ghost", `{}`)},
			wantErr:  "orphaned tool result",
		},
		{
			name:     "tool call answered by nothing before the next assistant message",
			messages: []llm.Message{user("go"), call("c1"), llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("moving on"))},
			wantErr:  "no tool result",
		},
		{
			name:     "tool call never answered",
			messages: []llm.Message{user("go"), call("c1"), user("hello?")},
			wantErr:  "no tool result",
		},
		{
			name:     "invalid JSON result",
			messages: []llm.Message{user("go"), call("c1"), result("c1", `{"truncated`)},
			wantErr:  "not valid JSON",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := fakellm.ValidateConversation(tt.messages)
			if tt.wantErr == "" {
				assert.NoError(t, err)
			} else {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
			}
		})
	}
}

// TestFakeRejectsInvalidConversation pins that the fake enforces the
// validator on its request path, as a provider would.
func TestFakeRejectsInvalidConversation(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondText("nope")

	req := &llm.Request{Messages: []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("ghost", "fetch", json.RawMessage(`{}`), false)),
	}}

	_, err := model.Generate(t.Context(), req)

	require.Error(t, err)
	require.ErrorIs(t, err, llm.ErrInvalidInput)
	require.ErrorIs(t, err, llm.ErrAPICall)
	assert.NotErrorIs(t, err, llm.ErrContextOverflow)
}
