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

package openaicompat

import (
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestMapMessages_EmptyAssistantContentIsDropped documents the openaicompat
// analogue of the anthropic empty-content guard. An assistant turn with zero
// content parts (e.g. a max_tokens cut whose only block was a dropped partial
// tool_use) must never map to an invalid content-less assistant message. Unlike
// Anthropic — which requires a substituted placeholder to preserve strict
// role alternation — the Chat Completion API tolerates the resulting sequence,
// so the mapper simply omits the turn. This test locks in that no empty/invalid
// assistant message reaches the wire.
func TestMapMessages_EmptyAssistantContentIsDropped(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	model, err := provider.NewModel("gpt-4o-mini")
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok)

	req := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("run the tools")}},
		{Role: llm.RoleAssistant, Content: []llm.Part{}}, // poisoned truncated turn
		{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("continue")}},
	}

	messages, err := m.requestMapper.mapMessages(req)
	require.NoError(t, err)

	// The empty assistant turn is dropped; the two user turns survive.
	require.Len(t, messages, 2)

	for i, msg := range messages {
		if msg.OfAssistant != nil {
			t.Fatalf("message %d unexpectedly mapped to an assistant message", i)
		}
	}

	// Sanity: both surviving messages are the user turns, with content.
	for i, msg := range messages {
		require.NotNil(t, msg.OfUser, "message %d should be a user message", i)
		require.NotEqual(t, openai.ChatCompletionUserMessageParamContentUnion{}, msg.OfUser.Content)
	}
}
