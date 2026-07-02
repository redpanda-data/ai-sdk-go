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
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestMapAssistantMessage_EmptyContentAsLastMessage covers the trailing-turn
// variant of the empty-content replay bug: the poisoned assistant turn is the
// LAST message in the request. Beyond rejecting empty content, Anthropic
// rejects a final assistant turn that ends in trailing whitespace, and some
// API/SDK versions strip a whitespace-only text block back to empty. The
// substituted placeholder must therefore be non-empty AND not end in
// whitespace, so the repaired turn is valid even as the final message.
func TestMapAssistantMessage_EmptyContentAsLastMessage(t *testing.T) {
	t.Parallel()

	m := newWireTestModel(t)

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("run the tools")}},
			// The poisoned turn as the FINAL message.
			{Role: llm.RoleAssistant, Content: []llm.Part{}},
		},
	}

	apiReq, err := m.requestMapper.ToProvider(req)
	require.NoError(t, err)

	require.Len(t, apiReq.Messages, 2)

	last := apiReq.Messages[len(apiReq.Messages)-1]
	require.Equal(t, anthropic.BetaMessageParamRoleAssistant, last.Role)
	require.NotEmpty(t, last.Content, "final assistant turn mapped to empty content")

	lastBlock := last.Content[len(last.Content)-1]
	require.NotNil(t, lastBlock.OfText, "final block must be a text block")

	text := lastBlock.OfText.Text
	require.NotEmpty(t, text, "substituted text block must be non-empty")
	assert.Equal(t, text, strings.TrimRight(text, " \t\r\n"),
		"final assistant text must not end in whitespace — Anthropic rejects a trailing-whitespace final turn")
}

// TestMapAssistantMessage_ConsecutiveEmptyContentTurns proves the guard repairs
// every empty assistant turn independently, not just the first: two poisoned
// turns in a row must both map to non-empty content.
func TestMapAssistantMessage_ConsecutiveEmptyContentTurns(t *testing.T) {
	t.Parallel()

	m := newWireTestModel(t)

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("start")}},
			{Role: llm.RoleAssistant, Content: []llm.Part{}},
			{Role: llm.RoleAssistant, Content: []llm.Part{}},
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("continue")}},
		},
	}

	apiReq, err := m.requestMapper.ToProvider(req)
	require.NoError(t, err)

	require.Len(t, apiReq.Messages, 4, "all four history messages must map through")

	for i, msg := range apiReq.Messages {
		assert.NotEmpty(t, msg.Content,
			"message %d (role %s) mapped to empty content", i, msg.Role)
	}

	for _, i := range []int{1, 2} {
		msg := apiReq.Messages[i]
		require.Equal(t, anthropic.BetaMessageParamRoleAssistant, msg.Role)
		require.Len(t, msg.Content, 1, "repaired turn %d must carry exactly one block", i)
		require.NotNil(t, msg.Content[0].OfText, "repaired block %d must be a text block", i)
		assert.NotEmpty(t, msg.Content[0].OfText.Text, "substituted block %d must be non-empty", i)
	}
}
