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
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestCacheBreakpointOnToolResultTurn is the regression test for the shape that
// dominates an agentic loop: the last message is a user turn holding only
// tool_result blocks. Marking text blocks only left that turn unmarked, so the
// request went out with a breakpoint on the system blocks alone and the whole
// conversation was re-billed at full input price after every tool call.
func TestCacheBreakpointOnToolResultTurn(t *testing.T) {
	t.Parallel()

	apiReq := mapCachedRequest(t, []llm.Message{
		{
			Role:    llm.RoleSystem,
			Content: []llm.Part{llm.NewTextPart("You are a helpful assistant.")},
		},
		{
			Role:    llm.RoleUser,
			Content: []llm.Part{llm.NewTextPart("What is the weather?")},
		},
		{
			Role: llm.RoleAssistant,
			Content: []llm.Part{
				llm.NewToolRequestPart("call_1", "get_weather", json.RawMessage(`{"city":"Berlin"}`)),
			},
		},
		{
			Role: llm.RoleUser,
			Content: []llm.Part{
				llm.NewToolResponsePart("call_1", "get_weather", json.RawMessage(`{"temp_c":21}`), false),
			},
		},
	})

	last := apiReq.Messages[len(apiReq.Messages)-1]
	require.Len(t, last.Content, 1)
	require.NotNil(t, last.Content[0].OfToolResult, "last block must be the tool_result")

	assert.True(t, hasCacheMarker(last.Content[0].OfToolResult.CacheControl),
		"tool_result-only turn must carry the cache breakpoint; without it the conversation never caches")
}

// TestCacheBreakpointOnParallelToolResults covers parallel tool calls, where the
// turn carries several tool_result blocks. Exactly one breakpoint belongs on the
// turn — the last block — because Anthropic allows only four per request.
func TestCacheBreakpointOnParallelToolResults(t *testing.T) {
	t.Parallel()

	apiReq := mapCachedRequest(t, []llm.Message{
		{
			Role:    llm.RoleUser,
			Content: []llm.Part{llm.NewTextPart("Compare Berlin and Paris.")},
		},
		{
			Role: llm.RoleAssistant,
			Content: []llm.Part{
				llm.NewToolRequestPart("call_1", "get_weather", json.RawMessage(`{"city":"Berlin"}`)),
				llm.NewToolRequestPart("call_2", "get_weather", json.RawMessage(`{"city":"Paris"}`)),
			},
		},
		{
			Role: llm.RoleUser,
			Content: []llm.Part{
				llm.NewToolResponsePart("call_1", "get_weather", json.RawMessage(`{"temp_c":21}`), false),
				llm.NewToolResponsePart("call_2", "get_weather", json.RawMessage(`{"temp_c":24}`), false),
			},
		},
	})

	last := apiReq.Messages[len(apiReq.Messages)-1]
	require.Len(t, last.Content, 2)
	require.NotNil(t, last.Content[0].OfToolResult)
	require.NotNil(t, last.Content[1].OfToolResult)

	assert.False(t, hasCacheMarker(last.Content[0].OfToolResult.CacheControl),
		"only the final block of the turn gets a breakpoint")
	assert.True(t, hasCacheMarker(last.Content[1].OfToolResult.CacheControl),
		"final tool_result of a parallel batch must carry the breakpoint")
}

// TestCacheBreakpointSkipsThinkingBlock pins the one block type Anthropic
// rejects a breakpoint on. An assistant turn ending in a thinking block must
// fall back to the previous cacheable block rather than marking the thinking
// block (a 400) or marking nothing (no caching).
func TestCacheBreakpointSkipsThinkingBlock(t *testing.T) {
	t.Parallel()

	apiReq := mapCachedRequest(t, []llm.Message{
		{
			Role:    llm.RoleUser,
			Content: []llm.Part{llm.NewTextPart("Think about this.")},
		},
		{
			Role: llm.RoleAssistant,
			Content: []llm.Part{
				llm.NewTextPart("Working on it."),
				llm.NewReasoningPart("internal reasoning"),
			},
		},
	})

	last := apiReq.Messages[len(apiReq.Messages)-1]
	require.Len(t, last.Content, 2)
	require.NotNil(t, last.Content[1].OfThinking, "trailing block must be the thinking block")
	require.NotNil(t, last.Content[0].OfText)

	assert.True(t, hasCacheMarker(last.Content[0].OfText.CacheControl),
		"breakpoint must fall back to the text block preceding the thinking block")
}

// TestCacheBreakpointOnThinkingOnlyTurn pins the deliberate no-op: when the last
// message holds nothing but thinking blocks there is no legal place for a
// breakpoint, so the turn goes uncached rather than earning a 400. Falling back
// to the previous message's tail would also be valid — a shorter cached prefix
// still caches — but the shape is rare enough not to be worth the complexity.
func TestCacheBreakpointOnThinkingOnlyTurn(t *testing.T) {
	t.Parallel()

	apiReq := mapCachedRequest(t, []llm.Message{
		{
			Role:    llm.RoleSystem,
			Content: []llm.Part{llm.NewTextPart("You are a helpful assistant.")},
		},
		{
			Role:    llm.RoleUser,
			Content: []llm.Part{llm.NewTextPart("Think.")},
		},
		{
			Role:    llm.RoleAssistant,
			Content: []llm.Part{llm.NewReasoningPart("internal reasoning")},
		},
	})

	last := apiReq.Messages[len(apiReq.Messages)-1]
	require.Len(t, last.Content, 1)
	require.NotNil(t, last.Content[0].OfThinking)

	assert.False(t, lastMessageHasCacheMarker(last),
		"a thinking-only turn must be left unmarked; Anthropic rejects cache_control on thinking blocks")

	// The system-side breakpoint is independent and must survive.
	require.NotEmpty(t, apiReq.System)
	assert.True(t, hasCacheMarker(apiReq.System[len(apiReq.System)-1].CacheControl),
		"skipping the message breakpoint must not disturb the system breakpoint")
}

// TestCacheBreakpointSerializesOnToolResult proves the marker survives to the
// wire, not just to the param struct: a hand-built `cache_control` on a
// tool_result is only useful if the SDK actually serializes it there.
func TestCacheBreakpointSerializesOnToolResult(t *testing.T) {
	t.Parallel()

	apiReq := mapCachedRequest(t, []llm.Message{
		{
			Role:    llm.RoleUser,
			Content: []llm.Part{llm.NewTextPart("Call the tool.")},
		},
		{
			Role: llm.RoleAssistant,
			Content: []llm.Part{
				llm.NewToolRequestPart("call_1", "get_weather", json.RawMessage(`{"city":"Berlin"}`)),
			},
		},
		{
			Role: llm.RoleUser,
			Content: []llm.Part{
				llm.NewToolResponsePart("call_1", "get_weather", json.RawMessage(`{"temp_c":21}`), false),
			},
		},
	})

	body, err := json.Marshal(apiReq)
	require.NoError(t, err)

	var wire struct {
		Messages []struct {
			Content []struct {
				Type         string          `json:"type"`
				CacheControl json.RawMessage `json:"cache_control"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &wire))

	require.NotEmpty(t, wire.Messages)
	lastBlocks := wire.Messages[len(wire.Messages)-1].Content
	require.Len(t, lastBlocks, 1)

	assert.Equal(t, "tool_result", lastBlocks[0].Type)
	assert.JSONEq(t, `{"type":"ephemeral"}`, string(lastBlocks[0].CacheControl),
		"cache_control must serialize onto the tool_result block itself")
}

// mapCachedRequest maps messages through a default (caching-on) model.
func mapCachedRequest(t *testing.T, messages []llm.Message) anthropic.BetaMessageNewParams {
	t.Helper()

	p, err := NewProvider("test-key")
	require.NoError(t, err)

	model, err := p.NewModel(ModelClaudeSonnet45)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	apiReq, err := m.requestMapper.ToProvider(&llm.Request{Messages: messages})
	require.NoError(t, err)
	require.NotEmpty(t, apiReq.Messages)

	return apiReq
}
