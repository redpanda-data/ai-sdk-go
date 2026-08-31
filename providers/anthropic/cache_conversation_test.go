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

package anthropic_test

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
	"github.com/redpanda-data/ai-sdk-go/providers/testutil"
)

// TestAnthropicConversationCaching_Integration is the end-to-end proof that an
// agentic conversation caches, against the live API.
//
// It replays the shape of a real tool-calling loop: every turn appends an
// assistant tool_use and a user tool_result, so from turn 2 on the last message
// carries no text block at all. Two things are being established that a
// request-shape unit test cannot:
//
//  1. Anthropic accepts cache_control on a tool_result block. A rejected
//     breakpoint is a 400 on the whole request, so any Generate error fails here.
//  2. The conversation, not just the static tools+system prefix, is cached.
//     The signature is cache_read GROWING turn over turn. When the breakpoint is
//     restricted to text blocks the tool_result turns go unmarked, no cache entry
//     is ever written past the system blocks, and cache_read sits FLAT at the
//     system-prefix size while input_tokens climbs with the history.
func TestAnthropicConversationCaching_Integration(t *testing.T) {
	t.Parallel()

	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	provider, err := anthropic.NewProvider(apiKey, anthropic.WithTimeout(3*time.Minute))
	require.NoError(t, err)

	// Keep the output budget small: only usage accounting matters here.
	model, err := provider.NewModel(anthropictest.TestModelName, anthropic.WithMaxTokens(64))
	require.NoError(t, err)

	tools := []llm.ToolDefinition{{
		Name:        "lookup_record",
		Description: "Look up a record by id.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}`),
	}}

	// The cacheable prefix has a per-model token minimum (1024 on Sonnet), so
	// the system prompt has to clear it on its own for turn 1 to write anything.
	messages := []llm.Message{
		{
			Role:    llm.RoleSystem,
			Content: []llm.Part{llm.NewTextPart(testutil.GenerateLargePrompt(1800))},
		},
		{
			Role:    llm.RoleUser,
			Content: []llm.Part{llm.NewTextPart("Look up record alpha.")},
		},
	}

	ctx := context.Background()

	const turns = 4

	reads := make([]int, 0, turns)
	inputs := make([]int, 0, turns)

	for turn := 1; turn <= turns; turn++ {
		resp, err := model.Generate(ctx, &llm.Request{Messages: messages, Tools: tools})
		require.NoError(t, err, "turn %d: a rejected cache_control breakpoint surfaces as a 400 here", turn)
		require.NotNil(t, resp.Usage)

		reads = append(reads, resp.Usage.CachedInputTokens)
		inputs = append(inputs, resp.Usage.InputTokens)

		t.Logf("turn %d: input %d, cache_read %d, cache_write %d",
			turn, resp.Usage.InputTokens, resp.Usage.CachedInputTokens,
			resp.Usage.CacheCreation5mTokens+resp.Usage.CacheCreationUnknownTTLTokens)

		// Append a fixed tool round-trip rather than the model's own reply: the
		// cache is a byte-prefix match, so the history has to be reproduced
		// exactly on the next turn. A non-deterministic assistant turn would
		// invalidate the prefix and make this test measure nothing.
		callID := fmt.Sprintf("call_%d", turn)
		messages = append(messages,
			llm.Message{
				Role: llm.RoleAssistant,
				Content: []llm.Part{
					llm.NewToolRequestPart(callID, "lookup_record",
						json.RawMessage(fmt.Sprintf(`{"id":"record-%d"}`, turn))),
				},
			},
			llm.Message{
				Role: llm.RoleUser,
				Content: []llm.Part{
					llm.NewToolResponsePart(callID, "lookup_record",
						json.RawMessage(fmt.Sprintf(`{"id":"record-%d","body":%q}`,
							turn, testutil.GenerateLargePrompt(400))), false),
				},
			},
		)
	}

	// Turn 1 writes; from turn 2 on every request must read a prefix that
	// includes the preceding tool_result turns, so the read grows each time.
	require.Positive(t, reads[1], "turn 2 must read the cached prefix written by turn 1")

	for turn := 2; turn < turns; turn++ {
		require.Greater(t, reads[turn], reads[turn-1],
			"cache_read must grow as the conversation grows (turn %d: %d vs turn %d: %d); "+
				"a flat cache_read while input climbs (%v) means the tool_result turns carry no breakpoint "+
				"and only the static system prefix is cached",
			turn+1, reads[turn], turn, reads[turn-1], inputs)
	}
}
