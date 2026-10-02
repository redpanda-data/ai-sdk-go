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
	"errors"
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
//     The signature is the cached prefix (cache_read + cache_write) GROWING
//     turn over turn, each turn reading at least what the previous one left
//     cached; see checkConversationCached. When the breakpoint is restricted
//     to text blocks the tool_result turns go unmarked, no cache entry is ever
//     read or written past the system blocks, and the cached prefix sits FLAT
//     at the system-prefix size while input_tokens climbs with the history.
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

	// The cacheable prefix has a per-model token minimum (512 on Sonnet 5.5,
	// 1024 on earlier Sonnets), so the system prompt has to clear it on its own
	// for turn 1 to write anything.
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

	history := make([]cacheTurn, 0, turns)

	for turn := 1; turn <= turns; turn++ {
		resp, err := model.Generate(ctx, &llm.Request{Messages: messages, Tools: tools})
		require.NoError(t, err, "turn %d: a rejected cache_control breakpoint surfaces as a 400 here", turn)
		require.NotNil(t, resp.Usage)

		usage := resp.Usage
		got := cacheTurn{
			read:  usage.CachedInputTokens,
			write: usage.CacheCreation5mTokens + usage.CacheCreation1hTokens + usage.CacheCreationUnknownTTLTokens,
			input: usage.InputTokens,
		}
		history = append(history, got)

		t.Logf("turn %d: input %d, cache_read %d, cache_write %d", turn, got.input, got.read, got.write)

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

	require.NoError(t, checkConversationCached(history), "per-turn cache usage: %+v", history)
}

// cacheTurn is one request's prompt-cache accounting.
type cacheTurn struct {
	read  int // cache_read_input_tokens
	write int // cache_creation_input_tokens, every TTL
	input int // uncached tokens after the last breakpoint
}

// cached is the prefix the cache covers once the request has run: what it read
// plus what it wrote, which is everything up to its last breakpoint.
func (c cacheTurn) cached() int {
	return c.read + c.write
}

// checkConversationCached reports the first turn that breaks the signature of
// a cached conversation: every turn after the first extends the cached prefix
// and reads at least as much of it as the previous turn left cached.
//
// It deliberately does not require cache_read alone to grow. Caches are shared
// across the workspace and this test's prompts are deterministic, so another CI
// run of the same test can have cached this run's next prefix already: that
// turn reads all of it and writes nothing, and the turn after it then reads
// exactly as much again before writing its new tail. Only read+write is
// guaranteed to grow.
//
// Sharing cannot hide a missing breakpoint. A request reads and writes only
// back from its own breakpoints, so with the tool_result turns unmarked the
// last breakpoint is the system block, and read+write stays capped at the
// static prefix however much other runs have cached past it.
func checkConversationCached(turns []cacheTurn) error {
	if len(turns) == 0 || turns[0].cached() == 0 {
		return errors.New("turn 1 cached nothing: the prefix is below the model's cacheable minimum, " +
			"or no breakpoint was sent")
	}

	for i := 1; i < len(turns); i++ {
		prev, cur := turns[i-1], turns[i]

		if cur.cached() <= prev.cached() {
			return fmt.Errorf("turn %d: the cached prefix did not grow (%d vs turn %d: %d); "+
				"the tool_result turns carry no breakpoint, so only the static tools+system prefix is cached",
				i+1, cur.cached(), i, prev.cached())
		}

		if cur.read < prev.cached() {
			return fmt.Errorf("turn %d: read %d of the %d tokens turn %d left cached; "+
				"either the replayed history is not byte-identical, so each turn rewrites the cache instead of reading it, "+
				"or the earlier entry expired (5-minute TTL) or was evicted between turns",
				i+1, cur.read, prev.cached(), i)
		}
	}

	return nil
}

func TestCheckConversationCached(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		turns   []cacheTurn
		wantErr string
	}{
		{
			name: "cold run reads the previous turn and writes its tail",
			turns: []cacheTurn{
				{read: 0, write: 2878, input: 3},
				{read: 2878, write: 637, input: 5},
				{read: 3515, write: 638, input: 5},
				{read: 4153, write: 638, input: 5},
			},
		},
		{
			// Recorded in CI: another run had cached turns 1-3 already, so turn
			// 3 wrote nothing and turn 4 read exactly what turn 3 did.
			name: "another run cached the first turns already",
			turns: []cacheTurn{
				{read: 2878, write: 0, input: 3},
				{read: 3515, write: 0, input: 5},
				{read: 4153, write: 0, input: 5},
				{read: 4153, write: 638, input: 5},
			},
		},
		{
			name: "another run cached every turn already",
			turns: []cacheTurn{
				{read: 2878, write: 0, input: 3},
				{read: 3515, write: 0, input: 5},
				{read: 4153, write: 0, input: 5},
				{read: 4791, write: 0, input: 5},
			},
		},
		{
			name: "unmarked tool_result turns cache only the system prefix",
			turns: []cacheTurn{
				{read: 0, write: 2878, input: 3},
				{read: 2866, write: 0, input: 649},
				{read: 2866, write: 0, input: 1287},
				{read: 2866, write: 0, input: 1925},
			},
			wantErr: "turn 2: the cached prefix did not grow",
		},
		{
			name: "unmarked tool_result turns stay flat when another run cached further",
			turns: []cacheTurn{
				{read: 2878, write: 0, input: 3},
				{read: 2866, write: 0, input: 649},
				{read: 2866, write: 0, input: 1287},
			},
			wantErr: "turn 2: the cached prefix did not grow",
		},
		{
			name: "a changed history rewrites the tail instead of reading it",
			turns: []cacheTurn{
				{read: 0, write: 2878, input: 3},
				{read: 2866, write: 649, input: 5},
				{read: 2866, write: 1287, input: 5},
			},
			wantErr: "turn 2: read 2866 of the 2878 tokens turn 1 left cached",
		},
		{
			name: "nothing cached",
			turns: []cacheTurn{
				{read: 0, write: 0, input: 2881},
				{read: 0, write: 0, input: 3520},
			},
			wantErr: "turn 1 cached nothing",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := checkConversationCached(tt.turns)
			if tt.wantErr == "" {
				require.NoError(t, err)

				return
			}

			require.ErrorContains(t, err, tt.wantErr)
		})
	}
}
