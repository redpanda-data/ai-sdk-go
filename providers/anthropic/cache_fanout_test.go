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
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// fanOutConversation returns a conversation whose newest turn is an assistant
// message with width parallel tool calls followed by a user turn with their
// results. Its first user message is the tail the previous request cached.
func fanOutConversation(width int) []llm.Message {
	calls := make([]llm.Part, 0, width)
	results := make([]llm.Part, 0, width)

	for i := range width {
		id := fmt.Sprintf("call_%d", i)
		calls = append(calls, llm.NewToolRequestPart(id, "get_weather", json.RawMessage(`{"city":"Berlin"}`)))
		results = append(results, llm.NewToolResponsePart(id, "get_weather", json.RawMessage(`{"temp_c":21}`), false))
	}

	return []llm.Message{
		{Role: llm.RoleSystem, Content: []llm.Part{llm.NewTextPart("You are a helpful assistant.")}},
		{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("Check the weather everywhere.")}},
		{Role: llm.RoleAssistant, Content: calls},
		{Role: llm.RoleUser, Content: results},
	}
}

// TestCacheBreakpointOnPreviousTurnForWideFanOut covers #205: Anthropic looks
// back at most 20 blocks from a breakpoint, so a turn that appends 20 or more
// blocks hides the previous request's cache entry from the tail breakpoint.
// The user turn that ended the previous request must then carry a breakpoint
// of its own, and narrower turns must keep the two-breakpoint shape.
func TestCacheBreakpointOnPreviousTurnForWideFanOut(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		width        int
		wantPrevious bool
	}{
		{name: "single call", width: 1, wantPrevious: false},
		{name: "nine calls append 18 blocks", width: 9, wantPrevious: false},
		{name: "ten calls append 20 blocks", width: 10, wantPrevious: true},
		{name: "twenty calls append 40 blocks", width: 20, wantPrevious: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			apiReq := mapCachedRequest(t, fanOutConversation(tc.width))
			require.Len(t, apiReq.Messages, 3)

			previous := apiReq.Messages[0]
			require.Len(t, previous.Content, 1)
			require.NotNil(t, previous.Content[0].OfText)
			assert.Equal(t, tc.wantPrevious, hasCacheMarker(previous.Content[0].OfText.CacheControl),
				"breakpoint on the previous request's tail")

			assert.False(t, lastMessageHasCacheMarker(apiReq.Messages[1]),
				"the assistant fan-out turn needs no breakpoint")

			results := apiReq.Messages[2].Content
			assert.True(t, hasCacheMarker(results[len(results)-1].OfToolResult.CacheControl),
				"tail breakpoint stays on the last tool_result")

			body, err := json.Marshal(apiReq)
			require.NoError(t, err)

			want := 2
			if tc.wantPrevious {
				want = 3
			}

			assert.Equal(t, want, strings.Count(string(body), `"cache_control"`),
				"system, tail and (for wide turns) previous-tail breakpoints only")
		})
	}
}

// TestCacheBreakpointNoPreviousTurn checks the first request of a conversation,
// which has no earlier turn to anchor to.
func TestCacheBreakpointNoPreviousTurn(t *testing.T) {
	t.Parallel()

	parts := make([]llm.Part, 0, 25)
	for i := range 25 {
		parts = append(parts, llm.NewTextPart(fmt.Sprintf("document %d", i)))
	}

	apiReq := mapCachedRequest(t, []llm.Message{{Role: llm.RoleUser, Content: parts}})

	body, err := json.Marshal(apiReq)
	require.NoError(t, err)
	assert.Equal(t, 1, strings.Count(string(body), `"cache_control"`))
}
