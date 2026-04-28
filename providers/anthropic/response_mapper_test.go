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
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestResponseMapper_Metadata(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeOpus46])

	resp, err := mapper.FromProvider(&anthropic.BetaMessage{
		ID:    "msg_123",
		Model: anthropic.Model("claude-opus-4-6-20260401"),
		Content: []anthropic.BetaContentBlockUnion{{
			Type: blockTypeText,
			Text: "Hello",
		}},
		StopReason: anthropic.BetaStopReasonEndTurn,
		Usage: anthropic.BetaUsage{
			InputTokens:  10,
			OutputTokens: 5,
			ServiceTier:  anthropic.BetaUsageServiceTierStandard,
			Speed:        anthropic.BetaUsageSpeedFast,
			InferenceGeo: "us-east-1",
		},
	})
	require.NoError(t, err)

	assert.Equal(t, llm.ServiceTierDefault, resp.ServiceTier)
	assert.Equal(t, llm.SpeedFast, resp.Speed)
	assert.Equal(t, "us-east-1", resp.InferenceRegion)
	assert.Equal(t, ModelClaudeOpus46, resp.InvokedModelID)
}

// TestResponseMapper_FinishReasonTruncationWithToolCalls locks in the rule
// that truncation signals win over the hasToolCalls-based ToolCalls upgrade.
// Before the fix, any completed tool_use block forced FinishReason to
// ToolCalls, silently swallowing max_tokens / context_window signals.
func TestResponseMapper_FinishReasonTruncationWithToolCalls(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeOpus46])

	toolUseBlock := anthropic.BetaContentBlockUnion{
		Type:  blockTypeToolUse,
		ID:    "toolu_1",
		Name:  "query",
		Input: []byte(`{"q":"SELECT 1"}`),
	}
	textBlock := anthropic.BetaContentBlockUnion{Type: blockTypeText, Text: "done"}

	cases := []struct {
		name       string
		stopReason anthropic.BetaStopReason
		content    []anthropic.BetaContentBlockUnion
		want       llm.FinishReason
	}{
		{
			name:       "clean tool-use turn stays ToolCalls",
			stopReason: anthropic.BetaStopReasonToolUse,
			content:    []anthropic.BetaContentBlockUnion{toolUseBlock},
			want:       llm.FinishReasonToolCalls,
		},
		{
			name:       "end_turn with tool calls promotes to ToolCalls",
			stopReason: anthropic.BetaStopReasonEndTurn,
			content:    []anthropic.BetaContentBlockUnion{toolUseBlock},
			want:       llm.FinishReasonToolCalls,
		},
		{
			name:       "max_tokens with tool calls stays Length",
			stopReason: anthropic.BetaStopReasonMaxTokens,
			content:    []anthropic.BetaContentBlockUnion{toolUseBlock},
			want:       llm.FinishReasonLength,
		},
		{
			name:       "context_window with tool calls stays Length",
			stopReason: anthropic.BetaStopReasonModelContextWindowExceeded,
			content:    []anthropic.BetaContentBlockUnion{toolUseBlock},
			want:       llm.FinishReasonLength,
		},
		{
			name:       "refusal with tool calls stays ContentFilter",
			stopReason: anthropic.BetaStopReasonRefusal,
			content:    []anthropic.BetaContentBlockUnion{toolUseBlock},
			want:       llm.FinishReasonContentFilter,
		},
		{
			name:       "max_tokens without tool calls stays Length",
			stopReason: anthropic.BetaStopReasonMaxTokens,
			content:    []anthropic.BetaContentBlockUnion{textBlock},
			want:       llm.FinishReasonLength,
		},
		{
			name:       "end_turn without tool calls stays Stop",
			stopReason: anthropic.BetaStopReasonEndTurn,
			content:    []anthropic.BetaContentBlockUnion{textBlock},
			want:       llm.FinishReasonStop,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, err := mapper.FromProvider(&anthropic.BetaMessage{
				ID:         "msg_finish",
				Model:      anthropic.Model("claude-opus-4-6-20260401"),
				Content:    tc.content,
				StopReason: tc.stopReason,
				Usage:      anthropic.BetaUsage{InputTokens: 1, OutputTokens: 1},
			})
			require.NoError(t, err)
			assert.Equal(t, tc.want, resp.FinishReason)
		})
	}
}

// TestResponseMapper_EmptyContentIsError locks in the invariant that the
// provider never produces a Response with empty Message.Content. Sibling
// case to the truncation fix in PR #116: silent failure modes corrupt
// session state, so surface them as errors at the provider boundary
// rather than handing the agent loop a non-result it has to guess about.
//
// Sources of empty content we have observed (or that PR #116 made
// possible by correctly dropping partial tool_use blocks):
//
//   - max_tokens hit before any content block was emitted
//   - the only content block was a partial tool_use that stream
//     finalisation dropped (PR #116)
//   - refusal with no text emitted
//   - rare provider anomalies (200 OK with empty content array)
//
// In every case the caller has no useful output. Returning an error
// lets the agent loop's terminal-error handling fire (status: failure,
// nothing persisted) instead of appending an empty Message to session
// state that Anthropic rejects on every subsequent replay.
func TestResponseMapper_EmptyContentIsError(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeOpus46])

	cases := []struct {
		name       string
		stopReason anthropic.BetaStopReason
	}{
		{name: "end_turn with empty content", stopReason: anthropic.BetaStopReasonEndTurn},
		{name: "max_tokens with empty content", stopReason: anthropic.BetaStopReasonMaxTokens},
		{name: "refusal with empty content", stopReason: anthropic.BetaStopReasonRefusal},
		{name: "context_window with empty content", stopReason: anthropic.BetaStopReasonModelContextWindowExceeded},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, err := mapper.FromProvider(&anthropic.BetaMessage{
				ID:         "msg_empty",
				Model:      anthropic.Model("claude-opus-4-6-20260401"),
				Content:    []anthropic.BetaContentBlockUnion{},
				StopReason: tc.stopReason,
				Usage:      anthropic.BetaUsage{InputTokens: 1, OutputTokens: 0},
			})

			require.Error(t, err, "empty content must surface as error")
			assert.ErrorIs(t, err, llm.ErrResponseMapping)
			assert.Nil(t, resp)
		})
	}
}
