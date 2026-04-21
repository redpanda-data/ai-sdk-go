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

package bedrock

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestResponseMapper_FinishReasonTruncationWithToolCalls locks in the rule
// that truncation signals win over the hasToolCalls-based ToolCalls upgrade.
// See providers/anthropic/response_mapper_test.go for the full rationale.
func TestResponseMapper_FinishReasonTruncationWithToolCalls(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeSonnet46])

	toolUse := &types.ContentBlockMemberToolUse{
		Value: types.ToolUseBlock{
			ToolUseId: aws.String("tooluse_1"),
			Name:      aws.String("query"),
			Input:     document.NewLazyDocument(map[string]any{"q": "SELECT 1"}),
		},
	}
	text := &types.ContentBlockMemberText{Value: "done"}

	cases := []struct {
		name       string
		stopReason types.StopReason
		blocks     []types.ContentBlock
		want       llm.FinishReason
	}{
		{
			name:       "clean tool_use turn stays ToolCalls",
			stopReason: types.StopReasonToolUse,
			blocks:     []types.ContentBlock{toolUse},
			want:       llm.FinishReasonToolCalls,
		},
		{
			name:       "end_turn with tool calls promotes to ToolCalls",
			stopReason: types.StopReasonEndTurn,
			blocks:     []types.ContentBlock{toolUse},
			want:       llm.FinishReasonToolCalls,
		},
		{
			name:       "max_tokens with tool calls stays Length",
			stopReason: types.StopReasonMaxTokens,
			blocks:     []types.ContentBlock{toolUse},
			want:       llm.FinishReasonLength,
		},
		{
			name:       "content_filtered with tool calls stays ContentFilter",
			stopReason: types.StopReasonContentFiltered,
			blocks:     []types.ContentBlock{toolUse},
			want:       llm.FinishReasonContentFilter,
		},
		{
			name:       "max_tokens without tool calls stays Length",
			stopReason: types.StopReasonMaxTokens,
			blocks:     []types.ContentBlock{text},
			want:       llm.FinishReasonLength,
		},
		{
			name:       "end_turn without tool calls stays Stop",
			stopReason: types.StopReasonEndTurn,
			blocks:     []types.ContentBlock{text},
			want:       llm.FinishReasonStop,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			output := &types.ConverseOutputMemberMessage{
				Value: types.Message{
					Role:    types.ConversationRoleAssistant,
					Content: tc.blocks,
				},
			}

			resp, err := mapper.FromConverseOutput(tc.stopReason, output, nil, nil, nil, nil)
			require.NoError(t, err)
			assert.Equal(t, tc.want, resp.FinishReason)
		})
	}
}
