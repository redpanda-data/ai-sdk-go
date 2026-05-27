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

package openai

import (
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestResponseMapper_Metadata(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelGPT5Mini])

	resp, err := mapper.FromProvider(&responses.Response{
		ID:          "resp_123",
		Model:       ModelGPT5Mini,
		Status:      responses.ResponseStatusCompleted,
		ServiceTier: responses.ResponseServiceTierDefault,
		Output: []responses.ResponseOutputItemUnion{{
			Type: outputTypeMessage,
			Content: []responses.ResponseOutputMessageContentUnion{{
				Type: contentTypeOutputText,
				Text: "Hello",
			}},
		}},
		Usage: responses.ResponseUsage{
			InputTokens:  10,
			OutputTokens: 5,
			TotalTokens:  15,
		},
	})
	require.NoError(t, err)

	assert.Equal(t, llm.ServiceTierDefault, resp.ServiceTier)
	assert.Equal(t, llm.ModelID(ModelGPT5Mini), resp.InvokedModelID)
}

// TestResponseMapper_FinishReasonTruncationWithToolCalls locks in the rule
// that truncation signals win over the hasToolCalls-based ToolCalls upgrade.
// See providers/anthropic/response_mapper_test.go for the full rationale.
func TestResponseMapper_FinishReasonTruncationWithToolCalls(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelGPT5Mini])

	toolOutput := responses.ResponseOutputItemUnion{
		Type:   outputTypeFunctionCall,
		CallID: "call_1",
		Name:   "query",
		Arguments: responses.ResponseOutputItemUnionArguments{
			OfString: `{"q":"SELECT 1"}`,
		},
	}
	textOutput := responses.ResponseOutputItemUnion{
		Type: outputTypeMessage,
		Content: []responses.ResponseOutputMessageContentUnion{{
			Type: contentTypeOutputText,
			Text: "done",
		}},
	}

	cases := []struct {
		name       string
		status     responses.ResponseStatus
		incomplete responses.ResponseIncompleteDetails
		output     []responses.ResponseOutputItemUnion
		want       llm.FinishReason
	}{
		{
			name:   "completed with tool calls promotes to ToolCalls",
			status: responses.ResponseStatusCompleted,
			output: []responses.ResponseOutputItemUnion{toolOutput},
			want:   llm.FinishReasonToolCalls,
		},
		{
			name:   "completed without tool calls stays Stop",
			status: responses.ResponseStatusCompleted,
			output: []responses.ResponseOutputItemUnion{textOutput},
			want:   llm.FinishReasonStop,
		},
		{
			name:       "incomplete max_output_tokens with tool calls stays Length",
			status:     responses.ResponseStatusIncomplete,
			incomplete: responses.ResponseIncompleteDetails{Reason: "max_output_tokens"},
			output:     []responses.ResponseOutputItemUnion{toolOutput},
			want:       llm.FinishReasonLength,
		},
		{
			name:       "incomplete content_filter with tool calls stays ContentFilter",
			status:     responses.ResponseStatusIncomplete,
			incomplete: responses.ResponseIncompleteDetails{Reason: "content_filter"},
			output:     []responses.ResponseOutputItemUnion{toolOutput},
			want:       llm.FinishReasonContentFilter,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, err := mapper.FromProvider(&responses.Response{
				ID:                "resp_finish",
				Model:             ModelGPT5Mini,
				Status:            tc.status,
				IncompleteDetails: tc.incomplete,
				Output:            tc.output,
				Usage: responses.ResponseUsage{
					InputTokens:  1,
					OutputTokens: 1,
					TotalTokens:  2,
				},
			})
			require.NoError(t, err)
			assert.Equal(t, tc.want, resp.FinishReason)
		})
	}
}
