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
	"encoding/json"
	"fmt"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestResponseMapper_Metadata(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelGPT5Mini], "eu")

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
	assert.Equal(t, ModelGPT5Mini, resp.InvokedModelID)
	assert.Equal(t, "eu", resp.InferenceRegion)
}

func TestResponseMapper_CacheUsageBucketsAreDisjoint(t *testing.T) {
	t.Parallel()

	payload := `{
		"id": "resp_cache",
		"model": "gpt-5-mini",
		"status": "completed",
		"output": [{
			"id": "msg_cache",
			"type": "message",
			"role": "assistant",
			"status": "completed",
			"content": [{"type": "output_text", "text": "ok", "annotations": []}]
		}],
		"usage": {
			"input_tokens": 100,
			"input_tokens_details": {
				"cached_tokens": 30,
				"cache_write_tokens": 20
			},
			"output_tokens": 5,
			"total_tokens": 105
		}
	}`

	var providerResponse responses.Response
	require.NoError(t, json.Unmarshal([]byte(payload), &providerResponse))

	mapper := NewResponseMapper(supportedModels[ModelGPT5Mini], "")
	resp, err := mapper.FromProvider(&providerResponse)
	require.NoError(t, err)
	require.NotNil(t, resp.Usage)

	assert.Equal(t, 50, resp.Usage.InputTokens)
	assert.Equal(t, 30, resp.Usage.CachedInputTokens)
	assert.Equal(t, 20, resp.Usage.CacheCreationUnknownTTLTokens)
	assert.Equal(t, 100, resp.Usage.BilledInputTokens())
}

func TestResponseMapper_RejectsInvalidUsageCounters(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		usage responses.ResponseUsage
	}{
		{
			name: "cache subsets exceed input total",
			usage: responses.ResponseUsage{
				InputTokens: 100,
				InputTokensDetails: responses.ResponseUsageInputTokensDetails{
					CachedTokens:     90,
					CacheWriteTokens: 20,
				},
			},
		},
		{
			name: "reasoning exceeds output total",
			usage: responses.ResponseUsage{
				OutputTokens: 10,
				OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
					ReasoningTokens: 11,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			mapper := NewResponseMapper(supportedModels[ModelGPT5Mini], "")
			_, err := mapper.FromProvider(&responses.Response{
				Status: responses.ResponseStatusCompleted,
				Output: []responses.ResponseOutputItemUnion{{
					Type: outputTypeMessage,
					Content: []responses.ResponseOutputMessageContentUnion{{
						Type: contentTypeOutputText,
						Text: "ok",
					}},
				}},
				Usage: tt.usage,
			})
			require.ErrorIs(t, err, llm.ErrResponseMapping)
		})
	}
}

// TestResponseMapper_FinishReasonTruncationWithToolCalls locks in the rule
// that truncation signals win over the hasToolCalls-based ToolCalls upgrade.
// See providers/anthropic/response_mapper_test.go for the full rationale.
func TestResponseMapper_FinishReasonTruncationWithToolCalls(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelGPT5Mini], "")

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

// TestResponseMapper_FunctionCallArgumentsNormalization locks in how raw
// function-call argument strings map to ToolRequestPart.Arguments. OpenAI
// emits an empty arguments string for zero-parameter tool calls; passing
// that through as an empty RawMessage makes downstream tool executors
// serialize the call as `"arguments": null`, which strict MCP servers
// reject. The Bedrock mapper already defaults absent input to {} — this
// keeps providers consistent.
//
// The response is built from JSON because the openai-go output-item union
// reconstructs variants from the captured raw JSON (AsFunctionCall reads
// u.JSON.raw); struct-literal unions carry no field data into AsAny().
func TestResponseMapper_FunctionCallArgumentsNormalization(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelGPT5Mini], "")

	cases := []struct {
		name string
		args string
		want string
	}{
		{"empty arguments become an object", ``, `{}`},
		{"populated arguments pass through", `{\"q\":\"SELECT 1\"}`, `{"q":"SELECT 1"}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			payload := fmt.Sprintf(`{
				"id": "resp_args",
				"model": "%s",
				"status": "completed",
				"output": [{
					"type": "function_call",
					"call_id": "call_1",
					"name": "get_me",
					"arguments": "%s"
				}],
				"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
			}`, ModelGPT5Mini, tc.args)

			var r responses.Response
			require.NoError(t, json.Unmarshal([]byte(payload), &r))

			resp, err := mapper.FromProvider(&r)
			require.NoError(t, err)

			require.Len(t, resp.Message.Content, 1)
			part, ok := resp.Message.Content[0].(*llm.ToolRequestPart)
			require.True(t, ok, "expected *llm.ToolRequestPart, got %T", resp.Message.Content[0])
			assert.JSONEq(t, tc.want, string(part.Arguments))
		})
	}
}
