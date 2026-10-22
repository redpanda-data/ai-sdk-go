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
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestResponseMapper_ReasoningPartsKeepItemID checks that every summary of
// one reasoning item carries the item's own ID, so the turn can be sent
// back as OpenAI issued it. The encrypted reasoning rides on the first part.
func TestResponseMapper_ReasoningPartsKeepItemID(t *testing.T) {
	t.Parallel()

	payload := `{
		"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5.4-mini",
		"output": [
			{"type": "reasoning", "id": "rs_abc", "encrypted_content": "enc-blob",
			 "summary": [{"type": "summary_text", "text": "first"}, {"type": "summary_text", "text": ""}, {"type": "summary_text", "text": "third"}]},
			{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather", "arguments": "{}", "status": "completed"}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
	}`

	var providerResponse responses.Response
	require.NoError(t, json.Unmarshal([]byte(payload), &providerResponse))

	resp, err := NewResponseMapper().FromProvider(&providerResponse)
	require.NoError(t, err)

	var parts []*llm.ReasoningPart

	for _, part := range resp.Message.Content {
		if rp, ok := part.(*llm.ReasoningPart); ok {
			parts = append(parts, rp)
		}
	}

	require.Len(t, parts, 2)
	assert.Equal(t, "rs_abc", parts[0].ID)
	assert.Equal(t, "rs_abc", parts[1].ID)
	assert.Equal(t, 0, parts[0].Metadata[reasoningMetadataSummaryIndex])
	assert.Equal(t, 2, parts[1].Metadata[reasoningMetadataSummaryIndex])
	assert.Equal(t, "enc-blob", parts[0].Signature)
	assert.Empty(t, parts[1].Signature)
}

// TestRequestMapper_ReplaysReasoningItemsAsIssued checks that reasoning
// parts go back as the items OpenAI issued: one item per item ID with its
// summaries in order, sessions saved with the old "<id>-<n>" part IDs mapped
// back to the item ID, and parts with no item ID skipped.
func TestRequestMapper_ReplaysReasoningItemsAsIssued(t *testing.T) {
	t.Parallel()

	type wantItem struct {
		id        string
		summaries []string
		encrypted string
	}

	tests := []struct {
		name  string
		parts []llm.Part
		want  []wantItem
	}{
		{
			name: "summaries of one item merge",
			parts: []llm.Part{
				&llm.ReasoningPart{ID: "rs_abc", Text: "first", Signature: "enc-blob", Metadata: map[string]any{"summary_index": 0}},
				&llm.ReasoningPart{ID: "rs_abc", Text: "third", Metadata: map[string]any{"summary_index": 2}},
			},
			want: []wantItem{{id: "rs_abc", summaries: []string{"first", "third"}, encrypted: "enc-blob"}},
		},
		{
			name: "separate items stay separate",
			parts: []llm.Part{
				&llm.ReasoningPart{ID: "rs_a", Text: "a"},
				&llm.ReasoningPart{ID: "rs_b", Text: "b"},
			},
			want: []wantItem{{id: "rs_a", summaries: []string{"a"}}, {id: "rs_b", summaries: []string{"b"}}},
		},
		{
			name: "legacy suffixed IDs from a JSON-persisted session",
			parts: []llm.Part{
				&llm.ReasoningPart{ID: "rs_abc-0", Text: "first", Metadata: map[string]any{"summary_index": float64(0)}},
				&llm.ReasoningPart{ID: "rs_abc-1", Text: "second", Metadata: map[string]any{"summary_index": float64(1)}},
			},
			want: []wantItem{{id: "rs_abc", summaries: []string{"first", "second"}}},
		},
		{
			name: "a part with no item ID has nothing to reference",
			parts: []llm.Part{
				&llm.ReasoningPart{Text: "thinking from another provider", Signature: "sig"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			parts := append(slicesClone(tt.parts), llm.NewToolRequestPart("call_1", "get_weather", json.RawMessage(`{}`)))

			items, err := NewRequestMapper(&Config{}).mapMessagesToInputItems([]llm.Message{
				{Role: llm.RoleAssistant, Content: parts},
			})
			require.NoError(t, err)

			var got []wantItem

			for _, item := range items {
				if item.OfReasoning == nil {
					continue
				}

				w := wantItem{id: item.OfReasoning.ID, encrypted: item.OfReasoning.EncryptedContent.Value}
				for _, s := range item.OfReasoning.Summary {
					w.summaries = append(w.summaries, s.Text)
				}

				got = append(got, w)
			}

			assert.Equal(t, tt.want, got)
			require.NotNil(t, items[len(items)-1].OfFunctionCall, "the tool call still follows its reasoning")
		})
	}
}

func slicesClone(parts []llm.Part) []llm.Part {
	return append([]llm.Part(nil), parts...)
}
