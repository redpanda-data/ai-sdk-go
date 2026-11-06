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

package openaicompat

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestNativeReasoningToolReplay(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("../fireworks/testdata/contracts/tool-corrections.json")
	require.NoError(t, err)

	var fixture struct {
		Evidence []struct {
			Case struct {
				Model  string `json:"model"`
				API    string `json:"api"`
				Stream bool   `json:"stream"`
			} `json:"case"`
			Exchanges []struct {
				Response string `json:"response"`
			} `json:"exchanges"`
		} `json:"evidence"`
	}
	require.NoError(t, json.Unmarshal(data, &fixture))

	var cases int

	for _, evidence := range fixture.Evidence {
		if evidence.Case.API != "openai_chat_completions" || evidence.Case.Stream {
			continue
		}

		cases++

		t.Run(evidence.Case.Model, func(t *testing.T) {
			t.Parallel()

			require.NotEmpty(t, evidence.Exchanges)

			raw := []byte(evidence.Exchanges[0].Response)
			var native struct {
				Choices []struct {
					Message struct {
						ReasoningContent string           `json:"reasoning_content"`
						ToolCalls        []map[string]any `json:"tool_calls"`
					} `json:"message"`
				} `json:"choices"`
			}
			require.NoError(t, json.Unmarshal(raw, &native))
			require.Len(t, native.Choices, 1)
			require.NotEmpty(t, native.Choices[0].Message.ReasoningContent)
			require.Len(t, native.Choices[0].Message.ToolCalls, 1)

			var apiResp openai.ChatCompletion
			require.NoError(t, apiResp.UnmarshalJSON(raw))

			response, err := NewResponseMapper(getDefaultConstraints()).FromProvider(&apiResp)
			require.NoError(t, err)

			mapper := NewRequestMapper(&Config{ModelName: evidence.Case.Model, SupportsReasoning: true})
			request, err := mapper.ToProvider(&llm.Request{Messages: []llm.Message{
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("look up x")),
				response.Message,
				llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(
					apiResp.Choices[0].Message.ToolCalls[0].ID, "lookup", json.RawMessage(`"found"`), false,
				)),
			}})
			require.NoError(t, err)

			encoded, err := request.MarshalJSON()
			require.NoError(t, err)

			var body struct {
				Messages []map[string]any `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(encoded, &body))
			require.Len(t, body.Messages, 3)
			assert.Equal(t, native.Choices[0].Message.ReasoningContent, body.Messages[1]["reasoning_content"])

			toolCall := native.Choices[0].Message.ToolCalls[0]
			mappedTools, ok := body.Messages[1]["tool_calls"].([]any)
			require.True(t, ok)
			require.Len(t, mappedTools, 1)
			mappedTool, ok := mappedTools[0].(map[string]any)
			require.True(t, ok)
			assert.Equal(t, toolCall["id"], mappedTool["id"])
			assert.Equal(t, toolCall["function"], mappedTool["function"])
			assert.Equal(t, toolCall["id"], body.Messages[2]["tool_call_id"])
		})
	}

	require.NotZero(t, cases)
}

func TestReasoningHistoryOptIn(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		enabled   bool
		parts     []llm.Part
		reasoning string
	}{
		{name: "default omitted", parts: []llm.Part{llm.NewReasoningPart("private"), llm.NewTextPart("answer")}},
		{name: "text and reasoning", enabled: true, parts: []llm.Part{llm.NewReasoningPart("think"), llm.NewTextPart("answer")}, reasoning: "think"},
		{name: "reasoning only", enabled: true, parts: []llm.Part{llm.NewReasoningPart("think")}, reasoning: "think"},
		{name: "multiple fragments", enabled: true, parts: []llm.Part{llm.NewReasoningPart("think"), llm.NewReasoningPart(" more"), llm.NewTextPart("answer")}, reasoning: "think more"},
		{name: "foreign item id", enabled: true, parts: []llm.Part{&llm.ReasoningPart{Text: "opaque", ID: "foreign"}, llm.NewTextPart("answer")}},
		{name: "foreign signature", enabled: true, parts: []llm.Part{&llm.ReasoningPart{Text: "opaque", Signature: "foreign"}, llm.NewTextPart("answer")}},
		{name: "foreign metadata", enabled: true, parts: []llm.Part{&llm.ReasoningPart{Text: "opaque", Metadata: map[string]any{"foreign": true}}, llm.NewTextPart("answer")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			request, err := NewRequestMapper(&Config{ModelName: "custom-model", SupportsReasoning: tt.enabled}).ToProvider(&llm.Request{
				Messages: []llm.Message{{Role: llm.RoleAssistant, Content: tt.parts}},
			})
			require.NoError(t, err)

			encoded, err := request.MarshalJSON()
			require.NoError(t, err)

			var body struct {
				Messages []map[string]any `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(encoded, &body))
			require.Len(t, body.Messages, 1)

			if tt.reasoning == "" {
				assert.NotContains(t, body.Messages[0], "reasoning_content")
				assert.NotContains(t, string(encoded), "opaque")
				assert.NotContains(t, string(encoded), "foreign")
			} else {
				assert.Equal(t, tt.reasoning, body.Messages[0]["reasoning_content"])
			}
		})
	}
}
