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

package openai_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
)

// TestReasoningSummaryToolLoop_Integration replays a reasoning model's
// tool-call turn, reasoning summaries included, against the live Responses
// API, the way llmagent and the Vercel UI stream adapter do: the second
// request must be accepted.
func TestReasoningSummaryToolLoop_Integration(t *testing.T) {
	t.Parallel()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	provider, err := openai.NewProvider(apiKey, openai.WithTimeout(2*time.Minute))
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName,
		openai.WithReasoningEffort(openai.ReasoningEffortMedium),
		openai.WithReasoningSummary(openai.ReasoningSummaryDetailed),
	)
	require.NoError(t, err)

	weather := llm.ToolDefinition{
		Name:        "get_weather",
		Description: "Get the current weather for a city",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}
	question := llm.NewMessage(llm.RoleUser, llm.NewTextPart(
		"Which of Paris, Oslo, or Cairo is most likely to be the coldest right now? "+
			"Reason about it briefly, then check the one you pick with the get_weather tool."))

	first, err := model.Generate(t.Context(), &llm.Request{
		Messages: []llm.Message{question},
		Tools:    []llm.ToolDefinition{weather},
	})
	require.NoError(t, err)

	calls := first.Message.ToolRequests()
	if len(calls) == 0 {
		t.Skip("model answered without calling the tool")
	}

	reasoning := 0

	for _, part := range first.Message.Content {
		if _, ok := part.(*llm.ReasoningPart); ok {
			reasoning++
		}
	}

	if reasoning == 0 {
		t.Skip("model returned no reasoning summary, so there is nothing to replay")
	}

	second, err := model.Generate(t.Context(), &llm.Request{
		Messages: []llm.Message{
			question,
			first.Message,
			llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(calls[0].ID, calls[0].Name, json.RawMessage(`{"temperature_c":-3,"condition":"snow"}`), false)),
		},
		Tools: []llm.ToolDefinition{weather},
	})
	require.NoError(t, err, "the API rejected the replayed turn (%d reasoning parts)", reasoning)
	assert.NotEmpty(t, second.TextContent())
}
