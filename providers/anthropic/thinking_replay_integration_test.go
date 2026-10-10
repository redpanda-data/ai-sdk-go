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
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
)

// TestThinkingBlockBinding_ToolLoop_Integration runs a tool-use loop on a
// prefix-checked model against the live API: the bound request must be
// accepted, and the turn, signature-only thinking included, must replay with
// its tool result. Only a 400 fails it; an unavailable model skips.
func TestThinkingBlockBinding_ToolLoop_Integration(t *testing.T) {
	t.Parallel()

	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	provider, err := anthropic.NewProvider(apiKey, anthropic.WithTimeout(2*time.Minute))
	require.NoError(t, err)

	model, err := provider.NewModel(anthropic.ModelClaudeSonnet55, anthropic.WithMaxTokens(4096))
	require.NoError(t, err)

	weather := llm.ToolDefinition{
		Name:        "get_weather",
		Description: "Get the current weather for a city",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}
	question := llm.NewMessage(llm.RoleUser, llm.NewTextPart("What is the weather in Paris right now? Use the get_weather tool."))

	first, err := model.Generate(t.Context(), &llm.Request{
		Messages: []llm.Message{question},
		Tools:    []llm.ToolDefinition{weather},
	})
	if err != nil && !errors.Is(err, llm.ErrInvalidInput) {
		t.Skipf("%s unavailable: %v", anthropic.ModelClaudeSonnet55, err)
	}

	require.NoError(t, err, "the API rejected the bound request")

	calls := first.Message.ToolRequests()
	if len(calls) == 0 {
		t.Skip("model answered without calling the tool")
	}

	for _, part := range first.Message.Content {
		if rp, ok := part.(*llm.ReasoningPart); ok {
			assert.NotEmpty(t, rp.Signature, "replayable thinking keeps its signature")
		}
	}

	second, err := model.Generate(t.Context(), &llm.Request{
		Messages: []llm.Message{
			question,
			first.Message,
			llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(calls[0].ID, calls[0].Name, json.RawMessage(`{"temperature_c":18,"condition":"cloudy"}`), false)),
		},
		Tools: []llm.ToolDefinition{weather},
	})
	require.NoError(t, err, "the API rejected the replayed turn")
	assert.NotEmpty(t, second.TextContent())
}
