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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestRequestMapper_WedgedSessionReplay reproduces a session-replay wedge.
//
// When a streaming turn is cut off mid-tool-call (stop_reason=max_tokens
// during input_json_delta accumulation), the partial tool arguments get
// persisted to session state as truncated JSON (e.g. `{"query":`). The agent
// notices at tool-execution time and records an error tool_result, so the
// turn itself completes, but the broken assistant tool_use block stays in
// history.
//
// Every subsequent turn then rebuilds the full conversation and dies in
// mapAssistantMessage with "unexpected end of JSON input", permanently
// wedging the session.
func TestRequestMapper_WedgedSessionReplay(t *testing.T) {
	t.Parallel()

	mapper := NewRequestMapper(&Config{
		ModelName: "claude-opus-4-5-20250929",
		MaxTokens: 4096,
	})

	// Mirrors the shape of the poisoned session at the point of replay:
	// user prompt -> assistant tool_use with truncated args -> user tool_result
	// carrying the original parse error -> user text asking to try again.
	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role:    llm.RoleUser,
				Content: []*llm.Part{llm.NewTextPart("rerun the analysis")},
			},
			{
				Role: llm.RoleAssistant,
				Content: []*llm.Part{
					llm.NewToolRequestPart(&llm.ToolRequest{
						ID:        "toolu_truncated",
						Name:      "db_query",
						Arguments: json.RawMessage(`{"query":`), // truncated mid-stream
					}),
				},
			},
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewToolResponsePart(&llm.ToolResponse{
						ID:    "toolu_truncated",
						Name:  "db_query",
						Error: `tool "db_query" arguments must be a JSON object: unexpected end of JSON input`,
					}),
				},
			},
			{
				Role:    llm.RoleUser,
				Content: []*llm.Part{llm.NewTextPart("run the analysis again")},
			},
		},
	}

	apiReq, err := mapper.ToProvider(req)
	require.NoError(t, err, "wedged session must be mappable so the conversation can recover")
	require.Len(t, apiReq.Messages, 4)

	// The truncated tool_use survives to Anthropic as a valid JSON object.
	// An empty object is acceptable: the paired tool_result carries the real
	// parse error, and the model has enough context to retry.
	assistant := apiReq.Messages[1]
	require.Len(t, assistant.Content, 1)

	toolUse := assistant.Content[0].OfToolUse
	require.NotNil(t, toolUse, "assistant tool_use block must be emitted")
	assert.Equal(t, "toolu_truncated", toolUse.ID)

	input, ok := toolUse.Input.(map[string]any)
	require.True(t, ok, "tool_use input must be a JSON object, got %T", toolUse.Input)
	assert.Empty(t, input, "truncated arguments should collapse to an empty object")
}

// TestRequestMapper_EmptyToolArguments covers the no-arg tool call case:
// when ToolRequest.Arguments is empty bytes (the wire form for tools with
// no parameters, also the value FinalizeToolArgs coerces empty streaming
// accumulations to), the mapper must emit a valid empty object rather than
// failing message mapping.
func TestRequestMapper_EmptyToolArguments(t *testing.T) {
	t.Parallel()

	mapper := NewRequestMapper(&Config{
		ModelName: "claude-opus-4-5-20250929",
		MaxTokens: 4096,
	})

	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleAssistant,
				Content: []*llm.Part{
					llm.NewToolRequestPart(&llm.ToolRequest{
						ID:        "toolu_noargs",
						Name:      "list_users",
						Arguments: nil,
					}),
				},
			},
		},
	}

	apiReq, err := mapper.ToProvider(req)
	require.NoError(t, err)
	require.Len(t, apiReq.Messages, 1)

	toolUse := apiReq.Messages[0].Content[0].OfToolUse
	require.NotNil(t, toolUse)

	input, ok := toolUse.Input.(map[string]any)
	require.True(t, ok)
	assert.Empty(t, input)
}
