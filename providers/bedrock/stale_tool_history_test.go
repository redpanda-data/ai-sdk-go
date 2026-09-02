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
	"context"
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestToConverseInput_StaleToolHistoryWithoutToolConfig is the Bedrock parity
// regression for the toolConfig-vs-history mismatch: Bedrock's Converse API
// rejects a request whose messages contain toolUse/toolResult blocks unless
// toolConfig is set on that same request, even when the current turn has no
// tools to offer — e.g. a tool used earlier in the conversation has since
// been removed from the agent. The mapper must flatten those blocks to text
// instead of sending a request Bedrock will reject, mirroring langchain-aws's
// fix for the same constraint (PR #595).
func TestToConverseInput_StaleToolHistoryWithoutToolConfig(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{Region: "us-east-1"}), WithCachingDisabled())
	require.NoError(t, err)

	model, err := p.NewModel(ModelClaudeSonnet46)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("what's the weather in Berlin?")}},
			{Role: llm.RoleAssistant, Content: []llm.Part{
				llm.NewToolRequestPart("call_1", "get_weather", json.RawMessage(`{"city":"Berlin"}`)),
			}},
			{Role: llm.RoleUser, Content: []llm.Part{
				llm.NewToolResponsePart("call_1", "get_weather", json.RawMessage(`{"tempC":18}`), false),
			}},
		},
		// No Tools: get_weather has since been removed from the agent.
	}

	input, err := m.requestMapper.ToConverseInput(req)
	require.NoError(t, err)

	assert.Nil(t, input.ToolConfig, "toolConfig must stay unset when the current turn has no tools")

	for _, msg := range input.Messages {
		for _, block := range msg.Content {
			_, isToolUse := block.(*types.ContentBlockMemberToolUse)
			_, isToolResult := block.(*types.ContentBlockMemberToolResult)
			assert.False(t, isToolUse || isToolResult, "tool blocks must be flattened to text when toolConfig is absent")
		}
	}

	assistant := input.Messages[1]
	require.Len(t, assistant.Content, 1)
	toolUseText, ok := assistant.Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, `[Called get_weather with parameters: {"city":"Berlin"}]`, toolUseText.Value)

	toolResultMsg := input.Messages[2]
	require.Len(t, toolResultMsg.Content, 1)
	toolResultText, ok := toolResultMsg.Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, `[Tool output: {"tempC":18}]`, toolResultText.Value)
}

// TestToConverseInput_ToolHistoryPreservedWithToolConfig proves the
// flattening only kicks in when toolConfig would otherwise be absent: with a
// live tool list, historical tool blocks must round-trip untouched.
func TestToConverseInput_ToolHistoryPreservedWithToolConfig(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{Region: "us-east-1"}), WithCachingDisabled())
	require.NoError(t, err)

	model, err := p.NewModel(ModelClaudeSonnet46)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, Content: []llm.Part{
				llm.NewToolRequestPart("call_1", "get_weather", json.RawMessage(`{"city":"Berlin"}`)),
			}},
			{Role: llm.RoleUser, Content: []llm.Part{
				llm.NewToolResponsePart("call_1", "get_weather", json.RawMessage(`{"tempC":18}`), false),
			}},
		},
		Tools: []llm.ToolDefinition{
			{Name: "get_weather", Description: "Get the weather", Parameters: json.RawMessage(`{"type":"object"}`)},
		},
	}

	input, err := m.requestMapper.ToConverseInput(req)
	require.NoError(t, err)
	require.NotNil(t, input.ToolConfig)

	assistant := input.Messages[0]
	require.Len(t, assistant.Content, 1)
	_, ok = assistant.Content[0].(*types.ContentBlockMemberToolUse)
	assert.True(t, ok, "tool blocks must stay structured when toolConfig is present")

	toolResultMsg := input.Messages[1]
	require.Len(t, toolResultMsg.Content, 1)
	_, ok = toolResultMsg.Content[0].(*types.ContentBlockMemberToolResult)
	assert.True(t, ok, "tool blocks must stay structured when toolConfig is present")
}

// TestConvertToolBlocksToText_NoParameters covers the falsy-parameters branch:
// a ToolUse block with an empty input map renders without a "with
// parameters" clause.
func TestConvertToolBlocksToText_NoParameters(t *testing.T) {
	t.Parallel()

	messages := []types.Message{
		{
			Role: types.ConversationRoleAssistant,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberToolUse{
					Value: types.ToolUseBlock{
						ToolUseId: aws.String("call_1"),
						Name:      aws.String("ping"),
						Input:     document.NewLazyDocument(map[string]any{}),
					},
				},
			},
		},
	}

	converted := convertToolBlocksToText(messages)

	require.Len(t, converted, 1)
	require.Len(t, converted[0].Content, 1)
	text, ok := converted[0].Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, "[Called ping]", text.Value)
}

// TestConvertToolBlocksToText_EmptyToolResultIsDropped covers the
// empty-result branch: a ToolResult block with no representable content is
// dropped rather than emitted as an empty "[Tool output: ]" block.
func TestConvertToolBlocksToText_EmptyToolResultIsDropped(t *testing.T) {
	t.Parallel()

	messages := []types.Message{
		{
			Role: types.ConversationRoleUser,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberToolResult{
					Value: types.ToolResultBlock{
						ToolUseId: aws.String("call_1"),
						Content:   []types.ToolResultContentBlock{},
					},
				},
			},
		},
	}

	converted := convertToolBlocksToText(messages)

	require.Len(t, converted, 1)
	assert.Empty(t, converted[0].Content)
}
