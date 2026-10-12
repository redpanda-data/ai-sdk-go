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
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestReasoningEffortMapsToConverseRequest(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	model, err := provider.NewModel(ModelClaudeOpus47US, WithReasoningEffort(ReasoningEffortMedium))
	require.NoError(t, err)

	bedrockModel, ok := model.(*Model)
	require.True(t, ok)

	input, err := bedrockModel.requestMapper.ToConverseInput(&llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Solve this.")),
		},
	})
	require.NoError(t, err)
	require.NotNil(t, input.AdditionalModelRequestFields)

	payload, err := input.AdditionalModelRequestFields.MarshalSmithyDocument()
	require.NoError(t, err)

	var fields map[string]any
	require.NoError(t, json.Unmarshal(payload, &fields))
	assert.Equal(t, map[string]any{
		"thinking": map[string]any{
			"type": "adaptive",
		},
		"output_config": map[string]any{
			"effort": "medium",
		},
	}, fields)
}

func TestThinkingWithoutBudgetOrEffortMapsToAdaptive(t *testing.T) {
	t.Parallel()

	mapper := NewRequestMapper(&Config{
		ModelName:      ModelClaudeOpus47US,
		APIModelID:     ModelClaudeOpus47US,
		EnableThinking: true,
	})

	input, err := mapper.ToConverseInput(&llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Solve this.")),
		},
	})
	require.NoError(t, err)

	payload, err := input.AdditionalModelRequestFields.MarshalSmithyDocument()
	require.NoError(t, err)

	var fields map[string]any
	require.NoError(t, json.Unmarshal(payload, &fields))
	assert.Equal(t, map[string]any{
		"thinking": map[string]any{
			"type": "adaptive",
		},
	}, fields)
}

func TestSignatureOnlyReasoningRoundTrips(t *testing.T) {
	t.Parallel()

	responseMapper := NewResponseMapper(catalog.Offering{})
	part := responseMapper.mapReasoningBlock(&types.ReasoningContentBlockMemberReasoningText{
		Value: types.ReasoningTextBlock{
			Text:      aws.String(""),
			Signature: aws.String("opaque-signature"),
		},
	})
	require.NotNil(t, part)

	reasoning, ok := part.(*llm.ReasoningPart)
	require.True(t, ok)
	assert.Empty(t, reasoning.Text)
	assert.Equal(t, "opaque-signature", reasoning.Signature)

	requestMapper := NewRequestMapper(&Config{})
	message, err := requestMapper.mapAssistantMessage(llm.NewMessage(llm.RoleAssistant, reasoning))
	require.NoError(t, err)
	require.Len(t, message.Content, 1)

	block, ok := message.Content[0].(*types.ContentBlockMemberReasoningContent)
	require.True(t, ok)

	reasoningText, ok := block.Value.(*types.ReasoningContentBlockMemberReasoningText)
	require.True(t, ok)
	require.NotNil(t, reasoningText.Value.Text)
	require.NotNil(t, reasoningText.Value.Signature)
	assert.Empty(t, *reasoningText.Value.Text)
	assert.Equal(t, "opaque-signature", *reasoningText.Value.Signature)
}

func TestSignatureOnlyReasoningSurvivesStreamingFinalization(t *testing.T) {
	t.Parallel()

	acc := &contentBlockAccumulator{}
	event, yielded := processReasoningDelta(acc, &types.ContentBlockDeltaMemberReasoningContent{
		Value: &types.ReasoningContentBlockDeltaMemberSignature{
			Value: "opaque-signature",
		},
	}, 0)
	assert.False(t, yielded)
	assert.Nil(t, event)

	parts := (&Model{}).buildFinalParts(map[int]*contentBlockAccumulator{0: acc})
	require.Len(t, parts, 1)

	reasoning, ok := parts[0].(*llm.ReasoningPart)
	require.True(t, ok)
	assert.Empty(t, reasoning.Text)
	assert.Equal(t, "opaque-signature", reasoning.Signature)
}

// redactedPayload is binary on purpose: Converse returns redactedContent as
// bytes, and they must survive JSON persistence of the session unchanged.
var redactedPayload = []byte{0x00, 0xff, 0x10, 'x', 0xc3}

func TestRedactedReasoningRoundTrips(t *testing.T) {
	t.Parallel()

	part := NewResponseMapper(catalog.Offering{}).mapReasoningBlock(&types.ReasoningContentBlockMemberRedactedContent{
		Value: redactedPayload,
	})
	require.NotNil(t, part)

	reasoning, ok := part.(*llm.ReasoningPart)
	require.True(t, ok)
	assert.Equal(t, map[string]any{"redacted": true, "redacted_provider": "aws.bedrock"}, reasoning.Metadata)

	assertReplaysRedacted(t, reasoning)
}

func TestRedactedReasoningSurvivesStreamingFinalization(t *testing.T) {
	t.Parallel()

	acc := &contentBlockAccumulator{}
	event, yielded := processReasoningDelta(acc, &types.ContentBlockDeltaMemberReasoningContent{
		Value: &types.ReasoningContentBlockDeltaMemberRedactedContent{
			Value: redactedPayload,
		},
	}, 0)
	assert.False(t, yielded)
	assert.Nil(t, event)

	parts := (&Model{}).buildFinalParts(map[int]*contentBlockAccumulator{0: acc})
	require.Len(t, parts, 1)

	reasoning, ok := parts[0].(*llm.ReasoningPart)
	require.True(t, ok)

	assertReplaysRedacted(t, reasoning)
}

// TestRequestMapper_SkipsUnreplayableRedactedReasoning covers redacted parts
// the mapper cannot send back as the original redactedContent. Dropping the
// block lets the conversation continue; failing the request would fail every
// later turn of a persisted session too.
func TestRequestMapper_SkipsUnreplayableRedactedReasoning(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		part *llm.ReasoningPart
	}{
		{
			name: "missing data",
			part: &llm.ReasoningPart{Text: "[redacted thinking]", Metadata: map[string]any{"redacted": true, "redacted_provider": "aws.bedrock"}},
		},
		{
			name: "data that is not base64",
			part: &llm.ReasoningPart{Text: "[redacted thinking]", Signature: "not base64!", Metadata: map[string]any{"redacted": true, "redacted_provider": "aws.bedrock"}},
		},
		{
			name: "persisted before the provider stamp",
			part: &llm.ReasoningPart{Text: "[redacted thinking]", Signature: "AP8QeMM=", Metadata: map[string]any{"redacted": true}},
		},
		{
			// Anthropic stores the redacted_thinking data string verbatim. It
			// can decode as base64, so only the stamp keeps it off the wire.
			name: "produced by Anthropic",
			part: &llm.ReasoningPart{
				Text:      "[redacted thinking]",
				Signature: "RW5jcnlwdGVkIHJlYXNvbmluZw==",
				Metadata:  map[string]any{"redacted": true, "redacted_provider": "anthropic"},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			message, err := NewRequestMapper(&Config{}).mapAssistantMessage(
				llm.NewMessage(llm.RoleAssistant, tc.part, llm.NewTextPart("hello")),
			)
			require.NoError(t, err)
			assert.Equal(t, []types.ContentBlock{&types.ContentBlockMemberText{Value: "hello"}}, message.Content)
		})
	}
}

// assertReplaysRedacted persists part the way a session store does and checks
// the request mapper sends it back as redactedContent with the original bytes.
func assertReplaysRedacted(t *testing.T, part *llm.ReasoningPart) {
	t.Helper()

	persisted, err := json.Marshal(llm.NewMessage(llm.RoleAssistant, part))
	require.NoError(t, err)

	var restored llm.Message
	require.NoError(t, json.Unmarshal(persisted, &restored))

	message, err := NewRequestMapper(&Config{}).mapAssistantMessage(restored)
	require.NoError(t, err)
	require.Len(t, message.Content, 1)

	block, ok := message.Content[0].(*types.ContentBlockMemberReasoningContent)
	require.True(t, ok)

	redacted, ok := block.Value.(*types.ReasoningContentBlockMemberRedactedContent)
	require.True(t, ok, "redacted reasoning must not be replayed as reasoningText")
	assert.Equal(t, redactedPayload, redacted.Value)
}

func TestModelThinkingCapabilities(t *testing.T) {
	t.Parallel()

	provider := &Provider{}

	tests := []struct {
		model            string
		efforts          []ReasoningEffort
		supportsAdaptive bool
		supportsBudget   bool
	}{
		{
			model:            ModelClaudeFable51US,
			efforts:          []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
			supportsAdaptive: true,
		},
		{
			model:            ModelClaudeFable5US,
			efforts:          []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
			supportsAdaptive: true,
		},
		{
			model:            ModelClaudeSonnet5US,
			efforts:          []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
			supportsAdaptive: true,
		},
		{
			model:            ModelClaudeOpus48US,
			efforts:          []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
			supportsAdaptive: true,
		},
		{
			model:            ModelClaudeOpus47US,
			efforts:          []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax},
			supportsAdaptive: true,
		},
		{
			model:            ModelClaudeOpus46US,
			efforts:          []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortMax},
			supportsAdaptive: true,
			supportsBudget:   true,
		},
		{
			model:            ModelClaudeSonnet46US,
			efforts:          []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortMax},
			supportsAdaptive: true,
			supportsBudget:   true,
		},
		{
			model:          ModelClaudeSonnet45US,
			supportsBudget: true,
		},
		{
			model: ModelNova2LiteUS,
		},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			t.Parallel()

			model, err := provider.NewModel(tt.model)
			require.NoError(t, err)

			bedrockModel, ok := model.(*Model)
			require.True(t, ok)
			assert.Equal(t, tt.efforts, bedrockModel.SupportedReasoningEfforts())
			assert.Equal(t, tt.supportsAdaptive, bedrockModel.SupportsAdaptiveThinking())
			assert.Equal(t, tt.supportsBudget, bedrockModel.SupportsThinkingBudget())
		})
	}
}

func TestNewModelRejectsUnsupportedThinkingConfiguration(t *testing.T) {
	t.Parallel()

	provider := &Provider{}

	tests := []struct {
		name      string
		model     string
		option    Option
		wantError string
	}{
		{
			name:      "manual budget on adaptive-only model",
			model:     ModelClaudeFable5US,
			option:    WithThinking(4096),
			wantError: "does not support a manual thinking budget",
		},
		{
			name:      "reasoning effort on budget-only model",
			model:     ModelClaudeSonnet45US,
			option:    WithReasoningEffort(ReasoningEffortLow),
			wantError: "does not support reasoning effort",
		},
		{
			name:      "unsupported effort",
			model:     ModelClaudeSonnet46US,
			option:    WithReasoningEffort(ReasoningEffortXHigh),
			wantError: "does not support reasoning effort",
		},
		{
			name:      "unknown effort",
			model:     ModelClaudeOpus47US,
			option:    WithReasoningEffort(ReasoningEffort("extreme")),
			wantError: "does not support reasoning effort",
		},
		{
			name:      "Anthropic thinking mode on non-Anthropic model",
			model:     ModelNova2LiteUS,
			option:    WithThinking(4096),
			wantError: "does not support a manual thinking budget",
		},
		{
			name:      "reasoning effort on mantle model without effort control",
			model:     ModelGemma4E2B,
			option:    WithReasoningEffort(ReasoningEffortLow),
			wantError: "does not support reasoning effort",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := provider.NewModel(tt.model, tt.option)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantError)
		})
	}
}

func TestNewModelRejectsReasoningEffortAndBudgetTogether(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	_, err := provider.NewModel(
		ModelClaudeOpus46US,
		WithReasoningEffort(ReasoningEffortHigh),
		WithThinking(4096),
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "reasoning effort and a manual thinking budget cannot be combined")
}

func TestThinkingBudgetRequiresProviderMinimum(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	_, err := provider.NewModel(ModelClaudeSonnet45US, WithThinking(1023))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "budget_tokens must be at least 1024")
}

func TestThinkingBudgetMapsToConverseRequest(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	model, err := provider.NewModel(ModelClaudeSonnet45US, WithThinking(4096))
	require.NoError(t, err)

	bedrockModel, ok := model.(*Model)
	require.True(t, ok)

	input, err := bedrockModel.requestMapper.ToConverseInput(&llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Solve this.")),
		},
	})
	require.NoError(t, err)

	payload, err := input.AdditionalModelRequestFields.MarshalSmithyDocument()
	require.NoError(t, err)

	var fields map[string]any
	require.NoError(t, json.Unmarshal(payload, &fields))
	assert.Equal(t, map[string]any{
		"thinking": map[string]any{
			"type":          "enabled",
			"budget_tokens": float64(4096),
		},
	}, fields)
}
