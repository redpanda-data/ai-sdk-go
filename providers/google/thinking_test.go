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

package google

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestReasoningEffortMapsToGemini3Request(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider(context.Background(), "test-api-key")
	require.NoError(t, err)

	model, err := provider.NewModel(ModelGemini3FlashPreview, WithReasoningEffort(ReasoningEffortMedium))
	require.NoError(t, err)

	googleModel, ok := model.(*Model)
	require.True(t, ok)

	_, config, err := googleModel.requestMapper.ToProvider(&llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Solve this.")),
		},
	})
	require.NoError(t, err)
	require.NotNil(t, config.ThinkingConfig)
	assert.Equal(t, "MEDIUM", string(config.ThinkingConfig.ThinkingLevel))
	assert.Nil(t, config.ThinkingConfig.ThinkingBudget)

	payload, err := json.Marshal(config)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, json.Unmarshal(payload, &wire))
	assert.Equal(t, map[string]any{
		"includeThoughts": true,
		"thinkingLevel":   "MEDIUM",
	}, wire["thinkingConfig"])
}

func TestModelSupportedReasoningEfforts(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider(context.Background(), "test-api-key")
	require.NoError(t, err)

	tests := []struct {
		model string
		want  []ReasoningEffort
	}{
		{
			model: ModelGemini35Flash,
			want:  []ReasoningEffort{ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		},
		{
			model: ModelGemini31ProPreview,
			want:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		},
		{
			model: ModelGemini3ProPreview,
			want:  []ReasoningEffort{ReasoningEffortLow, ReasoningEffortHigh},
		},
		{
			model: ModelGemini3FlashPreview,
			want:  []ReasoningEffort{ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh},
		},
		{
			model: ModelGemini25Flash,
			want:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			t.Parallel()

			model, modelErr := provider.NewModel(tt.model)
			require.NoError(t, modelErr)

			googleModel, ok := model.(*Model)
			require.True(t, ok)
			assert.Equal(t, tt.want, googleModel.SupportedReasoningEfforts())
		})
	}
}

func TestNewModelRejectsUnsupportedReasoningEffort(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider(context.Background(), "test-api-key")
	require.NoError(t, err)

	tests := []struct {
		name   string
		model  string
		effort ReasoningEffort
	}{
		{
			name:   "unsupported effort for Gemini 3",
			model:  ModelGemini3ProPreview,
			effort: ReasoningEffortMedium,
		},
		{
			name:   "reasoning effort on Gemini 2.5",
			model:  ModelGemini25Flash,
			effort: ReasoningEffortLow,
		},
		{
			name:   "unknown effort",
			model:  ModelGemini3FlashPreview,
			effort: ReasoningEffort("extreme"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, modelErr := provider.NewModel(tt.model, WithReasoningEffort(tt.effort))
			require.Error(t, modelErr)
			assert.Contains(t, modelErr.Error(), "does not support reasoning effort")
		})
	}
}

func TestThinkingBudgetMapsToGemini25Request(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider(context.Background(), "test-api-key")
	require.NoError(t, err)

	model, err := provider.NewModel(ModelGemini25Flash, WithThinkingBudget(4096))
	require.NoError(t, err)

	googleModel, ok := model.(*Model)
	require.True(t, ok)

	_, config, err := googleModel.requestMapper.ToProvider(&llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Solve this.")),
		},
	})
	require.NoError(t, err)
	require.NotNil(t, config.ThinkingConfig)
	require.NotNil(t, config.ThinkingConfig.ThinkingBudget)
	assert.Equal(t, int32(4096), *config.ThinkingConfig.ThinkingBudget)
	assert.Empty(t, config.ThinkingConfig.ThinkingLevel)
}

func TestNewModelRejectsUnsupportedThinkingBudget(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider(context.Background(), "test-api-key")
	require.NoError(t, err)

	tests := []struct {
		name   string
		model  string
		budget int32
	}{
		{
			name:   "budget on Gemini 3",
			model:  ModelGemini3FlashPreview,
			budget: 4096,
		},
		{
			name:   "Gemini 2.5 Pro cannot disable thinking",
			model:  ModelGemini25Pro,
			budget: 0,
		},
		{
			name:   "Gemini 2.5 Flash Lite budget below minimum",
			model:  ModelGemini25FlashLite,
			budget: 1,
		},
		{
			name:   "Gemini 2.5 Flash budget above maximum",
			model:  ModelGemini25Flash,
			budget: 24577,
		},
		{
			name:   "unknown negative budget",
			model:  ModelGemini25Flash,
			budget: -2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, modelErr := provider.NewModel(tt.model, WithThinkingBudget(tt.budget))
			require.Error(t, modelErr)
			assert.Contains(t, modelErr.Error(), "does not support thinking budget")
		})
	}
}

func TestNewModelRejectsReasoningEffortAndBudgetTogether(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider(context.Background(), "test-api-key")
	require.NoError(t, err)

	_, err = provider.NewModel(
		ModelGemini3FlashPreview,
		WithReasoningEffort(ReasoningEffortLow),
		WithThinkingBudget(4096),
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "reasoning effort and a thinking budget cannot be combined")
}
