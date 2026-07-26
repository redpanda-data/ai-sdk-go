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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelResolution(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		modelKey      string
		expectedModel string
	}{
		{
			name:          "claude-fable-5 family name resolves",
			modelKey:      "claude-fable-5",
			expectedModel: "claude-fable-5",
		},
		{
			name:          "claude-fable-5 timestamped preserves original",
			modelKey:      "claude-fable-5-20260604",
			expectedModel: "claude-fable-5-20260604",
		},
		{
			name:          "claude-sonnet-5 family name resolves",
			modelKey:      "claude-sonnet-5",
			expectedModel: "claude-sonnet-5",
		},
		{
			name:          "claude-sonnet-5 timestamped preserves original",
			modelKey:      "claude-sonnet-5-20260601",
			expectedModel: "claude-sonnet-5-20260601",
		},
		{
			name:          "claude-sonnet-4-6 family name resolves",
			modelKey:      "claude-sonnet-4-6",
			expectedModel: "claude-sonnet-4-6",
		},
		{
			name:          "claude-sonnet-4-5 family name resolves",
			modelKey:      "claude-sonnet-4-5",
			expectedModel: "claude-sonnet-4-5",
		},
		{
			name:          "claude-sonnet-4-5 timestamped preserves original",
			modelKey:      "claude-sonnet-4-5-20250929",
			expectedModel: "claude-sonnet-4-5-20250929",
		},
		{
			name:          "claude-haiku-4-5 family name resolves",
			modelKey:      "claude-haiku-4-5",
			expectedModel: "claude-haiku-4-5",
		},
		{
			name:          "claude-haiku-4-5 timestamped preserves original",
			modelKey:      "claude-haiku-4-5-20251001",
			expectedModel: "claude-haiku-4-5-20251001",
		},
		{
			name:          "claude-opus-4-6 family name resolves",
			modelKey:      "claude-opus-4-6",
			expectedModel: "claude-opus-4-6",
		},
		{
			name:          "claude-opus-4-8 family name resolves",
			modelKey:      "claude-opus-4-8",
			expectedModel: "claude-opus-4-8",
		},
		{
			name:          "claude-opus-4-8 timestamped preserves original",
			modelKey:      "claude-opus-4-8-20260528",
			expectedModel: "claude-opus-4-8-20260528",
		},
		{
			name:          "claude-opus-4-7 family name resolves",
			modelKey:      "claude-opus-4-7",
			expectedModel: "claude-opus-4-7",
		},
		{
			name:          "claude-opus-4-7 timestamped preserves original",
			modelKey:      "claude-opus-4-7-20260416",
			expectedModel: "claude-opus-4-7-20260416",
		},
		{
			name:          "claude-opus-4-5 family name resolves",
			modelKey:      "claude-opus-4-5",
			expectedModel: "claude-opus-4-5",
		},
		{
			name:          "claude-opus-4-5 timestamped preserves original",
			modelKey:      "claude-opus-4-5-20251101",
			expectedModel: "claude-opus-4-5-20251101",
		},
		{
			name:          "claude-opus-4-1 family name resolves",
			modelKey:      "claude-opus-4-1",
			expectedModel: "claude-opus-4-1",
		},
		{
			name:          "claude-opus-4-1 arbitrary timestamp resolves",
			modelKey:      "claude-opus-4-1-20240101",
			expectedModel: "claude-opus-4-1-20240101",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider, err := NewProvider("test-key")
			require.NoError(t, err)

			model, err := provider.NewModel(tt.modelKey)
			require.NoError(t, err)
			require.NotNil(t, model)

			m, ok := model.(*Model)
			require.True(t, ok)
			assert.Equal(t, tt.expectedModel, m.config.ModelName)
		})
	}
}

func TestCustomModelName(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	model, err := provider.NewModel(
		"claude-opus-4-1",
		WithCustomModelName("claude-opus-4-2-beta"),
	)
	require.NoError(t, err)
	require.NotNil(t, model)

	m, ok := model.(*Model)
	require.True(t, ok)

	assert.Equal(t, "claude-opus-4-1", m.config.ModelName)
	assert.Equal(t, "claude-opus-4-2-beta", m.config.CustomModelName)
	assert.Equal(t, int(200000), m.config.Constraints.MaxInputTokens)
}

func TestSonnet46ContextLimits(t *testing.T) {
	t.Parallel()

	def, ok := supportedModels[ModelClaudeSonnet46]
	require.True(t, ok)
	assert.Equal(t, 1_000_000, def.Constraints.MaxInputTokens)
	assert.Equal(t, 128_000, def.Constraints.MaxOutputTokens)
}

func TestUnsupportedModelRejected(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	_, err = provider.NewModel("claude-nonexistent-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported Anthropic model")
}

func TestCustomModelNameValidation(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	_, err = provider.NewModel(
		"claude-opus-4-1",
		WithCustomModelName(""),
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "custom model name cannot be empty")
}

func TestWithThinkingBudget(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	t.Run("valid budget on supported model", func(t *testing.T) {
		t.Parallel()

		model, err := provider.NewModel(ModelClaudeSonnet46, WithThinkingBudget(2048))
		require.NoError(t, err)

		m, ok := model.(*Model)
		require.True(t, ok)
		assert.True(t, m.config.EnableThinking, "WithThinkingBudget should implicitly enable thinking")
		require.NotNil(t, m.config.ThinkingBudget)
		assert.Equal(t, int64(2048), *m.config.ThinkingBudget)
	})

	t.Run("minimum budget enforced", func(t *testing.T) {
		t.Parallel()

		_, err := provider.NewModel(ModelClaudeSonnet46, WithThinkingBudget(512))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "thinking_budget must be at least 1024")
	})

	t.Run("rejected on model without thinking_budget support", func(t *testing.T) {
		t.Parallel()

		_, err := provider.NewModel(ModelClaudeSonnet45, WithThinkingBudget(2048))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "thinking_budget")
	})

	t.Run("rejected on Fable 5", func(t *testing.T) {
		t.Parallel()

		_, err := provider.NewModel(ModelClaudeFable5, WithThinkingBudget(2048))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "thinking_budget")
	})
}

func TestWithEffort(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	t.Run("valid effort levels on Sonnet 4.6", func(t *testing.T) {
		t.Parallel()

		for _, effort := range []Effort{EffortLow, EffortMedium, EffortHigh} {
			model, err := provider.NewModel(ModelClaudeSonnet46, WithEffort(effort))
			require.NoError(t, err)

			m, ok := model.(*Model)
			require.True(t, ok)
			require.NotNil(t, m.config.Effort)
			assert.Equal(t, effort, *m.config.Effort)
		}
	})

	t.Run("EffortMax rejected on Sonnet 4.6", func(t *testing.T) {
		t.Parallel()

		_, err := provider.NewModel(ModelClaudeSonnet46, WithEffort(EffortMax))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "does not support effort 'max'")
	})

	t.Run("EffortMax accepted on Opus 4.6", func(t *testing.T) {
		t.Parallel()

		model, err := provider.NewModel(ModelClaudeOpus46, WithEffort(EffortMax))
		require.NoError(t, err)

		m, ok := model.(*Model)
		require.True(t, ok)
		require.NotNil(t, m.config.Effort)
		assert.Equal(t, EffortMax, *m.config.Effort)
	})

	t.Run("all effort levels accepted on Sonnet 5", func(t *testing.T) {
		t.Parallel()

		for _, effort := range []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax} {
			model, err := provider.NewModel(ModelClaudeSonnet5, WithEffort(effort))
			require.NoError(t, err)

			m, ok := model.(*Model)
			require.True(t, ok)
			require.NotNil(t, m.config.Effort)
			assert.Equal(t, effort, *m.config.Effort)
		}
	})

	t.Run("all effort levels accepted on Fable 5", func(t *testing.T) {
		t.Parallel()

		for _, effort := range []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax} {
			model, err := provider.NewModel(ModelClaudeFable5, WithEffort(effort))
			require.NoError(t, err)

			m, ok := model.(*Model)
			require.True(t, ok)
			require.NotNil(t, m.config.Effort)
			assert.Equal(t, effort, *m.config.Effort)
		}
	})

	t.Run("rejected on model without effort support", func(t *testing.T) {
		t.Parallel()

		_, err := provider.NewModel(ModelClaudeSonnet45, WithEffort(EffortHigh))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "effort")
	})
}

func TestWithSpeed(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	t.Run("fast speed on Opus 4.8", func(t *testing.T) {
		t.Parallel()

		model, err := provider.NewModel(ModelClaudeOpus48, WithSpeed(SpeedFast))
		require.NoError(t, err)

		m, ok := model.(*Model)
		require.True(t, ok)
		require.NotNil(t, m.config.Speed)
		assert.Equal(t, SpeedFast, *m.config.Speed)
	})

	for _, model := range []string{ModelClaudeOpus46, ModelClaudeOpus47} {
		t.Run("standard speed on "+model, func(t *testing.T) {
			t.Parallel()

			got, err := provider.NewModel(model, WithSpeed(SpeedStandard))
			require.NoError(t, err)

			m, ok := got.(*Model)
			require.True(t, ok)
			require.NotNil(t, m.config.Speed)
			assert.Equal(t, SpeedStandard, *m.config.Speed)
		})
	}

	for _, model := range []string{
		ModelClaudeFable5,
		ModelClaudeSonnet46,
		ModelClaudeSonnet45,
	} {
		t.Run("rejected on "+model, func(t *testing.T) {
			t.Parallel()

			_, err := provider.NewModel(model, WithSpeed(SpeedFast))
			require.Error(t, err)
			assert.Contains(t, err.Error(), "speed")
		})
	}

	for _, model := range []string{ModelClaudeOpus46, ModelClaudeOpus47} {
		t.Run("fast rejected on "+model, func(t *testing.T) {
			t.Parallel()

			_, err := provider.NewModel(model, WithSpeed(SpeedFast))
			require.Error(t, err)
			assert.Contains(t, err.Error(), "speed")
		})
	}
}

func TestSamplingParametersRejected(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	options := []struct {
		name string
		opt  Option
		want string
	}{
		{name: "temperature", opt: WithTemperature(0.5), want: "temperature"},
		{name: "top_p", opt: WithTopP(0.9), want: "top_p"},
		{name: "top_k", opt: WithTopK(10), want: "top_k"},
	}

	for _, model := range []string{
		ModelClaudeOpus47,
		ModelClaudeOpus48,
		ModelClaudeSonnet5,
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			for _, tt := range options {
				t.Run(tt.name, func(t *testing.T) {
					t.Parallel()

					_, err := provider.NewModel(model, tt.opt)
					require.Error(t, err)
					assert.Contains(t, err.Error(), tt.want)
				})
			}
		})
	}
}

func TestFable5SamplingParameters(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("test-key")
	require.NoError(t, err)

	for _, tt := range []struct {
		name    string
		opt     Option
		wantErr bool
	}{
		{"default temperature", WithTemperature(1), false},
		{"non-default temperature", WithTemperature(0.5), true},
		{"minimum top_p", WithTopP(0.99), false},
		{"top_p below minimum", WithTopP(0.98), true},
		{"top_p maximum is exclusive", WithTopP(1), true},
		{"top_k unsupported", WithTopK(10), true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := provider.NewModel(ModelClaudeFable5, tt.opt)
			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
		})
	}
}
