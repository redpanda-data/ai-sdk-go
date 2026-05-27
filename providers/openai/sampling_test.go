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
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestRequestMapper_SamplingOverridesConfig(t *testing.T) {
	t.Parallel()

	defaultTemp := 0.5
	defaultMax := 2048

	cfg := &Config{
		ModelName:   ModelGPT5,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 400000},
		Temperature: &defaultTemp,
		MaxTokens:   &defaultMax,
	}

	mapper := NewRequestMapper(cfg)

	override := &llm.SamplingParams{
		Temperature:     new(0.9),
		MaxOutputTokens: new(1024),
	}

	apiReq, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: override,
	})
	require.NoError(t, err)
	require.True(t, apiReq.Temperature.Valid())
	assert.InDelta(t, 0.9, apiReq.Temperature.Value, 0.001)
	require.True(t, apiReq.MaxOutputTokens.Valid())
	assert.Equal(t, int64(1024), apiReq.MaxOutputTokens.Value)
}

func TestRequestMapper_SamplingFallsBackToConfig(t *testing.T) {
	t.Parallel()

	defaultTemp := 0.5
	defaultMax := 2048

	cfg := &Config{
		ModelName:   ModelGPT5,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 400000},
		Temperature: &defaultTemp,
		MaxTokens:   &defaultMax,
	}

	mapper := NewRequestMapper(cfg)

	// Sampling sets only MaxOutputTokens; Temperature should fall back to Config.
	apiReq, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{MaxOutputTokens: new(512)},
	})
	require.NoError(t, err)
	require.True(t, apiReq.Temperature.Valid())
	assert.InDelta(t, 0.5, apiReq.Temperature.Value, 0.001)
	require.True(t, apiReq.MaxOutputTokens.Valid())
	assert.Equal(t, int64(512), apiReq.MaxOutputTokens.Value)
}

func TestRequestMapper_SamplingValidation(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:   ModelGPT5,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 400000},
	}

	mapper := NewRequestMapper(cfg)

	_, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{Temperature: new(3.0)},
	})
	require.Error(t, err)
}

// TestRequestMapper_TopPWired verifies the TopP sampling override
// reaches apiReq.TopP.
func TestRequestMapper_TopPWired(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:   ModelGPT5,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 400000},
	}

	mapper := NewRequestMapper(cfg)

	apiReq, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{TopP: new(0.3)},
	})
	require.NoError(t, err)
	require.True(t, apiReq.TopP.Valid())
	assert.InDelta(t, 0.3, apiReq.TopP.Value, 0.001)
}

// TestRequestMapper_LogProbsWired verifies WithLogProbs/WithTopLogProbs
// produce the right Include entry and TopLogprobs setting.
func TestRequestMapper_LogProbsWired(t *testing.T) {
	t.Parallel()

	logProbs := true
	topN := 5
	cfg := &Config{
		ModelName:   ModelGPT5,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 400000},
		LogProbs:    &logProbs,
		TopLogProbs: &topN,
	}

	mapper := NewRequestMapper(cfg)

	apiReq, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
	})
	require.NoError(t, err)
	assert.Contains(t, apiReq.Include, responses.ResponseIncludable(includeOutputTextLogprobs))
	require.True(t, apiReq.TopLogprobs.Valid())
	assert.Equal(t, int64(5), apiReq.TopLogprobs.Value)
}

// TestRequestMapper_RejectUnsupportedSampling verifies that explicitly
// setting a knob the Responses API does not support produces an error
// rather than silently dropping the value.
func TestRequestMapper_RejectUnsupportedSampling(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:   ModelGPT5,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 400000},
	}

	cases := []struct {
		name     string
		sampling *llm.SamplingParams
	}{
		{"seed", &llm.SamplingParams{Seed: new(int64(42))}},
		{"presence_penalty", &llm.SamplingParams{PresencePenalty: new(0.5)}},
		{"frequency_penalty", &llm.SamplingParams{FrequencyPenalty: new(0.5)}},
		{"stop_sequences", &llm.SamplingParams{StopSequences: []string{"END"}}},
		{"top_k", &llm.SamplingParams{TopK: new(40)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mapper := NewRequestMapper(cfg)

			_, err := mapper.ToProvider(&llm.Request{
				Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
				Sampling: tc.sampling,
			})
			require.Error(t, err)
			assert.ErrorIs(t, err, llm.ErrInvalidInput)
		})
	}
}

// TestRequestMapper_MaxOutputTokensValidated verifies the resolved
// MaxOutputTokens is checked against Constraints.MaxOutputTokens (not
// MaxInputTokens) at request-mapping time.
func TestRequestMapper_MaxOutputTokensValidated(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:   ModelGPT5,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 1000, MaxInputTokens: 400000},
	}

	mapper := NewRequestMapper(cfg)

	// Within limit.
	_, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{MaxOutputTokens: new(500)},
	})
	require.NoError(t, err)

	// Over the output limit.
	_, err = mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{MaxOutputTokens: new(2000)},
	})
	require.Error(t, err)
	assert.ErrorIs(t, err, llm.ErrInvalidInput)
}
