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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestRequestMapper_SamplingOverridesConfig(t *testing.T) {
	t.Parallel()

	defaultTemp := 0.5
	defaultTopP := 0.8
	defaultTopK := int32(40)
	defaultMax := int32(2048)
	defaultPresence := float32(0.1)
	defaultFrequency := float32(0.2)

	cfg := &Config{
		ModelName:        ModelGemini35Flash,
		Constraints:      llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 1000000},
		Temperature:      &defaultTemp,
		TopP:             &defaultTopP,
		TopK:             &defaultTopK,
		MaxTokens:        &defaultMax,
		Stop:             []string{"END"},
		PresencePenalty:  &defaultPresence,
		FrequencyPenalty: &defaultFrequency,
	}

	mapper := NewRequestMapper(cfg)

	override := &llm.SamplingParams{
		Temperature:      new(0.9),
		TopP:             new(0.95),
		TopK:             new(64),
		MaxOutputTokens:  new(1024),
		StopSequences:    []string{"STOP"},
		PresencePenalty:  new(0.5),
		FrequencyPenalty: new(0.6),
	}

	_, config, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: override,
	})
	require.NoError(t, err)
	require.NotNil(t, config)
	assert.InDelta(t, float32(0.9), *config.Temperature, 0.001)
	assert.InDelta(t, float32(0.95), *config.TopP, 0.001)
	assert.InDelta(t, float32(64), *config.TopK, 0.001)
	assert.Equal(t, int32(1024), config.MaxOutputTokens)
	assert.Equal(t, []string{"STOP"}, config.StopSequences)
	assert.InDelta(t, float32(0.5), *config.PresencePenalty, 0.001)
	assert.InDelta(t, float32(0.6), *config.FrequencyPenalty, 0.001)
}

func TestRequestMapper_SamplingFallsBackToConfig(t *testing.T) {
	t.Parallel()

	defaultTemp := 0.5
	defaultMax := int32(2048)

	cfg := &Config{
		ModelName:   ModelGemini35Flash,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 1000000},
		Temperature: &defaultTemp,
		MaxTokens:   &defaultMax,
	}

	mapper := NewRequestMapper(cfg)

	_, config, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{TopP: new(0.7)},
	})
	require.NoError(t, err)
	require.NotNil(t, config)
	assert.InDelta(t, float32(0.5), *config.Temperature, 0.001)
	assert.InDelta(t, float32(0.7), *config.TopP, 0.001)
	assert.Equal(t, int32(2048), config.MaxOutputTokens)
}

func TestRequestMapper_SamplingValidation(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:   ModelGemini35Flash,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 1000000},
	}

	mapper := NewRequestMapper(cfg)

	_, _, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{Temperature: new(3.0)},
	})
	require.Error(t, err)
}
