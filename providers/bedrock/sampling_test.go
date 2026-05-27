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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestRequestMapper_SamplingOverridesConfig(t *testing.T) {
	t.Parallel()

	defaultTemp := 0.5
	defaultTopP := 0.8
	defaultMax := int32(2048)

	cfg := &Config{
		ModelName:   ModelClaudeSonnet46,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 1}, MaxOutputTokens: 8192},
		Temperature: &defaultTemp,
		TopP:        &defaultTopP,
		MaxTokens:   &defaultMax,
		Stop:        []string{"END"},
	}

	mapper := NewRequestMapper(cfg)

	// Override every field; expect provider to use overrides, not config.
	override := &llm.SamplingParams{
		Temperature:     new(0.9),
		TopP:            new(0.95),
		MaxOutputTokens: new(1024),
		StopSequences:   []string{"STOP"},
	}

	input, err := mapper.ToConverseInput(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: override,
	})
	require.NoError(t, err)
	require.NotNil(t, input.InferenceConfig)
	assert.InDelta(t, 0.9, *input.InferenceConfig.Temperature, 0.001, "Sampling.Temperature should override Config")
	assert.InDelta(t, 0.95, *input.InferenceConfig.TopP, 0.001, "Sampling.TopP should override Config")
	assert.Equal(t, int32(1024), *input.InferenceConfig.MaxTokens, "Sampling.MaxOutputTokens should override Config")
	assert.Equal(t, []string{"STOP"}, input.InferenceConfig.StopSequences, "Sampling.StopSequences should override Config")
}

func TestRequestMapper_SamplingFallsBackToConfig(t *testing.T) {
	t.Parallel()

	defaultTemp := 0.5
	defaultMax := int32(2048)

	cfg := &Config{
		ModelName:   ModelClaudeSonnet46,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 1}, MaxOutputTokens: 8192},
		Temperature: &defaultTemp,
		MaxTokens:   &defaultMax,
	}

	mapper := NewRequestMapper(cfg)

	// Sampling is non-nil but only sets TopP; other fields should fall back to Config.
	input, err := mapper.ToConverseInput(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{TopP: new(0.7)},
	})
	require.NoError(t, err)
	require.NotNil(t, input.InferenceConfig)
	assert.InDelta(t, 0.5, *input.InferenceConfig.Temperature, 0.001)
	assert.InDelta(t, 0.7, *input.InferenceConfig.TopP, 0.001)
	assert.Equal(t, int32(2048), *input.InferenceConfig.MaxTokens)
}

func TestRequestMapper_SamplingValidation(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:   ModelClaudeSonnet46,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 1}, MaxOutputTokens: 8192},
	}

	mapper := NewRequestMapper(cfg)

	_, err := mapper.ToConverseInput(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{Temperature: new(2.0)},
	})
	require.Error(t, err, "out-of-range temperature should be rejected")
}
