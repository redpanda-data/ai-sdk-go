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

package openaicompat

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
	defaultMax := 2048
	defaultFreq := 0.1
	defaultPres := 0.2
	defaultSeed := 42

	cfg := &Config{
		ModelName:        "deepseek-chat",
		Constraints:      llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 100000},
		Temperature:      &defaultTemp,
		TopP:             &defaultTopP,
		MaxTokens:        &defaultMax,
		FrequencyPenalty: &defaultFreq,
		PresencePenalty:  &defaultPres,
		Seed:             &defaultSeed,
		Stop:             []string{"END"},
	}

	mapper := NewRequestMapper(cfg)

	override := &llm.SamplingParams{
		Temperature:      new(0.9),
		TopP:             new(0.95),
		MaxOutputTokens:  new(1024),
		FrequencyPenalty: new(0.4),
		PresencePenalty:  new(0.5),
		Seed:             new(int64(99)),
		StopSequences:    []string{"STOP"},
	}

	apiReq, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: override,
	})
	require.NoError(t, err)

	require.True(t, apiReq.Temperature.Valid())
	assert.InDelta(t, 0.9, apiReq.Temperature.Value, 0.001)
	require.True(t, apiReq.TopP.Valid())
	assert.InDelta(t, 0.95, apiReq.TopP.Value, 0.001)
	require.True(t, apiReq.MaxTokens.Valid())
	assert.Equal(t, int64(1024), apiReq.MaxTokens.Value)
	require.True(t, apiReq.FrequencyPenalty.Valid())
	assert.InDelta(t, 0.4, apiReq.FrequencyPenalty.Value, 0.001)
	require.True(t, apiReq.PresencePenalty.Valid())
	assert.InDelta(t, 0.5, apiReq.PresencePenalty.Value, 0.001)
	require.True(t, apiReq.Seed.Valid())
	assert.Equal(t, int64(99), apiReq.Seed.Value)
	assert.Equal(t, []string{"STOP"}, apiReq.Stop.OfStringArray)
}

func TestRequestMapper_SamplingFallsBackToConfig(t *testing.T) {
	t.Parallel()

	defaultTemp := 0.5
	defaultMax := 2048
	defaultSeed := 42

	cfg := &Config{
		ModelName:   "deepseek-chat",
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 100000},
		Temperature: &defaultTemp,
		MaxTokens:   &defaultMax,
		Seed:        &defaultSeed,
	}

	mapper := NewRequestMapper(cfg)

	// Sampling sets only Temperature; Seed and MaxTokens should fall back to Config.
	apiReq, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{Temperature: new(0.9)},
	})
	require.NoError(t, err)

	require.True(t, apiReq.Temperature.Valid())
	assert.InDelta(t, 0.9, apiReq.Temperature.Value, 0.001)
	require.True(t, apiReq.MaxTokens.Valid())
	assert.Equal(t, int64(2048), apiReq.MaxTokens.Value)
	require.True(t, apiReq.Seed.Valid())
	assert.Equal(t, int64(42), apiReq.Seed.Value)
}

func TestRequestMapper_SamplingValidation(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:   "deepseek-chat",
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 2}, MaxOutputTokens: 8192, MaxInputTokens: 100000},
	}

	mapper := NewRequestMapper(cfg)

	_, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{Temperature: new(3.0)},
	})
	require.Error(t, err)
}
