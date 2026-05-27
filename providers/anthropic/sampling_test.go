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

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestRequestMapper_SamplingOverridesConfig(t *testing.T) {
	t.Parallel()

	defaultTemp := 0.5
	defaultTopP := 0.8
	defaultTopK := 40

	cfg := &Config{
		ModelName:   ModelClaudeSonnet46,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 1}, MaxOutputTokens: 8192},
		Temperature: &defaultTemp,
		TopP:        &defaultTopP,
		TopK:        &defaultTopK,
		MaxTokens:   2048,
		Stop:        []string{"END"},
	}

	mapper := NewRequestMapper(cfg)

	override := &llm.SamplingParams{
		Temperature:     new(0.9),
		TopP:            new(0.95),
		TopK:            new(64),
		MaxOutputTokens: new(1024),
		StopSequences:   []string{"STOP"},
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
	require.True(t, apiReq.TopK.Valid())
	assert.Equal(t, int64(64), apiReq.TopK.Value)
	assert.Equal(t, int64(1024), apiReq.MaxTokens)
	assert.Equal(t, []string{"STOP"}, apiReq.StopSequences)
}

func TestRequestMapper_SamplingFallsBackToConfig(t *testing.T) {
	t.Parallel()

	defaultTemp := 0.5

	cfg := &Config{
		ModelName:   ModelClaudeSonnet46,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 1}, MaxOutputTokens: 8192},
		Temperature: &defaultTemp,
		MaxTokens:   2048,
	}

	mapper := NewRequestMapper(cfg)

	// Sampling sets only TopP; Temperature/MaxTokens come from Config.
	apiReq, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{TopP: new(0.7)},
	})
	require.NoError(t, err)

	require.True(t, apiReq.Temperature.Valid())
	assert.InDelta(t, 0.5, apiReq.Temperature.Value, 0.001)
	require.True(t, apiReq.TopP.Valid())
	assert.InDelta(t, 0.7, apiReq.TopP.Value, 0.001)
	assert.Equal(t, int64(2048), apiReq.MaxTokens)
}

func TestRequestMapper_SamplingValidation(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:   ModelClaudeSonnet46,
		Constraints: llm.ModelConstraints{TemperatureRange: [2]float64{0, 1}, MaxOutputTokens: 8192},
		MaxTokens:   2048,
	}

	mapper := NewRequestMapper(cfg)

	_, err := mapper.ToProvider(&llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi"))},
		Sampling: &llm.SamplingParams{Temperature: new(1.5)},
	})
	require.Error(t, err)
}
