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

package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTokenUsage_BilledInputTokens(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		usage *TokenUsage
		want  int
	}{
		{"nil", nil, 0},
		{"zero", &TokenUsage{}, 0},
		{
			name: "all input buckets disjoint and summed",
			usage: &TokenUsage{
				InputTokens:           100,
				CachedInputTokens:     50,
				CacheCreation5mTokens: 30,
				CacheCreation1hTokens: 20,
				ToolUseInputTokens:    10,
			},
			want: 210,
		},
		{
			name: "only fresh input",
			usage: &TokenUsage{
				InputTokens: 75,
			},
			want: 75,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, tt.usage.BilledInputTokens())
		})
	}
}

func TestTokenUsage_BilledOutputTokens(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		usage *TokenUsage
		want  int
	}{
		{"nil", nil, 0},
		{"zero", &TokenUsage{}, 0},
		{
			name: "output + reasoning disjoint",
			usage: &TokenUsage{
				OutputTokens:    50,
				ReasoningTokens: 25,
			},
			want: 75,
		},
		{
			name: "with predicted outputs",
			usage: &TokenUsage{
				OutputTokens:             40,
				ReasoningTokens:          0,
				AcceptedPredictionTokens: 10,
				RejectedPredictionTokens: 5,
			},
			want: 55,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, tt.usage.BilledOutputTokens())
		})
	}
}

func TestTokenUsage_TotalBilledTokens(t *testing.T) {
	t.Parallel()

	u := &TokenUsage{
		InputTokens:           100,
		CachedInputTokens:     50,
		CacheCreation5mTokens: 30,
		OutputTokens:          40,
		ReasoningTokens:       10,
	}
	assert.Equal(t, 230, u.TotalBilledTokens())
}

func TestSumUsage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		usages   []*TokenUsage
		expected *TokenUsage
	}{
		{
			name:     "nil inputs",
			usages:   []*TokenUsage{nil, nil},
			expected: nil,
		},
		{
			name:     "empty inputs",
			usages:   []*TokenUsage{},
			expected: nil,
		},
		{
			name: "single usage is deep-copied",
			usages: []*TokenUsage{
				{
					InputTokens:           100,
					CachedInputTokens:     10,
					CacheCreation5mTokens: 5,
					OutputTokens:          50,
					ReasoningTokens:       5,
					ModalityInputTokens:   map[Modality]int{ModalityText: 100},
					ServerToolRequests:    map[ServerTool]int{ServerToolWebSearch: 2},
					Extra:                 map[string]any{"provider.flag": true},
				},
			},
			expected: &TokenUsage{
				InputTokens:           100,
				CachedInputTokens:     10,
				CacheCreation5mTokens: 5,
				OutputTokens:          50,
				ReasoningTokens:       5,
				ModalityInputTokens:   map[Modality]int{ModalityText: 100},
				ServerToolRequests:    map[ServerTool]int{ServerToolWebSearch: 2},
				Extra:                 map[string]any{"provider.flag": true},
			},
		},
		{
			name: "scalars add, maps merge",
			usages: []*TokenUsage{
				{
					InputTokens:         100,
					CachedInputTokens:   10,
					OutputTokens:        50,
					ReasoningTokens:     5,
					ModalityInputTokens: map[Modality]int{ModalityText: 80, ModalityImage: 20},
					ServerToolRequests:  map[ServerTool]int{ServerToolWebSearch: 2},
					GuardrailUnits:      map[string]int{"content_policy": 1},
				},
				{
					InputTokens:           200,
					CacheCreation5mTokens: 25,
					OutputTokens:          100,
					ReasoningTokens:       10,
					ModalityInputTokens:   map[Modality]int{ModalityText: 150, ModalityAudio: 50},
					ServerToolRequests:    map[ServerTool]int{ServerToolWebSearch: 1, ServerToolWebFetch: 3},
					GuardrailUnits:        map[string]int{"content_policy": 2, "topic_policy": 1},
				},
			},
			expected: &TokenUsage{
				InputTokens:           300,
				CachedInputTokens:     10,
				CacheCreation5mTokens: 25,
				OutputTokens:          150,
				ReasoningTokens:       15,
				ModalityInputTokens: map[Modality]int{
					ModalityText:  230,
					ModalityImage: 20,
					ModalityAudio: 50,
				},
				ServerToolRequests: map[ServerTool]int{
					ServerToolWebSearch: 3,
					ServerToolWebFetch:  3,
				},
				GuardrailUnits: map[string]int{"content_policy": 3, "topic_policy": 1},
			},
		},
		{
			name: "first non-empty string wins",
			usages: []*TokenUsage{
				{InputTokens: 10, ServiceTier: ServiceTierPriority, Speed: "fast"},
				{InputTokens: 20, ServiceTier: ServiceTierDefault, Speed: "standard"},
			},
			expected: &TokenUsage{
				InputTokens: 30,
				ServiceTier: ServiceTierPriority,
				Speed:       "fast",
			},
		},
		{
			name: "MaxInputTokens takes max",
			usages: []*TokenUsage{
				{InputTokens: 1, MaxInputTokens: 128_000},
				{InputTokens: 1, MaxInputTokens: 200_000},
			},
			expected: &TokenUsage{
				InputTokens:    2,
				MaxInputTokens: 200_000,
			},
		},
		{
			name: "mix of nil and non-nil",
			usages: []*TokenUsage{
				nil,
				{InputTokens: 100, OutputTokens: 50},
				nil,
				{InputTokens: 200, OutputTokens: 100},
				nil,
			},
			expected: &TokenUsage{
				InputTokens:  300,
				OutputTokens: 150,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := SumUsage(tt.usages...)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestSumUsage_NoAliasing verifies that SumUsage does not mutate input maps
// or cause aliasing between inputs and the result.
func TestSumUsage_NoAliasing(t *testing.T) {
	t.Parallel()

	original := map[Modality]int{ModalityText: 100}
	a := &TokenUsage{InputTokens: 10, ModalityInputTokens: original}
	b := &TokenUsage{InputTokens: 5, ModalityInputTokens: map[Modality]int{ModalityText: 50}}

	result := SumUsage(a, b)

	assert.Equal(t, 150, result.ModalityInputTokens[ModalityText])
	assert.Equal(t, 100, original[ModalityText], "SumUsage must not mutate the first input's map")
	assert.NotSame(t, &original, &result.ModalityInputTokens, "result must not alias the first input's map")
}
