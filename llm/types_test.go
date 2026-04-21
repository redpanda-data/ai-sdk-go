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
				InputTokens:                   100,
				CachedInputTokens:             50,
				CacheCreation5mTokens:         30,
				CacheCreation1hTokens:         20,
				CacheCreationUnknownTTLTokens: 5,
				ToolUseInputTokens:            10,
			},
			want: 215,
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
		InputTokens:                   100,
		CachedInputTokens:             50,
		CacheCreation5mTokens:         30,
		CacheCreationUnknownTTLTokens: 10,
		OutputTokens:                  40,
		ReasoningTokens:               10,
	}
	assert.Equal(t, 240, u.TotalBilledTokens())
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
					Extra:                 map[string]any{"provider.flag": true},
				},
			},
			expected: &TokenUsage{
				InputTokens:           100,
				CachedInputTokens:     10,
				CacheCreation5mTokens: 5,
				OutputTokens:          50,
				ReasoningTokens:       5,
				Extra:                 map[string]any{"provider.flag": true},
			},
		},
		{
			name: "scalars add, extra merges",
			usages: []*TokenUsage{
				{
					InputTokens:       100,
					CachedInputTokens: 10,
					OutputTokens:      50,
					ReasoningTokens:   5,
					Extra:             map[string]any{"bedrock.cache_write_ttl_1d_tokens": 8},
				},
				{
					InputTokens:                   200,
					CacheCreation5mTokens:         25,
					CacheCreationUnknownTTLTokens: 3,
					OutputTokens:                  100,
					ReasoningTokens:               10,
					Extra:                         map[string]any{"bedrock.cache_write_ttl_1d_tokens": 7},
				},
			},
			expected: &TokenUsage{
				InputTokens:                   300,
				CachedInputTokens:             10,
				CacheCreation5mTokens:         25,
				CacheCreationUnknownTTLTokens: 3,
				OutputTokens:                  150,
				ReasoningTokens:               15,
				Extra:                         map[string]any{"bedrock.cache_write_ttl_1d_tokens": 15},
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
		{
			// Same-typed numerics add across int, int64, and float64.
			// A differently-typed collision keeps the first writer —
			// e.g. int vs int64 for the same key would not coerce. A
			// type-consistent string collision also keeps the first.
			name: "extra merges int, int64, and float64 by type",
			usages: []*TokenUsage{
				{
					Extra: map[string]any{
						"provider.int":    10,
						"provider.int64":  int64(1000),
						"provider.float":  1.5,
						"provider.string": "a",
					},
				},
				{
					Extra: map[string]any{
						"provider.int":    25,
						"provider.int64":  int64(2500),
						"provider.float":  2.25,
						"provider.string": "b",
					},
				},
			},
			expected: &TokenUsage{
				Extra: map[string]any{
					"provider.int":    35,
					"provider.int64":  int64(3500),
					"provider.float":  3.75,
					"provider.string": "a",
				},
			},
		},
		{
			// Cross-type numeric collisions are first-writer-wins — we
			// don't silently coerce int -> int64 or int -> float64.
			name: "extra skips cross-type numeric collisions",
			usages: []*TokenUsage{
				{Extra: map[string]any{"provider.count": 10}},
				{Extra: map[string]any{"provider.count": int64(99)}},
			},
			expected: &TokenUsage{
				Extra: map[string]any{"provider.count": 10},
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

	original := map[string]any{"provider.x": 100}
	a := &TokenUsage{InputTokens: 10, Extra: original}
	b := &TokenUsage{InputTokens: 5, Extra: map[string]any{"provider.x": 50}}

	result := SumUsage(a, b)

	assert.Equal(t, 150, result.Extra["provider.x"])
	assert.Equal(t, 100, original["provider.x"], "SumUsage must not mutate the first input's Extra map")
	assert.NotSame(t, &original, &result.Extra, "result must not alias the first input's Extra map")
}
