package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

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
			name: "single usage",
			usages: []*TokenUsage{
				{
					InputTokens:     100,
					OutputTokens:    50,
					TotalTokens:     150,
					CachedTokens:    10,
					ReasoningTokens: 5,
				},
			},
			expected: &TokenUsage{
				InputTokens:     100,
				OutputTokens:    50,
				TotalTokens:     150,
				CachedTokens:    10,
				ReasoningTokens: 5,
			},
		},
		{
			name: "multiple usages",
			usages: []*TokenUsage{
				{
					InputTokens:     100,
					OutputTokens:    50,
					TotalTokens:     150,
					CachedTokens:    10,
					ReasoningTokens: 5,
				},
				{
					InputTokens:     200,
					OutputTokens:    100,
					TotalTokens:     300,
					CachedTokens:    20,
					ReasoningTokens: 10,
				},
				{
					InputTokens:     50,
					OutputTokens:    25,
					TotalTokens:     75,
					CachedTokens:    5,
					ReasoningTokens: 2,
				},
			},
			expected: &TokenUsage{
				InputTokens:     350,
				OutputTokens:    175,
				TotalTokens:     525,
				CachedTokens:    35,
				ReasoningTokens: 17,
			},
		},
		{
			name: "mix of nil and non-nil",
			usages: []*TokenUsage{
				nil,
				{
					InputTokens:  100,
					OutputTokens: 50,
					TotalTokens:  150,
				},
				nil,
				{
					InputTokens:  200,
					OutputTokens: 100,
					TotalTokens:  300,
				},
				nil,
			},
			expected: &TokenUsage{
				InputTokens:  300,
				OutputTokens: 150,
				TotalTokens:  450,
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
