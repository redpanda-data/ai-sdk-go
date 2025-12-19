package openai

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestGPT52ReasoningEffortIntegration tests that GPT-5.2 rejects 'minimal' reasoning effort
// and accepts 'none' when making actual API calls.
//
// This test requires OPENAI_API_KEY environment variable to be set.
// Run with: OPENAI_API_KEY=sk-... go test -v -run TestGPT52ReasoningEffortIntegration
func TestGPT52ReasoningEffortIntegration(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping integration test: OPENAI_API_KEY not set")
	}

	provider, err := NewProvider(apiKey)
	require.NoError(t, err)

	ctx := context.Background()

	testCases := []struct {
		name          string
		model         string
		reasoningOpt  Option
		wantErr       bool
		errContains   string
	}{
		{
			name:         "gpt-5.2 with ReasoningEffortNone should succeed",
			model:        ModelGPT5_2,
			reasoningOpt: WithReasoningEffort(ReasoningEffortNone),
			wantErr:      false,
		},
		{
			name:         "gpt-5.2 with ReasoningEffortMinimal should fail",
			model:        ModelGPT5_2,
			reasoningOpt: WithReasoningEffort(ReasoningEffortMinimal),
			wantErr:      true,
			errContains:  "minimal",
		},
		{
			name:         "gpt-5.1 with ReasoningEffortNone should succeed",
			model:        ModelGPT5_1,
			reasoningOpt: WithReasoningEffort(ReasoningEffortNone),
			wantErr:      false,
		},
		{
			name:         "gpt-5.1 with ReasoningEffortMinimal should fail",
			model:        ModelGPT5_1,
			reasoningOpt: WithReasoningEffort(ReasoningEffortMinimal),
			wantErr:      true,
			errContains:  "minimal",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create model with reasoning effort
			model, err := provider.NewModel(tc.model, tc.reasoningOpt)
			require.NoError(t, err, "Model creation should succeed")

			// Verify supported reasoning efforts
			supportedEfforts := model.(*Model).SupportedReasoningEfforts()
			t.Logf("Model %s supported reasoning efforts: %v", tc.model, supportedEfforts)
			assert.NotEmpty(t, supportedEfforts, "Should have supported reasoning efforts")

			// Make a simple API call
			req := &llm.Request{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []*llm.Part{
							llm.NewTextPart("What is 2+2? Answer with just the number."),
						},
					},
				},
			}

			resp, err := model.Generate(ctx, req)

			if tc.wantErr {
				require.Error(t, err, "Expected API call to fail")
				if tc.errContains != "" {
					assert.Contains(t, err.Error(), tc.errContains, "Error should mention the unsupported value")
				}
				t.Logf("Expected error occurred: %v", err)
			} else {
				require.NoError(t, err, "Expected API call to succeed")
				require.NotNil(t, resp, "Response should not be nil")
				assert.NotEmpty(t, resp.Message.Content, "Response should have content")
				if len(resp.Message.Content) > 0 && resp.Message.Content[0].Text != "" {
					t.Logf("Success! Response: %s", resp.Message.Content[0].Text)
				}
			}
		})
	}
}

// TestSupportedReasoningEfforts verifies that models report correct supported reasoning efforts.
func TestSupportedReasoningEfforts(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	testCases := []struct {
		name           string
		model          string
		expectedFirst  ReasoningEffort // First (safest) effort
		expectNone     bool            // Should support 'none'
		expectMinimal  bool            // Should support 'minimal'
	}{
		{
			name:          "gpt-5.2 should start with 'none'",
			model:         ModelGPT5_2,
			expectedFirst: ReasoningEffortNone,
			expectNone:    true,
			expectMinimal: false,
		},
		{
			name:          "gpt-5.1 should start with 'none'",
			model:         ModelGPT5_1,
			expectedFirst: ReasoningEffortNone,
			expectNone:    true,
			expectMinimal: false,
		},
		{
			name:          "gpt-5 should start with 'minimal'",
			model:         ModelGPT5,
			expectedFirst: ReasoningEffortMinimal,
			expectNone:    false,
			expectMinimal: true,
		},
		{
			name:          "o3 should start with 'minimal'",
			model:         ModelO3,
			expectedFirst: ReasoningEffortMinimal,
			expectNone:    false,
			expectMinimal: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			model, err := provider.NewModel(tc.model)
			require.NoError(t, err)

			m := model.(*Model)
			efforts := m.SupportedReasoningEfforts()

			require.NotEmpty(t, efforts, "Model should have supported reasoning efforts")
			assert.Equal(t, tc.expectedFirst, efforts[0], "First effort should be the safest/lowest")

			// Check for 'none' support
			hasNone := false
			hasMinimal := false
			for _, effort := range efforts {
				if effort == ReasoningEffortNone {
					hasNone = true
				}
				if effort == ReasoningEffortMinimal {
					hasMinimal = true
				}
			}

			assert.Equal(t, tc.expectNone, hasNone, "'none' support mismatch")
			assert.Equal(t, tc.expectMinimal, hasMinimal, "'minimal' support mismatch")

			t.Logf("Model %s supported efforts: %v", tc.model, efforts)
		})
	}
}
