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
// Run with: OPENAI_API_KEY=sk-... go test -v -run TestGPT52ReasoningEffortIntegration.
func TestGPT52ReasoningEffortIntegration(t *testing.T) {
	t.Parallel()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping integration test: OPENAI_API_KEY not set")
	}

	provider, err := NewProvider(apiKey)
	require.NoError(t, err)

	ctx := context.Background()

	testCases := []struct {
		name         string
		model        string
		reasoningOpt Option
		wantErr      bool
		errContains  string
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
			t.Parallel()

			// Create model with reasoning effort
			model, err := provider.NewModel(tc.model, tc.reasoningOpt)

			if tc.wantErr {
				// SDK should catch unsupported reasoning efforts early
				require.Error(t, err, "Model creation should fail for unsupported reasoning effort")

				if tc.errContains != "" {
					assert.Contains(t, err.Error(), tc.errContains, "Error should mention the unsupported value")
				}

				t.Logf("Expected SDK validation error occurred: %v", err)

				return
			}

			require.NoError(t, err, "Model creation should succeed")

			// Verify supported reasoning efforts
			m, ok := model.(*Model)
			require.True(t, ok, "Model should be *openai.Model")

			supportedEfforts := m.SupportedReasoningEfforts()
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
			require.NoError(t, err, "Expected API call to succeed")
			require.NotNil(t, resp, "Response should not be nil")
			assert.NotEmpty(t, resp.Message.Content, "Response should have content")

			if len(resp.Message.Content) > 0 && resp.Message.Content[0].Text != "" {
				t.Logf("Success! Response: %s", resp.Message.Content[0].Text)
			}
		})
	}
}

// TestAllModelsReasoningEffortsIntegration tests all reasoning models with all their supported reasoning efforts.
// This makes actual API calls to validate that each model accepts its advertised reasoning efforts.
//
// This test requires OPENAI_API_KEY environment variable to be set.
// Run with: OPENAI_API_KEY=sk-... go test -v -run TestAllModelsReasoningEffortsIntegration -timeout 30m.
func TestAllModelsReasoningEffortsIntegration(t *testing.T) {
	t.Parallel()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping integration test: OPENAI_API_KEY not set")
	}

	provider, err := NewProvider(apiKey)
	require.NoError(t, err)

	ctx := context.Background()

	// Get all reasoning models from the provider
	reasoningModels := []string{
		ModelGPT5,
		ModelGPT5Mini,
		ModelGPT5_1,
		ModelGPT5_2,
		ModelGPT5_2Instant,
		ModelGPT5_2Pro,
		ModelO3,
		ModelO4Mini,
		ModelO1Pro,
		ModelO3Pro,
	}

	for _, modelName := range reasoningModels {
		t.Run(modelName, func(t *testing.T) {
			t.Parallel()

			// Create model to get supported reasoning efforts
			model, err := provider.NewModel(modelName)
			require.NoError(t, err)

			m, ok := model.(*Model)
			require.True(t, ok, "Model should be *openai.Model")

			supportedEfforts := m.SupportedReasoningEfforts()
			require.NotEmpty(t, supportedEfforts, "Model should have supported reasoning efforts")

			t.Logf("Testing model %s with %d reasoning efforts: %v", modelName, len(supportedEfforts), supportedEfforts)

			// Test each supported reasoning effort
			for _, effort := range supportedEfforts {
				t.Run(string(effort), func(t *testing.T) {
					t.Parallel()

					// Create model with specific reasoning effort
					testModel, err := provider.NewModel(modelName, WithReasoningEffort(effort))
					require.NoError(t, err, "Should create model with effort %s", effort)

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

					resp, err := testModel.Generate(ctx, req)
					require.NoError(t, err, "API call should succeed with effort %s", effort)
					require.NotNil(t, resp, "Response should not be nil")
					assert.NotEmpty(t, resp.Message.Content, "Response should have content")

					if len(resp.Message.Content) > 0 && resp.Message.Content[0].Text != "" {
						t.Logf("✓ Model %s with effort %s succeeded. Response: %s",
							modelName, effort, resp.Message.Content[0].Text)
					}
				})
			}
		})
	}
}

// TestUnsupportedReasoningEffortsIntegration validates that models reject unsupported reasoning efforts.
// This makes actual API calls to verify the model definitions are accurate.
//
// This test requires OPENAI_API_KEY environment variable to be set.
// Run with: OPENAI_API_KEY=sk-... go test -v -run TestUnsupportedReasoningEffortsIntegration.
func TestUnsupportedReasoningEffortsIntegration(t *testing.T) {
	t.Parallel()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping integration test: OPENAI_API_KEY not set")
	}

	provider, err := NewProvider(apiKey)
	require.NoError(t, err)

	testCases := []struct {
		model        string
		effort       ReasoningEffort
		shouldReject bool
		reason       string
	}{
		{
			model:        ModelGPT5_2,
			effort:       ReasoningEffortMinimal,
			shouldReject: true,
			reason:       "GPT-5.2 doesn't support 'minimal'",
		},
		{
			model:        ModelGPT5_1,
			effort:       ReasoningEffortMinimal,
			shouldReject: true,
			reason:       "GPT-5.1 doesn't support 'minimal'",
		},
		{
			model:        ModelGPT5,
			effort:       ReasoningEffortNone,
			shouldReject: true,
			reason:       "GPT-5 doesn't support 'none'",
		},
		{
			model:        ModelO3,
			effort:       ReasoningEffortNone,
			shouldReject: true,
			reason:       "O3 doesn't support 'none'",
		},
		{
			model:        ModelO3,
			effort:       ReasoningEffortMinimal,
			shouldReject: true,
			reason:       "O3 doesn't support 'minimal'",
		},
		{
			model:        ModelO1Pro,
			effort:       ReasoningEffortMinimal,
			shouldReject: true,
			reason:       "O1-pro doesn't support 'minimal'",
		},
		{
			model:        ModelO4Mini,
			effort:       ReasoningEffortMinimal,
			shouldReject: true,
			reason:       "O4-mini doesn't support 'minimal'",
		},
		{
			model:        ModelGPT5_2Pro,
			effort:       ReasoningEffortNone,
			shouldReject: true,
			reason:       "GPT-5.2-pro doesn't support 'none'",
		},
		{
			model:        ModelGPT5_2Pro,
			effort:       ReasoningEffortLow,
			shouldReject: true,
			reason:       "GPT-5.2-pro doesn't support 'low'",
		},
		{
			model:        ModelGPT5_2Instant,
			effort:       ReasoningEffortNone,
			shouldReject: true,
			reason:       "GPT-5.2-instant only supports 'medium'",
		},
		{
			model:        ModelGPT5_2Instant,
			effort:       ReasoningEffortLow,
			shouldReject: true,
			reason:       "GPT-5.2-instant only supports 'medium'",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.model+"_with_"+string(tc.effort), func(t *testing.T) {
			t.Parallel()

			// SDK now validates reasoning efforts at model creation time
			// Create model with unsupported reasoning effort - should fail at SDK level
			_, err := provider.NewModel(tc.model, WithReasoningEffort(tc.effort))

			if tc.shouldReject {
				require.Error(t, err, "SDK should reject unsupported reasoning effort: %s", tc.reason)
				assert.Contains(t, err.Error(), "does not support reasoning effort", "Error should mention unsupported reasoning effort")
				t.Logf("✓ SDK correctly rejected: %v", err)
			} else {
				require.NoError(t, err, "SDK should accept reasoning effort")
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
		name          string
		model         string
		expectedFirst ReasoningEffort // First (safest) effort
		expectNone    bool            // Should support 'none'
		expectMinimal bool            // Should support 'minimal'
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
			name:          "o3 should start with 'low'",
			model:         ModelO3,
			expectedFirst: ReasoningEffortLow,
			expectNone:    false,
			expectMinimal: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			model, err := provider.NewModel(tc.model)
			require.NoError(t, err)

			m, ok := model.(*Model)
			require.True(t, ok, "Model should be *openai.Model")

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
