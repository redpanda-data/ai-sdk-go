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

package anthropic_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
)

// TestMaxTokens_PerRequestOverride_Integration proves end to end, against the
// real Anthropic API, that the per-request RequestOptions.MaxTokens override
// actually reaches the provider and constrains generation — not just that it is
// mapped onto the request struct.
func TestMaxTokens_PerRequestOverride_Integration(t *testing.T) {
	t.Parallel()

	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	provider, err := anthropic.NewProvider(apiKey)
	require.NoError(t, err)

	// Build the model with a generous per-model budget so the ONLY thing that can
	// cap the tiny-budget case below is the per-request override.
	model, err := provider.NewModel(anthropictest.TestModelName, anthropic.WithMaxTokens(4096))
	require.NoError(t, err)

	longPrompt := "Write a detailed 1000-word essay about the history of computing."

	t.Run("tiny per-request budget truncates the turn", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		budget := 16
		req := &llm.Request{
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart(longPrompt))},
			Options:  &anthropic.RequestOptions{MaxTokens: &budget},
		}

		resp, err := model.Generate(ctx, req)
		require.NoError(t, err)
		require.NotNil(t, resp)

		// The turn hit the per-request cap: truncated, not a natural stop.
		assert.Equal(t, llm.FinishReasonLength, resp.FinishReason,
			"a 16-token cap on a long-essay prompt must truncate")

		require.NotNil(t, resp.Usage)
		assert.LessOrEqual(t, resp.Usage.OutputTokens, 16,
			"output tokens must not exceed the per-request cap; got %d", resp.Usage.OutputTokens)
		assert.Positive(t, resp.Usage.OutputTokens, "the model should still produce some output")

		t.Logf("tiny-cap: finish=%s output_tokens=%d", resp.FinishReason, resp.Usage.OutputTokens)
	})

	t.Run("generous per-request budget completes normally", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		budget := 1024
		req := &llm.Request{
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Reply with exactly: hello"))},
			Options:  &anthropic.RequestOptions{MaxTokens: &budget},
		}

		resp, err := model.Generate(ctx, req)
		require.NoError(t, err)
		require.NotNil(t, resp)

		assert.Equal(t, llm.FinishReasonStop, resp.FinishReason,
			"a trivial prompt under a 1024-token cap must complete naturally")

		require.NotNil(t, resp.Usage)
		assert.Less(t, resp.Usage.OutputTokens, 1024,
			"a one-word reply must stay well under the cap")

		t.Logf("generous-cap: finish=%s output_tokens=%d", resp.FinishReason, resp.Usage.OutputTokens)
	})
}
