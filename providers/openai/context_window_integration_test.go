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

package openai_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
)

// TestContextWindow_LiveRejection_Integration sends an oversized prompt to the
// real API and asserts the rejection maps to llm.ErrContextOverflow.
//
// This is what keeps the matcher's phrase list honest: the live wording is
// asserted rather than assumed, and the log records it for when it changes.
// An over-window request is rejected before the model runs, so no tokens are
// billed.
func TestContextWindow_LiveRejection_Integration(t *testing.T) {
	t.Parallel()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName)
	require.NoError(t, err)

	// 1.5x the window at ~4 chars per token guarantees an overflow.
	window := model.Constraints().MaxInputTokens
	require.Positive(t, window)

	oversized := strings.Repeat("filler words for the window ", window*6/29+1)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	_, err = model.Generate(ctx, &llm.Request{Messages: []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart(oversized)),
	}})

	require.Error(t, err)
	t.Logf("live rejection: %v", err)

	require.ErrorIs(t, err, llm.ErrContextOverflow,
		"the live wording no longer matches isContextWindowRejection")
	require.ErrorIs(t, err, llm.ErrInvalidInput)
	assert.False(t, llm.IsRetryable(err))
}
