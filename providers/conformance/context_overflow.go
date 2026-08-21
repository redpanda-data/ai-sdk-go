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

package conformance

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestContextOverflow verifies that a request exceeding the model's context
// window is rejected with llm.ErrContextOverflow on both the Generate and
// GenerateEvents paths, and that the error is not retryable.
func (s *Suite) TestContextOverflow(t *testing.T) {
	t.Helper()
	testContextOverflow(t, s.fixture)
}

// overflowFiller is ~15-19 tokens per repetition across provider tokenizers.
const overflowFiller = "The quick brown fox jumps over the lazy dog near a riverbank at dawn. "

// oversizedPrompt returns text at least 1.5x the given window. Oversized
// requests are rejected at validation, before any tokens are billed.
func oversizedPrompt(windowTokens int) string {
	return strings.Repeat(overflowFiller, windowTokens/10)
}

// overflowRequest builds a single-message request that exceeds the window.
func overflowRequest(windowTokens int) *llm.Request {
	return &llm.Request{
		Messages: []llm.Message{
			{
				Role:    llm.RoleUser,
				Content: []llm.Part{llm.NewTextPart(oversizedPrompt(windowTokens))},
			},
		},
	}
}

// requireContextOverflow asserts err is a non-retryable llm.ErrContextOverflow.
func requireContextOverflow(t *testing.T, err error) {
	t.Helper()

	require.Error(t, err, "an over-window request must be rejected")
	require.ErrorIs(t, err, llm.ErrContextOverflow,
		"overflow must be classified as llm.ErrContextOverflow, got: %v", err)
	assert.False(t, llm.IsRetryable(err), "context overflow is not retryable as-is")
}

func testContextOverflow(t *testing.T, fixture Fixture) { //nolint:thelper // not a helper, called from t.Run subtest
	model := fixture.NewStandardModel(t)
	if model == nil {
		t.Skip("No standard model available")
	}

	window := model.Constraints().MaxInputTokens
	if window <= 0 {
		t.Skip("Model does not declare a context window")
	}

	request := overflowRequest(window)

	t.Run("generate rejects over-window input", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
		defer cancel()

		_, err := model.Generate(ctx, request)
		requireContextOverflow(t, err)
	})

	t.Run("streaming rejects over-window input", func(t *testing.T) {
		if !model.Capabilities().Streaming {
			t.Skip("Model does not support streaming")
		}

		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
		defer cancel()

		var streamErr error

		for event, err := range model.GenerateEvents(ctx, request) {
			if err != nil {
				streamErr = err
				break
			}

			if end, ok := event.(llm.StreamEndEvent); ok && end.Error != nil {
				streamErr = end.Error
				break
			}

			if errEvent, ok := event.(llm.ErrorEvent); ok {
				streamErr = errors.New(errEvent.Message)
				break
			}
		}

		requireContextOverflow(t, streamErr)
	})
}
