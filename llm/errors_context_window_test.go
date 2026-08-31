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

package llm_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestErrContextOverflow covers both origins of an overflow: a provider
// rejection, and one raised locally before any request is sent.
func TestErrContextOverflow(t *testing.T) {
	t.Parallel()

	overflow := &llm.ProviderError{
		Base:    llm.ErrContextOverflow,
		Code:    "400",
		Message: "prompt is too long: 243619 tokens > 200000 maximum",
	}

	otherBadRequest := &llm.ProviderError{
		Base:    llm.ErrInvalidInput,
		Code:    "400",
		Message: "invalid image format",
	}

	tests := []struct {
		name     string
		err      error
		overflow bool
	}{
		{name: "provider rejection", err: overflow, overflow: true},
		{
			name:     "wrapped by the agent",
			err:      fmt.Errorf("agent: model generation failed: %w", overflow),
			overflow: true,
		},
		{
			// A locally detected overflow has no ProviderError to fabricate.
			name:     "raised locally",
			err:      fmt.Errorf("request exceeds the context budget: %w", llm.ErrContextOverflow),
			overflow: true,
		},
		{name: "other bad request", err: otherBadRequest},
		{
			name: "rate limit",
			err: &llm.ProviderError{
				Base: llm.ErrRateLimitExceeded, Message: "too many tokens per minute",
			},
		},
		{name: "unrelated error", err: errors.New("boom")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tt.overflow, errors.Is(tt.err, llm.ErrContextOverflow))

			// Every overflow is also an invalid input, so callers matching the
			// broader category keep working.
			if tt.overflow {
				assert.ErrorIs(t, tt.err, llm.ErrInvalidInput)
			}
		})
	}
}

// TestProviderError_BaseIsMatchedNotCompared pins that Base may hold a
// refinement, so it must be matched with errors.Is rather than compared.
func TestProviderError_BaseIsMatchedNotCompared(t *testing.T) {
	t.Parallel()

	err := &llm.ProviderError{Base: llm.ErrContextOverflow}

	require.ErrorIs(t, err, llm.ErrContextOverflow)
	require.ErrorIs(t, err, llm.ErrInvalidInput)
	assert.NotEqual(t, llm.ErrInvalidInput, err.Base,
		"comparing Base would miss the refinement")
}

// TestErrContextOverflow_NotRetryable guards against an overflow being
// retried unchanged, which can only fail again.
func TestErrContextOverflow_NotRetryable(t *testing.T) {
	t.Parallel()

	assert.False(t, llm.IsRetryable(llm.ErrContextOverflow))
	assert.False(t, llm.IsRetryable(&llm.ProviderError{Base: llm.ErrContextOverflow}))
}
