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

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestIsHelpers(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		fn   func(error) bool
		base error
	}{
		{"IsRateLimit", llm.IsRateLimit, llm.ErrRateLimitExceeded},
		{"IsInvalidInput", llm.IsInvalidInput, llm.ErrInvalidInput},
		{"IsContentPolicy", llm.IsContentPolicy, llm.ErrContentPolicyViolation},
		{"IsServerError", llm.IsServerError, llm.ErrServerError},
		{"IsUnsupportedFeature", llm.IsUnsupportedFeature, llm.ErrUnsupportedFeature},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Sentinel itself matches.
			assert.True(t, tc.fn(tc.base))

			// Wrapped error matches via errors.Is chain.
			wrapped := fmt.Errorf("context: %w", tc.base)
			assert.True(t, tc.fn(wrapped))

			// *ProviderError with matching Base matches.
			perr := &llm.ProviderError{Base: tc.base, Code: "x", Message: "y"}
			assert.True(t, tc.fn(perr))

			// Unrelated sentinel does not match.
			assert.False(t, tc.fn(errors.New("plain")))

			// nil is false.
			assert.False(t, tc.fn(nil))
		})
	}
}

func TestProviderError_ErrorsAs(t *testing.T) {
	t.Parallel()

	base := &llm.ProviderError{
		Base:    llm.ErrRateLimitExceeded,
		Code:    "rate_limit_exceeded",
		Message: "throttled",
	}
	wrapped := fmt.Errorf("call failed: %w", base)

	var perr *llm.ProviderError
	require := assert.New(t)
	require.ErrorAs(wrapped, &perr)
	require.Equal("rate_limit_exceeded", perr.Code)
	require.Equal("throttled", perr.Message)
}
