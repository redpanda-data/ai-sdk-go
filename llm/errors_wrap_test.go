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
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWrapAPICall(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		err     error
		wantMsg string
	}{
		{
			name:    "already an API-call failure is not wrapped again",
			err:     &ProviderError{Base: ErrAPICall, Code: "guardrail_intervened", Message: "blocked"},
			wantMsg: "API call failed: [guardrail_intervened] blocked",
		},
		{
			name:    "other classifications gain the category prefix once",
			err:     &ProviderError{Base: ErrRateLimitExceeded, Code: "rate_limit_exceeded", Message: "slow down"},
			wantMsg: "API call failed: rate limit exceeded: [rate_limit_exceeded] slow down",
		},
		{
			name:    "plain error gains the prefix",
			err:     errors.New("boom"),
			wantMsg: "API call failed: boom",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := WrapAPICall(tc.err)
			require.Error(t, got)
			assert.Equal(t, tc.wantMsg, got.Error())
			require.ErrorIs(t, got, ErrAPICall, "the category must stay matchable")

			var perr *ProviderError
			if errors.As(tc.err, &perr) {
				require.ErrorIs(t, got, perr.Base, "the specific classification must stay matchable")
			}
		})
	}

	assert.NoError(t, WrapAPICall(nil))
}
