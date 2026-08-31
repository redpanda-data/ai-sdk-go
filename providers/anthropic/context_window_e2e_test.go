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

package anthropic

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestClassifyError_ContextWindowStreaming checks the refinement survives the
// SSE path. llmagent prefers GenerateEvents, so a refinement wired only into
// the HTTP branch would never fire in practice.
func TestClassifyError_ContextWindowStreaming(t *testing.T) {
	t.Parallel()

	sseError := func(errType, message string) error {
		return fmt.Errorf(
			"received error while streaming: {\"type\":\"error\",\"error\":"+
				"{\"type\":%q,\"message\":%q}}", errType, message)
	}

	tests := []struct {
		name     string
		err      error
		overflow bool
	}{
		{
			name: "prompt too long",
			err: sseError("invalid_request_error",
				"prompt is too long: 243619 tokens > 200000 maximum"),
			overflow: true,
		},
		{
			name: "max_tokens pushes past the window",
			err: sseError("invalid_request_error",
				"input length and `max_tokens` exceed context limit: 190000 + 16384 > 200000"),
			overflow: true,
		},
		{
			name: "other bad request",
			err:  sseError("invalid_request_error", "messages: at least one message is required"),
		},
		{
			name: "rate limit mentioning tokens",
			err:  sseError("rate_limit_error", "input tokens per minute limit exceeded"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := classifyError(tt.err)

			var provErr *llm.ProviderError
			require.ErrorAs(t, err, &provErr)

			if !tt.overflow {
				assert.NotErrorIs(t, err, llm.ErrContextOverflow)

				return
			}

			require.ErrorIs(t, err, llm.ErrContextOverflow)
			require.ErrorIs(t, err, llm.ErrInvalidInput,
				"the refinement keeps matching the broader category")
			assert.False(t, llm.IsRetryable(err),
				"an oversized prompt will not fix itself on retry")
		})
	}
}
