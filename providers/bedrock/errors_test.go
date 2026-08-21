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

package bedrock

import (
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Messages recorded from the live Converse API (2026-08-21) plus documented
// per-vendor variants.
func TestClassifyValidationException(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		message  string
		wantBase error
	}{
		{
			name:     "anthropic prompt too long",
			message:  "The model returned the following errors: prompt is too long: 206456 tokens > 200000 maximum",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "nova input tokens exceeded",
			message:  "The model returned the following errors: Input Tokens Exceeded: Number of input tokens exceeds maximum length. Please update the input to try again.",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "llama maximum context length",
			message:  "The model returned the following errors: This model's maximum context length is 131072 tokens. Please reduce the length of the prompt",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "bedrock normalized input too long",
			message:  "Input is too long for requested model.",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "pre-4.5 anthropic input plus max_tokens",
			message:  "input length and `max_tokens` exceed context limit: 213462 + 8192 > 204698",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "titan too many input tokens",
			message:  "Too many input tokens. Max input tokens: 8192, request input token count: 9000",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "anthropic byte cap",
			message:  "The model returned the following errors: too many total text bytes",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "titan malformed input maxLength",
			message:  "Malformed input request: expected maxLength: 42000, actual: 150180, please reformat your input and try again.",
			wantBase: llm.ErrContextOverflow,
		},
		{
			name:     "max tokens over output cap stays invalid input",
			message:  "The maximum tokens you requested exceeds the model limit of 64000. Try again with a maximum tokens value that is lower than 64000.",
			wantBase: llm.ErrInvalidInput,
		},
		{
			name:     "unrelated validation error stays invalid input",
			message:  "1 validation error detected: Value '99.0' at 'inferenceConfig.temperature' failed to satisfy constraint: Member must have value less than or equal to 1",
			wantBase: llm.ErrInvalidInput,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := classifyError(&types.ValidationException{Message: aws.String(tt.message)})
			require.Error(t, err)

			var pe *llm.ProviderError
			require.ErrorAs(t, err, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.False(t, pe.Retryable)
			assert.Equal(t, "ValidationException", pe.Code)
			assert.Equal(t, tt.message, pe.Message)

			if errors.Is(tt.wantBase, llm.ErrInvalidInput) && !errors.Is(tt.wantBase, llm.ErrContextOverflow) {
				assert.NotErrorIs(t, pe, llm.ErrContextOverflow)
			}
		})
	}
}

func TestClassifyError_ThrottlingNotOverflow(t *testing.T) {
	t.Parallel()

	// The throttling message mentions tokens but is not an overflow.
	err := classifyError(&types.ThrottlingException{
		Message: aws.String("Too many tokens, please wait before trying again."),
	})

	var pe *llm.ProviderError
	require.ErrorAs(t, err, &pe)
	require.ErrorIs(t, pe, llm.ErrRateLimitExceeded)
	assert.True(t, pe.Retryable)
}
