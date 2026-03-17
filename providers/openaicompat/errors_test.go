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

package openaicompat

import (
	"errors"
	"net/http"
	"net/url"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestClassifyError_Nil(t *testing.T) {
	t.Parallel()

	assert.NoError(t, classifyError(nil))
}

func TestClassifyError_UnknownError(t *testing.T) {
	t.Parallel()

	err := errors.New("something unexpected")
	result := classifyError(err)
	assert.Equal(t, err, result)
}

func TestClassifyHTTPError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		wantBase   error
		wantRetry  bool
	}{
		{"rate limit", 429, llm.ErrRateLimitExceeded, true},
		{"server error 500", 500, llm.ErrServerError, true},
		{"bad gateway 502", 502, llm.ErrServerError, true},
		{"service unavailable 503", 503, llm.ErrServerError, true},
		{"bad request 400", 400, llm.ErrInvalidInput, false},
		{"unauthorized 401", 401, llm.ErrAPICall, false},
		{"forbidden 403", 403, llm.ErrAPICall, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			reqURL, _ := url.Parse("https://api.openai.com/v1/chat/completions")
			apiErr := &openai.Error{
				StatusCode: tt.statusCode,
				Request:    &http.Request{Method: http.MethodPost, URL: reqURL},
				Response:   &http.Response{StatusCode: tt.statusCode},
			}

			result := classifyError(apiErr)
			require.Error(t, result)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.Equal(t, tt.wantRetry, pe.Retryable)
		})
	}
}

func TestClassifyHTTPError_ProviderErrorFields(t *testing.T) {
	t.Parallel()

	reqURL, _ := url.Parse("https://api.openai.com/v1/chat/completions")
	apiErr := &openai.Error{
		StatusCode: http.StatusTooManyRequests,
		Code:       "rate_limit_exceeded",
		Message:    "Rate limit exceeded",
		Request:    &http.Request{Method: http.MethodPost, URL: reqURL},
		Response:   &http.Response{StatusCode: http.StatusTooManyRequests},
	}

	result := classifyError(apiErr)

	var pe *llm.ProviderError
	require.ErrorAs(t, result, &pe)
	assert.Equal(t, "rate_limit_exceeded", pe.Code)
	assert.Equal(t, "Rate limit exceeded", pe.Message)
	assert.True(t, pe.Retryable)
}
