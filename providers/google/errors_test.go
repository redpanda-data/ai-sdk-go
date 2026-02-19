package google

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

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

func TestClassifyAPIError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		code      int
		status    string
		message   string
		wantBase  error
		wantRetry bool
	}{
		{"rate limit", 429, "RESOURCE_EXHAUSTED", "Rate limit exceeded", llm.ErrRateLimitExceeded, true},
		{"server error 500", 500, "INTERNAL", "Internal server error", llm.ErrServerError, true},
		{"bad gateway 502", 502, "UNAVAILABLE", "Bad gateway", llm.ErrServerError, true},
		{"service unavailable 503", 503, "UNAVAILABLE", "Service unavailable", llm.ErrServerError, true},
		{"bad request 400", 400, "INVALID_ARGUMENT", "Invalid request", llm.ErrInvalidInput, false},
		{"unauthorized 401", 401, "UNAUTHENTICATED", "Unauthorized", llm.ErrAPICall, false},
		{"forbidden 403", 403, "PERMISSION_DENIED", "Forbidden", llm.ErrAPICall, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			apiErr := &genai.APIError{
				Code:    tt.code,
				Status:  tt.status,
				Message: tt.message,
			}

			result := classifyError(apiErr)
			require.Error(t, result)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.Equal(t, tt.wantRetry, pe.Retryable)
			assert.Equal(t, tt.status, pe.Code)
			assert.Equal(t, tt.message, pe.Message)
		})
	}
}
