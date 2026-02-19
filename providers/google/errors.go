package google

import (
	"errors"

	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// classifyError maps Google genai SDK errors to *llm.ProviderError with the
// appropriate sentinel base and Retryable flag.
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *genai.APIError
	if errors.As(err, &apiErr) {
		return classifyAPIError(apiErr)
	}

	return err
}

// classifyAPIError maps a Google APIError to a *llm.ProviderError.
func classifyAPIError(apiErr *genai.APIError) *llm.ProviderError {
	retryable, base := classifyStatusCode(apiErr.Code)

	return &llm.ProviderError{
		Base:      base,
		Code:      apiErr.Status,
		Message:   apiErr.Message,
		Retryable: retryable,
	}
}

// classifyStatusCode maps HTTP status codes to sentinel errors and retryability.
func classifyStatusCode(code int) (bool, error) {
	switch code {
	case 429:
		return true, llm.ErrRateLimitExceeded
	case 500, 502, 503, 529:
		return true, llm.ErrServerError
	case 400:
		return false, llm.ErrInvalidInput
	case 401, 403:
		return false, llm.ErrAPICall
	default:
		if code >= 500 {
			return true, llm.ErrServerError
		}

		return false, llm.ErrAPICall
	}
}
