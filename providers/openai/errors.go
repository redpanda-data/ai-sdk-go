package openai

import (
	"errors"

	oai "github.com/openai/openai-go/v3"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// classifyError maps OpenAI SDK errors to *llm.ProviderError with the
// appropriate sentinel base and Retryable flag.
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *oai.Error
	if errors.As(err, &apiErr) {
		return classifyHTTPError(apiErr)
	}

	return err
}

// classifyHTTPError maps an OpenAI HTTP API error to a *llm.ProviderError.
func classifyHTTPError(apiErr *oai.Error) *llm.ProviderError {
	retryable, base := classifyStatusCode(apiErr.StatusCode)

	return &llm.ProviderError{
		Base:      base,
		Code:      apiErr.Code,
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
