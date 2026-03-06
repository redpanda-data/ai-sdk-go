package bedrock

import (
	"errors"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithy "github.com/aws/smithy-go"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// classifyError maps AWS SDK errors to *llm.ProviderError with the
// appropriate sentinel base and Retryable flag.
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	// Type-assert specific AWS error types
	var throttling *types.ThrottlingException
	if errors.As(err, &throttling) {
		return &llm.ProviderError{
			Base:      llm.ErrRateLimitExceeded,
			Code:      "ThrottlingException",
			Message:   throttling.ErrorMessage(),
			Retryable: true,
		}
	}

	var quotaExceeded *types.ServiceQuotaExceededException
	if errors.As(err, &quotaExceeded) {
		return &llm.ProviderError{
			Base:      llm.ErrRateLimitExceeded,
			Code:      "ServiceQuotaExceededException",
			Message:   quotaExceeded.ErrorMessage(),
			Retryable: true,
		}
	}

	var validation *types.ValidationException
	if errors.As(err, &validation) {
		return &llm.ProviderError{
			Base:      llm.ErrInvalidInput,
			Code:      "ValidationException",
			Message:   validation.ErrorMessage(),
			Retryable: false,
		}
	}

	var accessDenied *types.AccessDeniedException
	if errors.As(err, &accessDenied) {
		return &llm.ProviderError{
			Base:      llm.ErrAPICall,
			Code:      "AccessDeniedException",
			Message:   accessDenied.ErrorMessage(),
			Retryable: false,
		}
	}

	var notFound *types.ResourceNotFoundException
	if errors.As(err, &notFound) {
		return &llm.ProviderError{
			Base:      llm.ErrAPICall,
			Code:      "ResourceNotFoundException",
			Message:   notFound.ErrorMessage(),
			Retryable: false,
		}
	}

	var timeout *types.ModelTimeoutException
	if errors.As(err, &timeout) {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      "ModelTimeoutException",
			Message:   timeout.ErrorMessage(),
			Retryable: true,
		}
	}

	var internal *types.InternalServerException
	if errors.As(err, &internal) {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      "InternalServerException",
			Message:   internal.ErrorMessage(),
			Retryable: true,
		}
	}

	var unavailable *types.ServiceUnavailableException
	if errors.As(err, &unavailable) {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      "ServiceUnavailableException",
			Message:   unavailable.ErrorMessage(),
			Retryable: true,
		}
	}

	var modelErr *types.ModelErrorException
	if errors.As(err, &modelErr) {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      "ModelErrorException",
			Message:   modelErr.ErrorMessage(),
			Retryable: true,
		}
	}

	var streamErr *types.ModelStreamErrorException
	if errors.As(err, &streamErr) {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      "ModelStreamErrorException",
			Message:   streamErr.ErrorMessage(),
			Retryable: true,
		}
	}

	var modelNotReady *types.ModelNotReadyException
	if errors.As(err, &modelNotReady) {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      "ModelNotReadyException",
			Message:   modelNotReady.ErrorMessage(),
			Retryable: true,
		}
	}

	// Try smithy OperationError for HTTP status code classification
	var opErr *smithy.OperationError
	if errors.As(err, &opErr) {
		return &llm.ProviderError{
			Base:      llm.ErrAPICall,
			Code:      opErr.OperationName,
			Message:   opErr.Error(),
			Retryable: false,
		}
	}

	return err
}
