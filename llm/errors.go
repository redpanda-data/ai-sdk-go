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
	"fmt"
)

// Common error types used across all providers.
var (
	// ErrRequestMapping indicates failure to convert a Request to provider-specific format.
	ErrRequestMapping = errors.New("request mapping failed")

	// ErrResponseMapping indicates failure to convert a provider response to unified format.
	ErrResponseMapping = errors.New("response mapping failed")

	// ErrAPICall indicates a failure in the provider's API call.
	ErrAPICall = errors.New("API call failed")

	// ErrInvalidConfig indicates provider configuration is invalid.
	ErrInvalidConfig = errors.New("invalid configuration")

	// ErrUnsupportedFeature indicates the requested feature is not supported by the provider/model.
	ErrUnsupportedFeature = errors.New("unsupported feature")

	// ErrStreamClosed indicates the stream was closed unexpectedly.
	ErrStreamClosed = errors.New("stream closed")
)

// Provider error categories as sentinel errors.
// Use errors.Is() to check category, or type assert to ProviderError for details.
var (
	// ErrRateLimitExceeded indicates the provider's rate limit was hit.
	// Retryable with exponential backoff.
	// Maps from errors such as: "rate_limit_exceeded".
	ErrRateLimitExceeded = errors.New("rate limit exceeded")

	// ErrInvalidInput indicates malformed or invalid request input.
	// Not retryable - the caller must fix the input.
	// Maps from errors such as: "invalid_prompt", "invalid_image", "invalid_image_format",
	// 	"invalid_base64_image", "invalid_image_url", "image_too_large",
	// 	"image_too_small", "invalid_image_mode", "image_file_too_large",
	// 	"unsupported_image_media_type", "empty_image_file"
	ErrInvalidInput = errors.New("invalid input")

	// ErrContentPolicyViolation indicates content was blocked by provider safety policies.
	// Not retryable - the content violates provider terms.
	// Maps from errors such as: "image_content_policy_violation"
	// Note: Soft filtering (incomplete response) uses FinishReasonContentFilter instead.
	ErrContentPolicyViolation = errors.New("content policy violation")

	// ErrServerError indicates a transient provider-side error.
	// Retryable with backoff.
	// Maps from errors such as: "server_error", "vector_store_timeout", "image_parse_error",
	//   "failed_to_download_image", "image_file_not_found"
	ErrServerError = errors.New("server error")
)

// ProviderError wraps a sentinel error with provider-specific details.
// The Base field determines the error category for errors.Is() matching.
//
// Usage examples:
//
//	// Check category with errors.Is()
//	if errors.Is(err, llm.ErrRateLimitExceeded) {
//	    // Retry with backoff
//	}
//
//	// Get provider details with type assertion
//	if perr, ok := err.(*llm.ProviderError); ok {
//	    log.Printf("Provider error [%s]: %s", perr.Code, perr.Message)
//	}
type ProviderError struct {
	// Base is the sentinel error indicating the category
	// (ErrRateLimitExceeded, ErrInvalidInput, etc.)
	Base error

	// Code is the provider-specific error code ("rate_limit_exceeded", etc.)
	Code string

	// Message is a human-readable error description from the provider
	Message string

	// Retryable indicates whether this error represents a transient condition
	// that may succeed on retry (e.g., rate limits, server errors).
	Retryable bool
}

// IsRetryable checks if an error represents a transient condition that may
// succeed on retry. It works through error wrapping chains.
//
// Returns true if:
//   - The error is a *ProviderError with Retryable set to true
//   - The error wraps ErrServerError or ErrRateLimitExceeded
func IsRetryable(err error) bool {
	var perr *ProviderError
	if errors.As(err, &perr) {
		return perr.Retryable
	}

	return errors.Is(err, ErrServerError) || errors.Is(err, ErrRateLimitExceeded)
}

// Error implements the error interface.
func (e *ProviderError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: [%s] %s", e.Base, e.Code, e.Message)
	}

	return fmt.Sprintf("%s: %s", e.Base, e.Message)
}

// Unwrap returns the base error for errors.Is/As support.
func (e *ProviderError) Unwrap() error {
	return e.Base
}
