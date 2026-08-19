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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Codes for failures that never got far enough to carry an HTTP status. An
// OpenAI-compatible upstream has no error taxonomy of its own for these, so we
// supply one: without it the errors below reach the caller with no category and
// no code, which is exactly how an openai-go SSE decoding bug presented as a
// bare "unexpected end of JSON input" with nothing in the error to point at.
const (
	// codeResponseDecode: the upstream returned bytes we could not decode as an
	// OpenAI-shaped response.
	codeResponseDecode = "response_decode_error"

	// codeTransport: the request failed below HTTP — connection reset,
	// truncated body, DNS failure.
	codeTransport = "transport_error"

	// codeStreamError: the upstream reported a failure inside an SSE data frame
	// instead of as an HTTP error.
	codeStreamError = "stream_error"
)

// classifyError maps OpenAI SDK errors to *llm.ProviderError with the
// appropriate sentinel base and Retryable flag.
// This is used by both the model and response mapper.
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return classifyHTTPError(apiErr)
	}

	return classifyNonHTTPError(err)
}

// classifyNonHTTPError handles the failures that carry no HTTP status: decode
// failures, transport failures, and errors the upstream delivered inside an SSE
// frame. Anything it does not recognise is returned unchanged.
func classifyNonHTTPError(err error) error {
	// Cancellation and deadlines belong to the caller, not to the provider.
	// Return them untouched: ProviderError.Unwrap yields only Base, so wrapping
	// would sever the chain and break errors.Is at the call site.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}

	var streamErr *ssestream.StreamError
	if errors.As(err, &streamErr) {
		// No status code to map — the error arrived as a data frame. Not marked
		// retryable: the payload is opaque at this layer, and a mid-stream
		// failure is as likely to be terminal as transient.
		return &llm.ProviderError{
			Base:    llm.ErrAPICall,
			Code:    codeStreamError,
			Message: streamErr.Error(),
		}
	}

	var (
		syntaxErr *json.SyntaxError
		typeErr   *json.UnmarshalTypeError
	)

	if errors.As(err, &syntaxErr) || errors.As(err, &typeErr) {
		// Retryable: the usual cause is protocol noise on a slow upstream
		// response, which the same request often survives on a second attempt.
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      codeResponseDecode,
			Message:   fmt.Sprintf("could not decode provider response: %s", err),
			Retryable: true,
		}
	}

	var netErr net.Error
	if errors.As(err, &netErr) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      codeTransport,
			Message:   fmt.Sprintf("transport failure: %s", err),
			Retryable: true,
		}
	}

	return err
}

// classifyHTTPError maps an OpenAI HTTP API error to a *llm.ProviderError.
func classifyHTTPError(apiErr *openai.Error) *llm.ProviderError {
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
