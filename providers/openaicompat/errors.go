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

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	codeResponseDecode = "response_decode_error"
	codeTransport      = "transport_error"
	codeStreamError    = "stream_error"
)

// classifyError maps SDK errors to *llm.ProviderError.
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

func classifyNonHTTPError(err error) error {
	// Returned untouched: ProviderError.Unwrap yields only Base, so wrapping
	// these would break errors.Is against them at the call site.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}

	// Not retryable: the frame's payload is opaque here, and a mid-stream
	// failure is as likely to be terminal as transient.
	var streamErr *ssestream.StreamError
	if errors.As(err, &streamErr) {
		return &llm.ProviderError{
			Base:    llm.ErrAPICall,
			Code:    codeStreamError,
			Message: streamErr.Error(),
		}
	}

	// Well-formed JSON disagreeing with our structs replays identically.
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) {
		return &llm.ProviderError{
			Base:    llm.ErrResponseMapping,
			Code:    codeResponseDecode,
			Message: fmt.Sprintf("could not decode provider response: %s", err),
		}
	}

	// A mangled stream; a second attempt can land on a clean response.
	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      codeResponseDecode,
			Message:   fmt.Sprintf("could not decode provider response: %s", err),
			Retryable: true,
		}
	}

	// Do not narrow this to net.Error or an errno set. An h2 stream reset is
	// net/http's unexported http2StreamError: not a net.Error, unreachable by
	// errors.As, and h2 is the default for https upstreams. Whatever such a
	// check missed would silently become permanent, which costs the caller's
	// whole task where a needless retry costs three short attempts.
	return &llm.ProviderError{
		Base:      llm.ErrServerError,
		Code:      codeTransport,
		Message:   fmt.Sprintf("transport failure: %s", err),
		Retryable: true,
	}
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
