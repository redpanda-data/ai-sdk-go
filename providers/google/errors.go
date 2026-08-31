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

package google

import (
	"errors"
	"net"
	"strings"

	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// classifyError maps Google genai SDK errors to *llm.ProviderError with the
// appropriate sentinel base and Retryable flag. The genai SDK returns APIError
// by value, so the errors.As target must be a value — a pointer target never
// matches.
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return classifyAPIError(apiErr)
	}

	// Network timeouts (e.g., Client.Timeout exceeded) are retryable.
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return &llm.ProviderError{
			Base:      llm.ErrServerError,
			Code:      "TIMEOUT",
			Message:   err.Error(),
			Retryable: true,
		}
	}

	return err
}

// classifyAPIError maps a Google APIError to a *llm.ProviderError.
func classifyAPIError(apiErr genai.APIError) *llm.ProviderError {
	retryable, base := classifyStatusCode(apiErr.Code)

	if errors.Is(base, llm.ErrInvalidInput) && isContextOverflowMessage(apiErr.Message) {
		base = llm.ErrContextOverflow
	}

	return &llm.ProviderError{
		Base:      base,
		Code:      apiErr.Status,
		Message:   apiErr.Message,
		Retryable: retryable,
	}
}

// isContextOverflowMessage reports whether a 400 message describes the input
// exceeding the model's context window. The API carries no machine-readable
// reason for this, so message matching is the only option.
func isContextOverflowMessage(msg string) bool {
	msg = strings.ToLower(msg)

	// Out-of-range maxOutputTokens is a config error, not an overflow.
	if strings.Contains(msg, "maxoutputtokens") {
		return false
	}

	if strings.Contains(msg, "exceeds the maximum number of tokens allowed") {
		return true
	}

	// Legacy Vertex wording.
	return strings.Contains(msg, "input token count") &&
		(strings.Contains(msg, "exceeds") || strings.Contains(msg, "model only supports up to"))
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
