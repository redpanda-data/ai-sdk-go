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

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestClassify maps each host's answer, as its docs show it, to an outcome.
// "Not found" is the host saying the ID doesn't exist; "needs capacity or
// policy" is the host knowing the model but refusing this caller (a quota,
// a region, an access gate); anything else that isn't a success is
// inconclusive and goes to a person.
func TestClassify(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		status  int
		code    string
		message string
		want    string
	}{
		{"any host: success", 200, "", "", outcomeCallable},
		{"anthropic: unknown model", 404, "not_found_error", "model: claude-nope", outcomeNotFound},
		{"openai: unknown model", 404, "model_not_found", "The model `gpt-nope` does not exist or you do not have access to it.", outcomeNotFound},
		{"gemini: unknown model", 404, "NOT_FOUND", "models/gemini-nope is not found for API version v1beta", outcomeNotFound},
		{"bedrock: unknown model", 404, "ResourceNotFoundException", "Model not found", outcomeNotFound},
		{"bedrock: invalid model ID", 400, "ValidationException", "The provided model identifier is invalid.", outcomeNotFound},
		{"bedrock: needs a profile", 400, "ValidationException", "Invocation of model ID anthropic.claude-z with on-demand throughput isn't supported. Retry your request with the ID or ARN of an inference profile that contains this model.", outcomeNeedsCapacityOrPolicy},
		{"bedrock: access gate", 403, "AccessDeniedException", "You don't have access to the model with the specified model ID.", outcomeNeedsCapacityOrPolicy},
		{"bedrock: quota", 400, "ServiceQuotaExceededException", "Too many tokens", outcomeNeedsCapacityOrPolicy},
		{"bedrock: throttled", 429, "ThrottlingException", "Too many requests", outcomeNeedsCapacityOrPolicy},
		{"openai: region", 403, "unsupported_country_region_territory", "Country, region, or territory not supported", outcomeNeedsCapacityOrPolicy},
		{"gemini: location", 400, "FAILED_PRECONDITION", "User location is not supported for the API use.", outcomeNeedsCapacityOrPolicy},
		{"gemini: no quota", 429, "RESOURCE_EXHAUSTED", "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0", outcomeNeedsCapacityOrPolicy},
		{"anthropic: bad key", 401, "authentication_error", "invalid x-api-key", outcomeInconclusive},
		{"openai: bad parameter", 400, "unsupported_parameter", "Unsupported parameter: 'max_completion_tokens'", outcomeInconclusive},
		{"anthropic: overloaded", 529, "overloaded_error", "Overloaded", outcomeInconclusive},
		{"bedrock: unavailable", 503, "ServiceUnavailableException", "", outcomeInconclusive},
		{"no answer", 0, "", "dial tcp: i/o timeout", outcomeInconclusive},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tc.want, classify(tc.status, tc.code, tc.message))
		})
	}
}
