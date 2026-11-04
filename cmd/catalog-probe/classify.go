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

import "strings"

// The outcomes of a probe. Not found: the host says the ID doesn't exist.
// Needs capacity or policy: the host knows the model but won't serve this
// caller (a quota, a region, an access gate). Not subscribed: Bedrock's
// pre-check found no subscription, so nothing was called. Inconclusive:
// anything else, for a person to read. Skipped: never called (plan).
const (
	outcomeCallable              = "callable"
	outcomeNotFound              = "not_found"
	outcomeNeedsCapacityOrPolicy = "needs_capacity_or_policy"
	outcomeNotSubscribed         = "not_subscribed"
	outcomeInconclusive          = "inconclusive"
	outcomeSkipped               = "skipped"
)

// classify turns a host's answer into an outcome: its HTTP status first,
// then the few error codes that say more than the status does. Status 0 is
// no answer at all. Bedrock answers both an invalid ID and a model that
// needs a profile with ValidationException, so only there the message
// decides; a reworded one reads as inconclusive.
func classify(status int, code, message string) string {
	validation := code == "ValidationException"

	switch {
	case status >= 200 && status < 300:
		return outcomeCallable
	case status == 404, validation && strings.Contains(message, "model identifier is invalid"):
		return outcomeNotFound
	case status == 403, status == 429, code == "FAILED_PRECONDITION", code == "ServiceQuotaExceededException",
		validation && strings.Contains(message, "on-demand throughput isn't supported"):
		return outcomeNeedsCapacityOrPolicy
	default:
		return outcomeInconclusive
	}
}
