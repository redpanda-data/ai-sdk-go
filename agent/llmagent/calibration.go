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

package llmagent

import (
	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// Calibration reconciles the token estimate with what the provider actually
// bills. The heuristic (token_estimate.go) is biased high for English prose,
// but token-dense content - URLs, markup, non-Latin scripts - can tokenize
// at fewer chars per token than even the biased divisor assumes, which would
// silently defer compaction until the provider rejects the request. Every
// response's reported usage measures the estimator's true error on this
// session's content; the observed billed/estimated ratio converts the budget
// lines into estimate units (contextBudget.scaled).
//
// The scale is monotone - it only ever tightens the budget - so calibration
// can never make compaction lazier than the uncalibrated heuristic. It is
// written to the session metadata alongside the invocation metadata, so a
// persisted session carries its learned density into the next invocation;
// only the very first request on an unseeded session is covered by the
// heuristic plus the reactive overflow path. Applications that load a
// recorded session with a known provider-reported size can seed the scale
// up front with CalibrateSession.

// tokenScaleKey stores the observed billed/estimated input ratio on both the
// invocation metadata and the session metadata.
const tokenScaleKey = "llmagent.token_scale" //nolint:gosec // G101: LLM token accounting, not a credential

// tokenScale reads the observed scale - the invocation's own measurement or
// one persisted with the session, whichever is tighter; 1 means uncalibrated.
func tokenScale(inv *agent.InvocationMetadata) float64 {
	scale := 1.0

	if s, ok := inv.GetMetadata(tokenScaleKey).(float64); ok && s > scale {
		scale = s
	}

	if sess := inv.Session(); sess != nil {
		if s, ok := sess.Metadata[tokenScaleKey].(float64); ok && s > scale {
			scale = s
		}
	}

	return scale
}

// observeUsage updates the scale from one request's billed input tokens
// against the estimate of that same request. Ratchets up only, and persists
// the measurement on the session so the next invocation starts calibrated.
func observeUsage(inv *agent.InvocationMetadata, billed, estimated int) {
	if billed <= 0 || estimated <= 0 {
		return
	}

	s := float64(billed) / float64(estimated)
	if s <= tokenScale(inv) {
		return
	}

	inv.SetMetadata(tokenScaleKey, s)

	if sess := inv.Session(); sess != nil {
		if sess.Metadata == nil {
			sess.Metadata = make(map[string]any)
		}

		sess.Metadata[tokenScaleKey] = s
	}
}

// CalibrateSession seeds the compaction token calibration for a session whose
// real size is already known - typically one loaded from a recording that
// carries the provider's reported input tokens. Without a seed, the first
// request of the first invocation runs on the uncalibrated heuristic, which
// undercounts token-dense content and can defer compaction past the point the
// caller knows the session has reached. The seed only ever tightens the
// budget: a reported size at or below the heuristic estimate is a no-op.
func CalibrateSession(sess *session.State, reportedInputTokens int) {
	if sess == nil || reportedInputTokens <= 0 {
		return
	}

	estimated := estimateHistoryTokens(sess.Messages)
	if estimated <= 0 {
		return
	}

	s := float64(reportedInputTokens) / float64(estimated)
	if s <= 1 {
		return
	}

	if existing, ok := sess.Metadata[tokenScaleKey].(float64); ok && existing >= s {
		return
	}

	if sess.Metadata == nil {
		sess.Metadata = make(map[string]any)
	}

	sess.Metadata[tokenScaleKey] = s
}
