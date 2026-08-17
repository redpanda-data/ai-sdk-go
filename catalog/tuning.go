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

package catalog

import "github.com/redpanda-data/ai-sdk-go/llm"

// Tuning carries per-offering harness defaults: what a well-configured
// agent or gateway SHOULD do with this model, as opposed to Constraints,
// which say what the model permits. Keeping these in the catalog means a
// model upgrade cannot leave a stale hardcoded number behind in harness
// code.
//
// Every zero value means "no opinion" — consumers fall back to their own
// defaults. Nothing in this SDK acts on Tuning yet; it is the seam the
// gateway and the agent's context management read from.
type Tuning struct {
	// DefaultMaxOutputTokens is the suggested max_tokens for requests
	// that do not set one. Must be ≤ Constraints.MaxOutputTokens.
	DefaultMaxOutputTokens int

	// DefaultReasoningEffort is the suggested effort for requests that
	// do not set one. Must be one of Reasoning.Efforts when both are
	// set.
	DefaultReasoningEffort llm.ReasoningEffort

	// CompactAtInputTokens is the absolute context size at which a
	// harness should compact, set below the model's quality-degradation
	// knee. It is deliberately distinct from Constraints.MaxInputTokens:
	// degradation arrives before the window does, so "it still fits" is
	// the wrong compaction trigger. Must be < Constraints.MaxInputTokens.
	CompactAtInputTokens int
}
