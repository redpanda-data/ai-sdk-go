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

// ReasoningSupport describes which reasoning controls an offering
// accepts. The shape follows the former bedrock.ThinkingSupport, which
// all providers' catalogs converged on.
//
// The capability itself ("this model reasons") lives in
// llm.ModelCapabilities.Reasoning; this struct carries the wire controls
// for it, which genuinely differ per host.
type ReasoningSupport struct {
	// Efforts are the effort values the offering accepts, in ascending
	// order. Empty means the offering has no effort control — providers
	// reject a requested effort against an empty list rather than
	// passing it through.
	//
	// llm.ReasoningEffort is an open string type; the valid vocabulary
	// is provider-owned and these lists are its source of truth.
	Efforts []llm.ReasoningEffort

	// Adaptive reports whether the offering accepts adaptive thinking
	// (the model decides how long to think).
	Adaptive bool

	// Budget reports whether the offering accepts a manual thinking
	// token budget.
	Budget bool
}
