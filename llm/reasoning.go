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

// ReasoningEffort is the cross-provider type for how much work a model
// spends on reasoning before answering.
//
// It is deliberately an open string type with no canonical values in this
// package: effort vocabulary is provider-owned (OpenAI's "none".."max",
// Anthropic's "low".."max", Gemini's "minimal".."high"), and each provider
// package declares typed constants for exactly the values its models accept,
// so package-level autocomplete shows valid choices. Sharing only the type
// keeps the plumbing uniform — one WithReasoningEffort signature, one field
// shape in model catalogs, and the ReasoningEffortLister interface — while a
// new provider can introduce its own vocabulary without touching this
// package.
//
// Which values a specific model supports is model metadata, not type
// information: every provider validates the requested effort against its
// model catalog when the model is constructed and rejects unsupported
// values with the model's supported set in the error.
type ReasoningEffort string

// ReasoningEffortLister is implemented by provider models whose reasoning
// depth can be controlled through ReasoningEffort. Multi-provider consumers
// can discover a model's valid efforts (e.g. to populate a settings UI)
// without depending on the concrete provider type:
//
//	if lister, ok := model.(llm.ReasoningEffortLister); ok {
//	    efforts = lister.SupportedReasoningEfforts()
//	}
//
// An empty result means the model exposes no effort control; it may still
// support provider-specific reasoning options such as manual token budgets.
type ReasoningEffortLister interface {
	// SupportedReasoningEfforts returns the efforts the model accepts, in
	// ascending order of effort.
	SupportedReasoningEfforts() []ReasoningEffort
}
