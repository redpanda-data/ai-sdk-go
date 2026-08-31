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

import (
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Entry is the authored shape for one offering: one invokable model ID on
// one provider. Every provider package fills the same struct; New
// validates a provider's entries and freezes them into a Catalog.
//
// Provider-specific wire details that do not generalize (Google's numeric
// thinking-budget ranges, Bedrock's mantle routing) stay in provider-local
// side tables keyed by ID rather than growing fields here.
type Entry struct {
	// ID is the provider-native offering identifier, exactly what the
	// provider's NewModel accepts: "claude-opus-5",
	// "us.anthropic.claude-opus-5", "gpt-5.6-sol". Unique within a
	// provider.
	ID string

	// Model is the canonical cross-provider identity; it must exist in
	// the Registry the Catalog is built with. Different offerings of the
	// same model share it: Anthropic's ID "claude-opus-5" and Bedrock's
	// ID "us.anthropic.claude-opus-5" both set
	// Model: catalog.ModelClaudeOpus5 ("anthropic/claude-opus-5").
	Model ModelID

	// DisplayName is the display name including provider decoration:
	// "Claude Opus 5 (EU)". Empty defaults to Facts.DisplayName.
	DisplayName string

	// Aliases are additional exact IDs this provider accepts for this
	// offering ("gpt-5.6" on the gpt-5.6-sol entry). They participate in
	// resolution and in PricingByID.
	Aliases []string

	// Capabilities' Vision/Audio booleans are derived from the Modalities
	// list (image ⇒ Vision, audio ⇒ Audio); authoring them is optional,
	// and an authored true without the matching modality is rejected.
	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints

	// Modalities lists input/output content kinds. Empty normalizes to
	// text-only.
	Modalities Modalities

	// Reasoning describes which reasoning controls the offering accepts.
	Reasoning ReasoningSupport

	// Speeds are the inference speed modes the offering accepts. Empty
	// means standard only.
	Speeds []llm.Speed

	// Pricing is the offering's rate card. Required: input and output
	// rates must be priced (or explicitly pricing.RateFree).
	Pricing pricing.Info

	// Life is the provider's lifecycle schedule for this offering.
	Life Lifecycle

	// Attributes is a narrow escape hatch for provider-specific
	// discovery flags, so one provider's quirk never grows a field on
	// this shared struct. In use: Bedrock sets
	// "requires_provider_data_sharing": "true" on models that reject
	// requests until the account opts in — the conformance suite reads
	// it to skip those, and UIs badge it. Keys are snake_case; values
	// are strings so the snapshot stays stable.
	Attributes map[string]string
}
