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

import "maps"

// TokenUsage is the normalized, cross-provider accounting of tokens for a
// single LLM call.
//
// # Disjoint counters
//
// All token counters are DISJOINT. A prompt token is counted in exactly one of
// InputTokens, CachedInputTokens, CacheCreation5mTokens, CacheCreation1hTokens,
// CacheCreationUnknownTTLTokens, or ToolUseInputTokens. Likewise output tokens
// are counted in exactly one of OutputTokens or ReasoningTokens.
//
// This matches Anthropic's and Bedrock's native convention. OpenAI and Google
// extractors un-subset their native values so the invariant holds uniformly
// (see the per-provider response mappers).
//
// # Zero vs not-reported
//
// A zero value is ambiguous: it may mean "reported zero" or "not surfaced by
// the provider or the extractor." Consumers that need to distinguish should
// consult the provider's raw response. Within the SDK, extractors populate
// fields whenever the underlying SDK exposes them; additional fields are
// wired in over subsequent releases.
//
// # Billed totals
//
// Use BilledInputTokens, BilledOutputTokens, and TotalBilledTokens to compute
// sums. Counters are disjoint so these are simple additions; the helpers
// document intent and stay stable as fields are added.
//
// # What is NOT in TokenUsage
//
// Per-call metadata that describes HOW a request was processed
// (service tier, speed, inference region, invoked model ID, latencies) lives
// on llm.Response, not here. Those fields are not meaningfully additive
// across calls.
type TokenUsage struct {
	// InputTokens is the number of fresh (non-cached, non-cache-write,
	// non-tool-use) prompt tokens charged at the base input rate.
	InputTokens int `json:"input_tokens"`

	// CachedInputTokens is the number of prompt tokens served from the
	// provider's cache (cache read). Disjoint from InputTokens. Billed at a
	// reduced rate.
	//
	// Provider coverage: OpenAI (prompt_tokens_details.cached_tokens),
	// Anthropic (cache_read_input_tokens), Google (cached_content_token_count),
	// Bedrock Converse (cache_read_input_tokens), OpenAI-compatible APIs.
	CachedInputTokens int `json:"cached_input_tokens,omitempty"`

	// CacheCreation5mTokens is the number of prompt tokens written to the
	// provider's 5-minute-TTL cache. Disjoint from InputTokens. Billed at an
	// elevated rate.
	//
	// Provider coverage: Anthropic, Bedrock-Anthropic.
	CacheCreation5mTokens int `json:"cache_creation_5m_tokens,omitempty"`

	// CacheCreation1hTokens is the number of prompt tokens written to the
	// provider's 1-hour-TTL cache. Disjoint from InputTokens. Billed at a
	// higher elevated rate than 5m writes.
	//
	// Provider coverage: Anthropic, Bedrock-Anthropic.
	CacheCreation1hTokens int `json:"cache_creation_1h_tokens,omitempty"`

	// CacheCreationUnknownTTLTokens is the number of prompt tokens written
	// to the provider's cache when a TTL-specific field was not available
	// (older API shapes, future unknown TTLs). Disjoint from InputTokens and
	// from the 5m/1h counters.
	//
	// Extractors fall back to this field when a provider reports an
	// aggregate cache-write count without a per-TTL breakdown, so the total
	// stays reflected in BilledInputTokens() without pretending to know the
	// TTL. Provider coverage: OpenAI, Anthropic, Bedrock-Anthropic.
	CacheCreationUnknownTTLTokens int `json:"cache_creation_unknown_ttl_tokens,omitempty"`

	// ToolUseInputTokens is the number of prompt tokens consumed by
	// server-side tool invocations (e.g., function-calling loops billed
	// separately from the user-visible prompt). Disjoint from InputTokens.
	//
	// Provider coverage: Google (tool_use_prompt_token_count). Zero for
	// providers that fold tool-use tokens into InputTokens.
	ToolUseInputTokens int `json:"tool_use_input_tokens,omitempty"`

	// OutputTokens is the number of tokens in the assistant's visible
	// response. Does NOT include reasoning tokens — those are a separate
	// disjoint bucket (see ReasoningTokens).
	OutputTokens int `json:"output_tokens"`

	// ReasoningTokens is the number of tokens the model spent on hidden
	// reasoning/thinking. Disjoint from OutputTokens — the SDK un-subsets
	// OpenAI's native value so the invariant holds uniformly. Typically
	// billed at the output rate.
	//
	// Provider coverage: OpenAI o-series / GPT-5
	// (completion_tokens_details.reasoning_tokens), Google Gemini 2.5+
	// (thoughts_token_count), OpenAI-compatible reasoning models. Anthropic
	// and Bedrock do not surface thinking tokens as a separate counter —
	// Anthropic's thinking content is billed as regular output and
	// ReasoningTokens will be zero.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`

	// Extra carries provider-specific dimensions that the SDK does not
	// normalize today. Keys MUST be namespaced as "<provider>.<field>" to
	// prevent collisions (e.g., "anthropic.iterations",
	// "cohere.search_units", "bedrock.invocation_latency_ms").
	//
	// Extra is NOT included in BilledInputTokens / BilledOutputTokens —
	// anything billable should graduate to a first-class field.
	Extra map[string]any `json:"extra,omitempty"`
}

// BilledInputTokens returns the total input-side tokens that contribute to
// billing: InputTokens + CachedInputTokens + CacheCreation5mTokens +
// CacheCreation1hTokens + CacheCreationUnknownTTLTokens + ToolUseInputTokens.
//
// Counters are disjoint so this is a simple sum; the helper exists to
// document intent and remain stable as fields are added.
func (u *TokenUsage) BilledInputTokens() int {
	if u == nil {
		return 0
	}

	return u.InputTokens +
		u.CachedInputTokens +
		u.CacheCreation5mTokens +
		u.CacheCreation1hTokens +
		u.CacheCreationUnknownTTLTokens +
		u.ToolUseInputTokens
}

// BilledOutputTokens returns the total output-side tokens that contribute
// to billing: OutputTokens + ReasoningTokens.
func (u *TokenUsage) BilledOutputTokens() int {
	if u == nil {
		return 0
	}

	return u.OutputTokens + u.ReasoningTokens
}

// TotalBilledTokens returns BilledInputTokens + BilledOutputTokens.
func (u *TokenUsage) TotalBilledTokens() int {
	return u.BilledInputTokens() + u.BilledOutputTokens()
}

// SumUsage aggregates multiple TokenUsage values into a single cumulative
// result. Nil values are safely skipped. Returns nil if all inputs are nil.
//
// All scalar counters are added. The Extra map is merged per key; when
// both sides hold the same numeric type (int, int64, or float64) the
// values are summed. Any other collision (different numeric types,
// booleans, strings, nested maps) keeps the first-writer-wins value
// rather than silently coercing across types. Anything billable should
// graduate to a first-class field instead of relying on Extra merging.
//
// TokenUsage intentionally contains only additive accounting fields.
// Per-call metadata (service tier, speed, region, invoked model) lives on
// llm.Response, which is not summed.
//
// Example:
//
//	total := llm.SumUsage(turn1Usage, turn2Usage, turn3Usage)
func SumUsage(usages ...*TokenUsage) *TokenUsage {
	var result *TokenUsage

	for _, u := range usages {
		if u == nil {
			continue
		}

		if result == nil {
			result = cloneUsage(u)
			continue
		}

		result.InputTokens += u.InputTokens
		result.CachedInputTokens += u.CachedInputTokens
		result.CacheCreation5mTokens += u.CacheCreation5mTokens
		result.CacheCreation1hTokens += u.CacheCreation1hTokens
		result.CacheCreationUnknownTTLTokens += u.CacheCreationUnknownTTLTokens
		result.ToolUseInputTokens += u.ToolUseInputTokens

		result.OutputTokens += u.OutputTokens
		result.ReasoningTokens += u.ReasoningTokens

		result.Extra = mergeExtra(result.Extra, u.Extra)
	}

	return result
}

func cloneUsage(u *TokenUsage) *TokenUsage {
	out := *u
	out.Extra = cloneExtra(u.Extra)

	return &out
}

func cloneExtra(m map[string]any) map[string]any {
	if len(m) == 0 {
		return nil
	}

	out := make(map[string]any, len(m))
	maps.Copy(out, m)

	return out
}

func mergeExtra(dst, src map[string]any) map[string]any {
	if len(src) == 0 {
		return dst
	}

	if dst == nil {
		dst = make(map[string]any, len(src))
	}

	for k, v := range src {
		existing, ok := dst[k]
		if !ok {
			dst[k] = v
			continue
		}

		// Same-typed numeric values add; any other collision — including
		// differently-typed numerics like int vs int64 — keeps the first
		// writer rather than silently coercing. Billable counters should
		// graduate to a first-class TokenUsage field.
		switch existingValue := existing.(type) {
		case int:
			if incoming, ok := v.(int); ok {
				dst[k] = existingValue + incoming
			}
		case int64:
			if incoming, ok := v.(int64); ok {
				dst[k] = existingValue + incoming
			}
		case float64:
			if incoming, ok := v.(float64); ok {
				dst[k] = existingValue + incoming
			}
		}
	}

	return dst
}

// ServiceTier lives in llm/service_tier.go alongside NormalizeServiceTier.
// The pricing package consumes it as one selector dimension when choosing a
// rate card.

// FinishReason indicates why model generation stopped.
type FinishReason string

// FinishReason constants provide standardized values across providers.
const (
	// FinishReasonStop indicates the model completed naturally.
	FinishReasonStop FinishReason = "stop"

	// FinishReasonLength indicates the response was truncated due to length limits.
	FinishReasonLength FinishReason = "length"

	// FinishReasonToolCalls indicates the model wants to execute tools.
	FinishReasonToolCalls FinishReason = "tool_calls"

	// FinishReasonContentFilter indicates content was blocked by safety filters.
	FinishReasonContentFilter FinishReason = "content_filter"

	// FinishReasonInterrupted indicates the request was cancelled or interrupted.
	FinishReasonInterrupted FinishReason = "interrupted"

	// FinishReasonUnknown is used when the provider returns an unrecognized reason.
	FinishReasonUnknown FinishReason = "unknown"
)

// ModelCapabilities describes what features a model supports.
// This enables compile-time and runtime validation of requests.
type ModelCapabilities struct {
	Streaming        bool // Supports streaming responses
	Tools            bool // Supports function/tool calling
	JSONMode         bool // Supports JSON mode (response_format: json_object) - ensures valid JSON output
	StructuredOutput bool // Supports Structured Outputs (response_format: json_schema) - ensures schema adherence
	Vision           bool // Supports image inputs
	Audio            bool // Supports audio inputs
	MultiTurn        bool // Supports conversation history
	SystemPrompts    bool // Supports system role messages
	Reasoning        bool // Supports reasoning controls and exposes reasoning traces
}

// ModelPositioning describes whether a model should be recommended for new
// workloads. It is intentionally independent from ModelLifecycle: a model can
// remain active at its provider while being superseded by a newer model in the
// same recommendation group.
type ModelPositioning string

const (
	// ModelPositioningFrontier identifies a provider's highest-capability
	// current model.
	ModelPositioningFrontier ModelPositioning = "frontier"
	// ModelPositioningModern identifies a current, useful model that serves a
	// distinct cost, latency, or capability tier.
	ModelPositioningModern ModelPositioning = "modern"
	// ModelPositioningLegacy identifies a superseded model retained for
	// backwards compatibility.
	ModelPositioningLegacy ModelPositioning = "legacy"
)

// ModelLifecycle is the provider-confirmed availability state of a model.
// Dates never implicitly change this value. In particular, Retired is only set
// after an official provider source confirms that the model is unavailable.
type ModelLifecycle string

const (
	ModelLifecycleActive     ModelLifecycle = "active"
	ModelLifecycleDeprecated ModelLifecycle = "deprecated"
	ModelLifecycleRetired    ModelLifecycle = "retired"
)

// ModelReleaseStage describes the provider's stability commitment for a
// model, independently from recommendation and lifecycle state.
type ModelReleaseStage string

const (
	ModelReleaseStageStable  ModelReleaseStage = "stable"
	ModelReleaseStagePreview ModelReleaseStage = "preview"
)

// ModelCatalogMetadata is provider-scoped metadata used by model discovery
// surfaces to recommend current models without breaking existing resources
// that reference older IDs. FamilyKey is the stable product family used for UI
// grouping. RecommendationGroup is the narrower upgrade track; Bedrock groups
// include the routing geography because newer models are not always available
// in every inference profile.
//
// EndOfLifeDate and VerifiedDate use ISO 8601 calendar dates (YYYY-MM-DD), not
// timestamps. EndOfLifeDate is informational and never drives Lifecycle.
type ModelCatalogMetadata struct {
	FamilyKey           string
	RecommendationGroup string
	Positioning         ModelPositioning
	Lifecycle           ModelLifecycle
	ReleaseStage        ModelReleaseStage
	EndOfLifeDate       string
	Replacement         string
	OfficialSourceURL   string
	VerifiedDate        string
}

// ModelCatalogProvider exposes typed recommendation and lifecycle metadata
// without changing ModelDiscoveryInfo's public struct layout. Implementations
// resolve official aliases and dated snapshots to the same metadata as their
// canonical discovery entry.
type ModelCatalogProvider interface {
	ModelCatalog(model string) (ModelCatalogMetadata, bool)
}

// ModelCatalogOverridesProvider optionally enumerates exact lifecycle records
// for model IDs that are intentionally absent from Provider.Models(). These
// records let discovery clients classify existing resources that reference a
// retired or deprecated snapshot without offering that snapshot for new
// selections. Equivalent active aliases must remain excluded.
//
// The returned map is owned by the caller and may be modified safely.
type ModelCatalogOverridesProvider interface {
	ModelCatalogOverrides() map[string]ModelCatalogMetadata
}

// ModelDiscoveryInfo provides metadata about a model that can be discovered at
// runtime, without constructing the model. It is the static counterpart of the
// ModelInfo interface: Name, Provider, Capabilities, and Constraints mirror the
// ModelInfo accessors, plus discovery-only fields (Label, Metadata).
// This is returned by provider.Models() for model discovery and capability checking.
type ModelDiscoveryInfo struct {
	// Name is the model identifier used in API calls
	Name string

	// Label is a human-readable display name
	Label string

	// Capabilities describes what features this model supports
	Capabilities ModelCapabilities

	// Constraints carries the model's validation rules and token limits.
	// MaxInputTokens is the context window size; MaxOutputTokens the
	// per-response generation cap.
	Constraints ModelConstraints

	// Provider is the name of the provider that offers this model
	Provider string

	// Metadata carries provider-specific model metadata for discovery surfaces.
	Metadata map[string]string
}
