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
// are counted in exactly one of OutputTokens, ReasoningTokens, or
// RejectedPredictionTokens.
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
	// TTL.
	CacheCreationUnknownTTLTokens int `json:"cache_creation_unknown_ttl_tokens,omitempty"`

	// ToolUseInputTokens is the number of prompt tokens consumed by
	// server-side tool invocations (e.g., function-calling loops billed
	// separately from the user-visible prompt). Disjoint from InputTokens.
	//
	// Provider coverage: Google (tool_use_prompt_token_count). Zero for
	// providers that fold tool-use tokens into InputTokens.
	ToolUseInputTokens int `json:"tool_use_input_tokens,omitempty"`

	// OutputTokens is the number of tokens in the assistant's visible
	// response. Does NOT include reasoning tokens (disjoint, see
	// ReasoningTokens) or rejected predicted-output tokens (disjoint, see
	// RejectedPredictionTokens). Does INCLUDE accepted predicted-output
	// tokens, since those appear in the completion.
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

	// RejectedPredictionTokens is the number of OpenAI Predicted Outputs
	// tokens that did NOT appear in the completion. Billed as output but
	// never returned to the caller, so they are disjoint from OutputTokens.
	//
	// Provider coverage: OpenAI only
	// (completion_tokens_details.rejected_prediction_tokens).
	//
	// Note: accepted predicted-output tokens are NOT a separate counter —
	// they appear verbatim in the completion and are already included in
	// OutputTokens. Consumers that need the breakdown can read the raw
	// provider response.
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`

	// ModalityInputTokens breaks InputTokens down by modality when the
	// provider reports it. The sum over the map equals InputTokens.
	//
	// Provider coverage: Google (prompt_tokens_details[]). OpenAI audio
	// tokens are exposed here when present.
	ModalityInputTokens map[Modality]int `json:"modality_input_tokens,omitempty"`

	// ModalityOutputTokens breaks OutputTokens down by modality.
	ModalityOutputTokens map[Modality]int `json:"modality_output_tokens,omitempty"`

	// ModalityCachedInputTokens breaks CachedInputTokens down by modality.
	ModalityCachedInputTokens map[Modality]int `json:"modality_cached_input_tokens,omitempty"`

	// ServerToolRequests counts billable server-side tool invocations by
	// kind. Each entry is one request, not a token count.
	//
	// Provider coverage: Anthropic (web_search, web_fetch), Google (inferred
	// from GroundingMetadata), Cohere-on-Bedrock (search, classification).
	ServerToolRequests map[ServerTool]int `json:"server_tool_requests,omitempty"`

	// GuardrailUnits counts provider-reported guardrail policy units. Each
	// key is an independent billing SKU on Bedrock.
	//
	// Provider coverage: Bedrock. Keys include "content_policy",
	// "contextual_grounding", "sensitive_info", "sensitive_info_free",
	// "topic_policy", "word_policy".
	GuardrailUnits map[string]int `json:"guardrail_units,omitempty"`

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

// BilledOutputTokens returns the total output-side tokens that contribute to
// billing: OutputTokens + ReasoningTokens + RejectedPredictionTokens.
//
// AcceptedPredictionTokens are NOT added here — they appear verbatim in the
// completion and are already counted inside OutputTokens.
func (u *TokenUsage) BilledOutputTokens() int {
	if u == nil {
		return 0
	}

	return u.OutputTokens + u.ReasoningTokens + u.RejectedPredictionTokens
}

// TotalBilledTokens returns BilledInputTokens + BilledOutputTokens. Non-token
// fees (ServerToolRequests, GuardrailUnits) are not included — those are
// priced separately by the consumer.
func (u *TokenUsage) TotalBilledTokens() int {
	return u.BilledInputTokens() + u.BilledOutputTokens()
}

// SumUsage aggregates multiple TokenUsage values into a single cumulative
// result. Nil values are safely skipped. Returns nil if all inputs are nil.
//
// All scalar counters are added. Map fields (Modality*, ServerToolRequests,
// GuardrailUnits, Extra) are merged with per-key addition for numeric values;
// non-numeric Extra values take the first-writer-wins rule.
//
// TokenUsage intentionally contains only additive accounting fields.
// Per-call metadata (service tier, speed, region, invoked model, latency)
// lives on llm.Response, which is not summed.
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
		result.RejectedPredictionTokens += u.RejectedPredictionTokens

		result.ModalityInputTokens = mergeModality(result.ModalityInputTokens, u.ModalityInputTokens)
		result.ModalityOutputTokens = mergeModality(result.ModalityOutputTokens, u.ModalityOutputTokens)
		result.ModalityCachedInputTokens = mergeModality(result.ModalityCachedInputTokens, u.ModalityCachedInputTokens)
		result.ServerToolRequests = mergeServerTools(result.ServerToolRequests, u.ServerToolRequests)
		result.GuardrailUnits = mergeStringInt(result.GuardrailUnits, u.GuardrailUnits)
		result.Extra = mergeExtra(result.Extra, u.Extra)
	}

	return result
}

func cloneUsage(u *TokenUsage) *TokenUsage {
	out := *u
	out.ModalityInputTokens = cloneModality(u.ModalityInputTokens)
	out.ModalityOutputTokens = cloneModality(u.ModalityOutputTokens)
	out.ModalityCachedInputTokens = cloneModality(u.ModalityCachedInputTokens)
	out.ServerToolRequests = cloneServerTools(u.ServerToolRequests)
	out.GuardrailUnits = cloneStringInt(u.GuardrailUnits)
	out.Extra = cloneExtra(u.Extra)

	return &out
}

func cloneModality(m map[Modality]int) map[Modality]int {
	if len(m) == 0 {
		return nil
	}

	out := make(map[Modality]int, len(m))
	maps.Copy(out, m)

	return out
}

func cloneServerTools(m map[ServerTool]int) map[ServerTool]int {
	if len(m) == 0 {
		return nil
	}

	out := make(map[ServerTool]int, len(m))
	maps.Copy(out, m)

	return out
}

func cloneStringInt(m map[string]int) map[string]int {
	if len(m) == 0 {
		return nil
	}

	out := make(map[string]int, len(m))
	maps.Copy(out, m)

	return out
}

func cloneExtra(m map[string]any) map[string]any {
	if len(m) == 0 {
		return nil
	}

	out := make(map[string]any, len(m))
	maps.Copy(out, m)

	return out
}

func mergeModality(dst, src map[Modality]int) map[Modality]int {
	if len(src) == 0 {
		return dst
	}

	if dst == nil {
		dst = make(map[Modality]int, len(src))
	}

	for k, v := range src {
		dst[k] += v
	}

	return dst
}

func mergeServerTools(dst, src map[ServerTool]int) map[ServerTool]int {
	if len(src) == 0 {
		return dst
	}

	if dst == nil {
		dst = make(map[ServerTool]int, len(src))
	}

	for k, v := range src {
		dst[k] += v
	}

	return dst
}

func mergeStringInt(dst, src map[string]int) map[string]int {
	if len(src) == 0 {
		return dst
	}

	if dst == nil {
		dst = make(map[string]int, len(src))
	}

	for k, v := range src {
		dst[k] += v
	}

	return dst
}

func mergeExtra(dst, src map[string]any) map[string]any {
	if len(src) == 0 {
		return dst
	}

	if dst == nil {
		dst = make(map[string]any, len(src))
	}

	for k, v := range src {
		if existing, ok := dst[k]; ok {
			if ei, eok := existing.(int); eok {
				if vi, vok := v.(int); vok {
					dst[k] = ei + vi
					continue
				}
			}
			// Non-numeric collision: first writer wins.
			continue
		}

		dst[k] = v
	}

	return dst
}

// Modality identifies the medium of a chunk of tokens.
type Modality string

// Modality constants. Providers report a subset of these.
const (
	ModalityText     Modality = "text"
	ModalityImage    Modality = "image"
	ModalityAudio    Modality = "audio"
	ModalityVideo    Modality = "video"
	ModalityDocument Modality = "document"
)

// ServiceTier is the normalized request-processing variant reported on
// llm.Response. Not every provider reports every value; the empty string
// means default / unreported.
type ServiceTier string

// ServiceTier constants cover the union of variants across providers.
const (
	// ServiceTierDefault is the provider's standard tier.
	ServiceTierDefault ServiceTier = "default"

	// ServiceTierFlex is a discounted, best-effort-latency tier
	// (OpenAI "flex", Bedrock "flex", Google ON_DEMAND_FLEX).
	ServiceTierFlex ServiceTier = "flex"

	// ServiceTierPriority is a premium, lower-latency tier
	// (OpenAI "priority", Anthropic "priority", Bedrock "priority",
	// Google ON_DEMAND_PRIORITY).
	ServiceTierPriority ServiceTier = "priority"

	// ServiceTierBatch is the async 50%-discount batch tier (Anthropic
	// "batch", OpenAI Batch API). Note: OpenAI's Batch SKU is a separate
	// endpoint, not a service_tier on the sync API; extractors set this for
	// responses whose origin is the batch endpoint.
	ServiceTierBatch ServiceTier = "batch"

	// ServiceTierScale is OpenAI's enterprise scale tier.
	ServiceTierScale ServiceTier = "scale"

	// ServiceTierReserved is Bedrock's reserved-capacity tier.
	ServiceTierReserved ServiceTier = "reserved"

	// ServiceTierProvisionedThroughput maps Google TrafficType
	// PROVISIONED_THROUGHPUT and Bedrock Provisioned Throughput invocations.
	ServiceTierProvisionedThroughput ServiceTier = "provisioned_throughput"
)

// ServerTool identifies a server-side tool whose invocations are billed per
// request (not per token). Values are lower_snake_case.
type ServerTool string

// ServerTool constants cover the known server-side billable tools across
// providers. Custom values (via type conversion) are acceptable for
// consumer-specific extensions.
const (
	ServerToolWebSearch      ServerTool = "web_search"
	ServerToolWebFetch       ServerTool = "web_fetch"
	ServerToolImageSearch    ServerTool = "image_search"
	ServerToolCodeExecution  ServerTool = "code_execution"
	ServerToolClassification ServerTool = "classification" // Cohere on Bedrock
)

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

// ModelDiscoveryInfo provides metadata about a model that can be discovered at runtime.
// This is returned by provider.Models() for model discovery and capability checking.
type ModelDiscoveryInfo struct {
	// Name is the model identifier used in API calls
	Name string

	// Label is a human-readable display name
	Label string

	// Capabilities describes what features this model supports
	Capabilities ModelCapabilities

	// Provider is the name of the provider that offers this model
	Provider string
}
