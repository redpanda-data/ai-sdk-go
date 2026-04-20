# RFC 0001: Normalized `llm.TokenUsage`

**Status:** Accepted (PR 1 of 2)
**Authors:** @mms
**Date:** 2026-04-20

## Summary

Replace the current `llm.TokenUsage` with a shape that can losslessly carry what all four first-party providers (OpenAI, Anthropic, Google, AWS Bedrock) actually report, so that consumers — agents, gateways, billing/FinOps tools, dashboards — can price, audit, and sum usage without provider-specific workarounds.

## Motivation

The current struct has three concrete problems:

1. **Inconsistent semantics across providers.** `CachedTokens` is documented as a *subset* of `InputTokens`. That matches OpenAI, but:
   - Anthropic returns `input_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens` as **disjoint** buckets.
   - Google's `cached_content_token_count` is a *subset* of `prompt_token_count`.
   - Bedrock Converse reports `CacheReadInputTokens` and `CacheWriteInputTokens` as **disjoint** from `InputTokens`.

   The SDK's extractors silently fold these into the lossy subset model, losing the cache-write signal entirely (Anthropic / Bedrock) and making per-provider math diverge.

2. **Missing dimensions.** Providers already bill on fields the SDK does not surface:
   - Anthropic `cache_creation.ephemeral_5m_input_tokens` vs `ephemeral_1h_input_tokens` (two different write rates).
   - Anthropic `service_tier` (standard / priority / batch), `speed` (standard / fast), `inference_geo`.
   - Anthropic `server_tool_use.web_search_requests` / `web_fetch_requests` (billed per request).
   - OpenAI `accepted_prediction_tokens` / `rejected_prediction_tokens` (rejected is billed as output but never returned to the caller).
   - OpenAI `prompt_tokens_details.audio_tokens` / `completion_tokens_details.audio_tokens` (different rate).
   - OpenAI `service_tier` (default / flex / priority / scale).
   - Google `thoughts_token_count` (additive to total, separate from `candidates_token_count`).
   - Google `tool_use_prompt_token_count`.
   - Google `PromptTokensDetails[]` modality breakdown (text / image / audio / video / document).
   - Google `TrafficType` (ON_DEMAND / PRIORITY / FLEX / PROVISIONED_THROUGHPUT).
   - Bedrock `CacheDetails` per-TTL breakdown, `ServiceTier`, `PerformanceConfig.Latency`, `PromptRouterTrace.InvokedModelId`, `GuardrailUsage` six-counter billing surface.

3. **Reasoning-token semantics differ.** OpenAI reports `reasoning_tokens` as a *subset* of `completion_tokens`. Google reports `thoughts_token_count` as **additive** to `candidates_token_count`. The current struct inherits OpenAI's convention, which forces Google's Gemini 2.5+ counts to be mis-reported.

## Design

### Core invariant

**All token counters are disjoint.** A prompt token is counted in exactly one of:
`InputTokens`, `CachedInputTokens`, `CacheCreation5mTokens`, `CacheCreation1hTokens`, `ToolUseInputTokens`.

Same rule on the output side: `OutputTokens`, `ReasoningTokens`, `RejectedPredictionTokens` (plus `AcceptedPredictionTokens` when OpenAI Predicted Outputs is in use — see field doc).

This makes "billed tokens" a simple integer sum and removes the subset/additive ambiguity. It matches Anthropic's and Bedrock's native convention and requires mechanical remapping for OpenAI and Google.

### Field layout

```go
type TokenUsage struct {
    // Input side (all disjoint).
    InputTokens            int
    CachedInputTokens      int
    CacheCreation5mTokens  int
    CacheCreation1hTokens  int
    ToolUseInputTokens     int

    // Output side (all disjoint).
    OutputTokens             int
    ReasoningTokens          int
    AcceptedPredictionTokens int
    RejectedPredictionTokens int

    // Per-modality breakdown (optional; keys sum to the top-level counter).
    ModalityInputTokens       map[Modality]int
    ModalityOutputTokens      map[Modality]int
    ModalityCachedInputTokens map[Modality]int

    // Request posture / routing (informational, affects pricing).
    ServiceTier     ServiceTier
    RawServiceTier  string
    Speed           string
    InferenceRegion string
    InvokedModelID  string

    // Non-token billable counters.
    ServerToolRequests map[ServerTool]int
    GuardrailUnits     map[string]int

    // Performance (not billable but widely useful).
    LatencyMs          int64
    FirstByteLatencyMs int64

    // Model capability (config, not usage).
    MaxInputTokens int

    // Provider-specific dimensions we do not normalize.
    // Keys must be namespaced as "<provider>.<field>" to avoid collisions.
    Extra map[string]any
}
```

Helpers:

```go
func (u *TokenUsage) BilledInputTokens() int
func (u *TokenUsage) BilledOutputTokens() int
func (u *TokenUsage) TotalBilledTokens() int
```

### What was removed

- **`TotalTokens`**: providers disagree on its meaning (some include reasoning, some don't; some include cache writes, some don't). Callers use `TotalBilledTokens()`.
- **`CachedTokens`** → renamed to `CachedInputTokens` with new disjoint semantics. The rename forces consumers to notice the semantic change at compile time rather than silently miscount.

### Per-provider mapping contract

This is the contract every response mapper must satisfy. Full rationale per field is in the provider's `response_mapper.go`.

**Anthropic / Bedrock-Anthropic (already disjoint — direct copy):**

```
InputTokens            ← usage.input_tokens
CachedInputTokens      ← usage.cache_read_input_tokens
CacheCreation5mTokens  ← usage.cache_creation.ephemeral_5m_input_tokens
CacheCreation1hTokens  ← usage.cache_creation.ephemeral_1h_input_tokens
OutputTokens           ← usage.output_tokens
```

**OpenAI (normalize subset → disjoint):**

```
InputTokens      ← prompt_tokens - prompt_tokens_details.cached_tokens
CachedInputTokens ← prompt_tokens_details.cached_tokens
OutputTokens     ← completion_tokens - completion_tokens_details.reasoning_tokens
ReasoningTokens  ← completion_tokens_details.reasoning_tokens
```

**Google (normalize subset/additive → disjoint):**

```
InputTokens       ← prompt_token_count - cached_content_token_count
CachedInputTokens ← cached_content_token_count
ToolUseInputTokens ← tool_use_prompt_token_count
OutputTokens      ← candidates_token_count  // already excludes thoughts
ReasoningTokens   ← thoughts_token_count    // already additive
```

**Bedrock Converse:**

```
InputTokens            ← input_tokens
CachedInputTokens      ← cache_read_input_tokens
CacheCreation5mTokens  ← cache_details[ttl="5m"].input_tokens
CacheCreation1hTokens  ← cache_details[ttl="1h"].input_tokens
OutputTokens           ← output_tokens
```

## Breaking change posture

This is a v0 breaking change. The SDK is pre-1.0; consumers who care about stable semantics pin a version. Release notes + migration table ship with the PR.

## Non-goals (this RFC)

- **Iteration breakdown** (Anthropic Beta per-iteration array). Parked in `Extra["anthropic.iterations"]` for now; first-class field if there's demand.
- **Pricing/cost calculation.** Separate RFC. This RFC is strictly about the usage shape that pricing later consumes.
- **Latency histograms / per-phase timings** beyond what providers already return.

## Scope split

- **PR 1 (this):** new shape + `SumUsage` + helpers + conformance contract + minimal extractor updates (rename + disjoint math for fields already extracted today). Everything compiles, all existing tests pass with updated assertions.
- **PR 2 (follow-up):** richer extraction — populate `CacheCreation5m/1hTokens` separately (Anthropic / Bedrock), `Modality*Tokens` (Google, OpenAI audio), `GuardrailUnits` (Bedrock), `AcceptedPredictionTokens` / `RejectedPredictionTokens` (OpenAI), `ServiceTier` / `Speed` / `InferenceRegion`, `ServerToolRequests`.
