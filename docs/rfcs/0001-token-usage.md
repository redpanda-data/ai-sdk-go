# RFC 0001: Normalized `llm.TokenUsage`

**Status:** Accepted (PR 1 of 2)
**Authors:** @mms
**Date:** 2026-04-20

## Summary

Replace the current `llm.TokenUsage` with a **normalized** shape that captures the billable and operationally interesting dimensions the four first-party providers (OpenAI, Anthropic, Google, AWS Bedrock) report, so that consumers — agents, gateways, billing/FinOps tools, dashboards — can price and aggregate usage without provider-specific workarounds. This is not a lossless mirror of every provider field; dimensions the SDK does not normalize (e.g., provider-specific trace details, forthcoming billing lines) live in an `Extra` map keyed by `<provider>.<field>`.

## Motivation

The current struct has three concrete problems:

1. **Inconsistent semantics across providers.** `CachedTokens` is documented as a *subset* of `InputTokens`. That matches OpenAI, but:
   - Anthropic returns `input_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens` as **disjoint** buckets.
   - Google's `cached_content_token_count` is a *subset* of `prompt_token_count`.
   - Bedrock Converse reports `CacheReadInputTokens` and `CacheWriteInputTokens` as **disjoint** from `InputTokens`.

   The SDK's extractors silently fold these into the lossy subset model, losing the cache-write signal entirely (Anthropic / Bedrock) and making per-provider math diverge.

2. **Missing dimensions.** Providers already bill on fields the SDK does not surface:
   - Anthropic `cache_creation.ephemeral_5m_input_tokens` vs `ephemeral_1h_input_tokens` (two different write rates).
   - Anthropic `service_tier` (standard / priority / batch — each a distinct price sheet), `speed` (standard / fast — fast costs more), `inference_geo`.
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
`InputTokens`, `CachedInputTokens`, `CacheCreation5mTokens`, `CacheCreation1hTokens`, `CacheCreationUnknownTTLTokens`, `ToolUseInputTokens`.

Same rule on the output side: `OutputTokens`, `ReasoningTokens`.

This makes "billed tokens" a simple integer sum and removes the subset/additive ambiguity. It matches Anthropic's and Bedrock's native convention and requires mechanical remapping for OpenAI and Google.

### Deliberately minimal

The struct only carries counters that at least one extractor populates today. Per-modality breakdowns, server-tool request counts, guardrail-unit counts, and OpenAI predicted-output counts all exist at the provider level but aren't yet extracted — they can be added as additive first-class fields in follow-up work, or left to consumers via the raw provider response. The same principle applies to `Extra`: it exists only because Bedrock's fallback path already writes provider-specific audit keys into it.

### What lives where

`TokenUsage` contains **only** the additive token-accounting surface. Per-call metadata that affects how the request was priced — `ServiceTier`, `RawServiceTier`, `Speed`, `InferenceRegion`, `InvokedModelID` — lives on `llm.Response`. These fields are not meaningfully additive across calls, so forcing them through `SumUsage` (first-non-empty-wins, max, etc.) is a type smell.

Service tier, speed, and region are not cosmetic: every provider we support charges **different per-token rates** depending on tier. OpenAI's `flex` is ≈50% off, `priority` is ≈2×; Anthropic's `batch` is 50% off; Bedrock's `reserved` and Google's `PROVISIONED_THROUGHPUT` bypass per-token billing entirely. Anthropic `speed=fast` costs more than `standard`. `InvokedModelID` matters because Bedrock's PromptRouter can rewrite the model — pricing must be computed against the invoked model, not the requested one.

Pure observability fields (latency, TTFT) are deliberately out of scope for this RFC. They belong in a tracing/metrics layer, not on `Response`.

`MaxInputTokens` is a model-config value, not usage. It has been removed from `TokenUsage`; the canonical source is `modelDefinition.Constraints.MaxInputTokens`.

### TTL fallback for cache writes

Anthropic and Bedrock both report a per-TTL breakdown for cache-write tokens. When the breakdown is absent or covers fewer tokens than the aggregate (older API response shapes, partial CacheDetails), extractors route the remainder to `CacheCreationUnknownTTLTokens`. That way `BilledInputTokens()` always reflects the provider-reported billable amount without pretending to know a TTL we weren't told.

### Field layout

```go
type TokenUsage struct {
    // Input side (all disjoint).
    InputTokens                   int
    CachedInputTokens             int
    CacheCreation5mTokens         int
    CacheCreation1hTokens         int
    CacheCreationUnknownTTLTokens int // fallback when per-TTL breakdown is absent
    ToolUseInputTokens            int

    // Output side (all disjoint).
    OutputTokens    int
    ReasoningTokens int

    // Provider-specific audit keys the SDK does not normalize.
    // Keys MUST be namespaced as "<provider>.<field>" to avoid collisions.
    Extra map[string]any
}
```

Per-call metadata on `llm.Response` — each of these can change the per-token rate:

```go
type Response struct {
    // ...existing fields...

    ServiceTier     ServiceTier // default | flex | priority | batch | scale | reserved | provisioned_throughput
    RawServiceTier  string      // provider-native value when the normalized mapping is lossy
    Speed           string      // Anthropic: "standard" | "fast" (fast costs more)
    InferenceRegion string      // Anthropic inference_geo, Bedrock region
    InvokedModelID  string      // Bedrock PromptRouter: price against this, not the requested model
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

### PR 1 extractor coverage

| Field | Populated |
|---|---|
| `InputTokens`, `OutputTokens`, `CachedInputTokens` | All providers |
| `CacheCreation5mTokens`, `CacheCreation1hTokens` | Anthropic, Bedrock |
| `CacheCreationUnknownTTLTokens` | Anthropic, Bedrock (fallback path) |
| `ToolUseInputTokens` | Google |
| `ReasoningTokens` | OpenAI, OpenAI-compat, Google |
| `Extra` | Bedrock unknown-TTL audit keys |
| `Response.ServiceTier` / `RawServiceTier` / `Speed` / `InferenceRegion` / `InvokedModelID` | — (PR 2) |

### Per-provider mapping contract

This is the contract every response mapper must satisfy. Full rationale per field is in the provider's `response_mapper.go`.

**Anthropic (already disjoint — direct copy with TTL fallback):**

```
InputTokens                   ← usage.input_tokens
CachedInputTokens             ← usage.cache_read_input_tokens
CacheCreation5mTokens         ← usage.cache_creation.ephemeral_5m_input_tokens
CacheCreation1hTokens         ← usage.cache_creation.ephemeral_1h_input_tokens
CacheCreationUnknownTTLTokens ← max(0, usage.cache_creation_input_tokens - (5m + 1h))
OutputTokens                  ← usage.output_tokens
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

**Bedrock Converse (direct copy with TTL fallback):**

```
InputTokens                    ← input_tokens
CachedInputTokens              ← cache_read_input_tokens
CacheCreation5mTokens          ← sum(cache_details[ttl="5m"].input_tokens)
CacheCreation1hTokens          ← sum(cache_details[ttl="1h"].input_tokens)
CacheCreationUnknownTTLTokens  ← sum(cache_details[ttl=other].input_tokens)
                                 + max(0, cache_write_input_tokens - sum(cache_details))
OutputTokens                   ← output_tokens
```

Unknown TTL strings are also recorded in `Extra["bedrock.cache_write_ttl_<ttl>_tokens"]` for audit.

## Breaking change posture

This is a v0 breaking change. The SDK is pre-1.0; consumers who care about stable semantics pin a version. Release notes + migration table ship with the PR.

## Non-goals (this RFC)

- **Iteration breakdown** (Anthropic Beta per-iteration array). Parked in `Extra["anthropic.iterations"]` for now; first-class field if there's demand.
- **Pricing/cost calculation.** Separate RFC. This RFC is strictly about the usage shape that pricing later consumes.
- **Latency histograms / per-phase timings** beyond what providers already return.

## Scope split

- **PR 1 (this):** new shape + `SumUsage` + helpers + extractor updates for fields already extracted today (input/output/cached for every provider; cache-creation 5m/1h + unknown-TTL fallback for Anthropic and Bedrock; thoughts + tool-use for Google). Response gains per-call metadata fields but they are not populated yet. Everything compiles, all unit tests pass.
- **PR 2 (follow-up):** populate `Response.ServiceTier` / `RawServiceTier` / `Speed` / `InferenceRegion` / `InvokedModelID` from provider responses. Additional counters (per-modality breakdowns, guardrail units, server-tool requests, OpenAI predicted-output counts) remain candidates for future additive PRs but are intentionally out of scope here — the SDK normalizes what it actually extracts, not everything providers theoretically expose.
