---
name: model-catalog
description: >-
  How to source and encode model pricing in this repo's provider catalogs
  (providers/*/models.go) — where to pull authoritative rates, the microcents
  convention, flat vs context-tiered pricing (TieredInfo/Bracket), Anthropic's
  >200K long-context surcharge, cache-rate multipliers, and how to empirically
  verify a model's real context window. Use when adding or correcting Pricing
  on a catalog entry, or when a large-context request looks mis-costed. Pairs
  with the model-maintenance skill and the "Adding New Models" rules in
  CLAUDE.md.
---

# Model catalog pricing

Every entry in a provider's `supportedModels` map carries a `Pricing pricing.Info`.
Getting the numbers right — and the *shape* right (flat vs context-tiered) — is
the point of this skill. Read the "Adding New Models" section in `CLAUDE.md` for
the microcents rule; this skill covers where the rates come from and how to encode
tiers.

## Units: microcents per million tokens

Rates are stored as `int64` microcents per million tokens.

    dollars_per_million * 100_000_000 = microcents_per_million     # $2.50/M -> 250_000_000

The dollar-valued constructors do this for you: `pricing.NewRates(inUSD, outUSD,
cachedUSD)` and `.WithCacheCreation(w5mUSD, w1hUSD, unknownUSD)` take USD/M
directly (see `pricing/rates.go`). Prefer them over hand-writing microcents, and
keep a dollar comment on any non-obvious value.

## Where to pull the rates

1. **Provider's own pricing page is authoritative.** Always reconcile against it:
   - Anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing
   - OpenAI: https://openai.com/api/pricing/
   - Google: https://ai.google.dev/gemini-api/docs/pricing
   - Bedrock: https://aws.amazon.com/bedrock/pricing/
2. **LiteLLM's aggregated metadata** is the fastest cross-check and the only place
   that spells out per-tier and per-region rates in one machine-readable file.
   It is the source to grep for `*_above_200k_tokens`, cache-write, and regional
   multipliers:

   ```bash
   curl -sSL -o /tmp/litellm.json \
     https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json

   # Dump the fields for one model (USD/token — multiply by 1e6 for USD/M)
   python3 - <<'PY'
   import json
   d = json.load(open('/tmp/litellm.json'))
   v = d['claude-sonnet-4-5']
   for k, val in v.items():
       if 'cost' in k and isinstance(val, (int, float)):
           print(f'{k:55} {val*1_000_000:>10.4f} $/M')
   PY
   ```

   LiteLLM values are USD **per token**; multiply by 1_000_000 for USD/M before
   feeding the dollar constructors. LiteLLM lags new releases — if this repo's
   catalog is ahead of it (forward-looking model names), derive from the
   provider page + the documented multiplier structure below, and say so in a
   comment.

   License: LiteLLM is MIT (Copyright Berri AI), so referencing or even
   vendoring the file is fine with attribution. In practice we vendor nothing —
   we read it as a cross-check, and the rates themselves are facts, not
   copyrightable expression.

## Flat vs context-tiered pricing

- **Flat** (rate independent of request size): `pricing.FlatInfoFromRates(base)`.
- **Context-tiered** (rate steps up past a context-size threshold):
  `pricing.TieredInfo(base, pricing.Bracket{MinContextTokens: N, Rates: ...})`.

A `Bracket`'s `Rates` **fully replace** the base above `MinContextTokens` — they
do not overlay — so a bracket must specify *every* rate field, not just the ones
that change. The request's context size (`CalcRequest.ContextTokens`, falling
back to `usage.BilledInputTokens()`) selects the bracket at cost time; the applied
threshold is reported back as `Cost.AppliedBracketMinContextTokens`.

Speed/region variants are `Info.WithOverride(selector, RateCard{...})`. An
override `RateCard` has its **own** `Brackets` — if a model is context-tiered and
has a fast mode, the fast override needs its own bracket too, or large fast-mode
requests under-report.

## Anthropic long-context surcharge (>200K)

Anthropic bills 1M-window models at a higher rate once a request's context
exceeds the 200K standard window. The published multiplier, confirmed across
`claude-sonnet-4/4.5/4.6` on the provider page and LiteLLM's
`*_above_200k_tokens` fields, is uniform:

    input                2x base
    output               1.5x base
    cache read           2x base
    cache write (5m/1h)  2x base

Threshold is `MinContextTokens: 200_001` (matches Google's 200K and OpenAI's
272K tiers). Apply it to **every 1M-window Anthropic model**, on both the default
and any fast-mode override card. 200K-window models stay flat — no bracket. The
`TestLongContextBracketsMatchSurcharge` / `TestNonLongContextModelsStayFlat`
guards in `providers/anthropic/pricing_longcontext_test.go` enforce both
directions, so the arithmetic is checked for you.

Cache rates themselves derive from Anthropic's prompt-caching multipliers off
base input: 5m-write 1.25x, 1h-write 2x, cache-read 0.10x. These compose with the
long-context 2x, which is why the bracket's cache columns are exactly 2x the
base cache columns.

## Verify the real context window empirically

The catalog's `MaxInputTokens` and the beta headers a provider *accepts* are not
the same as what an account can actually use. Do not trust the number — probe it.
Send a prompt sized just over the threshold with the raw provider SDK and observe:

- A model whose window is genuinely 1M accepts a >200K request with **no** beta
  header. On current Anthropic 1M models (Sonnet 5, Opus 4.6/4.7/4.8, Fable 5)
  the `context-1m-2025-08-07` beta header is a **no-op** — the window is native,
  so do not add per-request beta plumbing for it.
- A 200K model returns HTTP 400 `prompt is too long: N tokens > 200000 maximum`,
  and on an account without a 1M grant the beta header does **not** lift that cap.
- Reported `Usage.InputTokens` excludes cache-write tokens (they land in
  `CacheCreation*`). Disable caching when measuring raw prompt size, or sum
  `BilledInputTokens()`.

Roughly 4 characters per token for English text; size the probe by bytes and
overshoot the threshold with margin, since tokenizers differ per model.

## Checklist for a pricing change

- [ ] Base rates match the provider page (cross-checked against LiteLLM).
- [ ] 1M-window models are `TieredInfo` with a `200_001` bracket; 200K models are flat.
- [ ] Every override card (fast/region) that needs a bracket has one.
- [ ] `task test:unit` passes — `TestAllModelsHavePricing`, the long-context guards,
      and per-provider ratio tests catch omissions and arithmetic slips.
