---
name: reconcile-models
description: >-
  Audit the provider catalogs (providers/*/models.go, catalog/facts_data.go)
  against vendor sources and fix drift: pricing, context/input limits,
  lifecycle dates (deprecations/retirements), knowledge cutoffs,
  capabilities/modalities. Use for periodic sync, when a vendor announces
  price or lifecycle changes, or when a cost/limit looks wrong. For adding
  a model, use add-model instead.
---

# Reconcile models against vendor sources

The catalog is authored data; vendors change prices, retire models, and
correct their own docs. Reconciliation = fetch current sources, diff against
our entries, fix, regenerate the snapshot. **The snapshot diff is the review
surface** — every change must be visible and intended in
`catalog/snapshot.json`.

## Sources, in order of authority

1. **Provider pages (authoritative — the only sources cited in code):**
   - Anthropic:
     - Pricing (includes the fast-mode, long-context, cache and batch
       sections): `https://platform.claude.com/docs/en/about-claude/pricing`
     - Lifecycle: `https://platform.claude.com/docs/en/about-claude/model-deprecations`
     - Per-model specs: `https://platform.claude.com/docs/en/models/<slug>/overview`
       (e.g. `sonnet-4-6`) — window, max output, both knowledge cutoffs
   - OpenAI:
     - Pricing: `https://developers.openai.com/api/docs/pricing`
     - **Lifecycle** (the only source for deprecation and shutdown dates):
       `https://developers.openai.com/api/docs/deprecations`
     - Per-model specs: `https://developers.openai.com/api/docs/models/<id>`
       — window, **maximum input tokens**, max output, per-tier pricing notes

     Append `.md` to any of these to get clean markdown
     (`.../deprecations.md`). Do this for the deprecations page especially:
     it is long, and summarizing it lossily has produced flatly
     contradictory answers about whether `o3`/`o3-pro` are deprecated at
     all. Fetch the `.md`, then grep it — the tables are the data.
   - Google — pricing on one site, lifecycle on another. Both are needed:
     - Pricing: `https://ai.google.dev/gemini-api/docs/pricing`
     - **GA lifecycle** (release + retirement + recommended replacement,
       all GA Gemini models in one table — the authoritative lifecycle
       source):
       `https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions`
     - Preview lifecycle: `https://ai.google.dev/gemini-api/docs/deprecations`

     The two lifecycle pages disagree and each is incomplete, so check
     both: `ai.google.dev` reports "No shutdown date announced" for the
     Gemini 2.5 models that `model-versions` retires on 2026-10-20, while
     `model-versions` omits previews entirely. Google marks a floor by
     appending "or later" ("May 19, 2027 or later") and a firm date as a
     bare date — so a bare date goes in `Retires`, an "or later" date does
     not. Per-model pages
     (`.../models/gemini/<slug>`, e.g. `2-5-pro`, `3-1-pro`) carry release
     and retirement but NOT the replacement column, so prefer
     `model-versions`.
   - Bedrock: `https://aws.amazon.com/bedrock/pricing/` · model cards ·
     `ListFoundationModels` (`modelLifecycle`) · lifecycle/EOL table:
     `https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html`
2. **Cross-checks (fetch and compare only — never cite in code):**

   ```bash
   curl -s https://openrouter.ai/api/v1/models -o /tmp/openrouter.json
   jq '.data[] | select(.id=="openai/gpt-5.4") | {pricing, context_length, top_provider}' /tmp/openrouter.json

   curl -sL https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json -o /tmp/litellm.json
   jq '."gpt-5.4"' /tmp/litellm.json   # values are USD PER TOKEN — ×1e6 for $/M
   ```

   A disagreement between an aggregator and our entry is a *lead*, not a
   verdict — resolve it on the provider page. Known aggregator traps:
   LiteLLM prices in USD/token and carries no release dates; OpenRouter
   truncates descriptions mid-sentence, lags new models, and its `created`
   (a good day-precision Released proxy otherwise) tracks slug listing —
   wrong for models that re-slugged preview→GA.

   **Neither is usable for lifecycle.** OpenRouter has an
   `expiration_date` field that is `null` for every model, retired ones
   included; a model silently *vanishing* from its list is the only
   lifecycle signal it gives. LiteLLM's single `deprecation_date` conflates
   three meanings, measured over our catalog: the exact shutdown date for
   OpenAI models and Gemini previews (correct wherever present — it is what
   catches a mis-filed batch like `o4-mini`); Anthropic's "not sooner than"
   *floor* for all 10 active Claude models (trusting it would falsely
   retire the entire live lineup); and simply absent for `gpt-5`, `o3` and
   the Gemini 2.5 models that do have announced dates. Use it to generate
   leads, never to set a date.

## Per-entry checklist

- **Rates**: base input/output/cached against the provider table. Check
  every override card (fast/speed/tier/region) and every bracket — an
  override card needs its own brackets, or large requests on that card
  underbill. Encode only the rate in effect today; scheduled future rates
  are comments, never data. Watch for intro-pricing notes ("introductory
  through …" — and vendors sometimes make intro prices permanent).
- **Tier thresholds**: only author a `Bracket` the vendor currently
  publishes. When a vendor raises a model's window, check whether a
  long-context surcharge came with it — a bigger window without its tier
  underbills; a tier the vendor dropped overbills.
- **Limits**: `MaxInputTokens` is the enforceable input cap (vendor's
  "maximum input tokens" when published; the window only when input isn't
  separately capped). `MaxOutputTokens` from the model page.
- **Lifecycle**: `Retires` only for exact announced shutdown dates —
  "not sooner than" floors stay out. Deprecations get `Life.Deprecated` +
  `ReplacedBy` (only if the replacement is in the same catalog). Retirement
  is append-only: keep the entry, add `Retires` and a `// Deprecated:` doc
  comment on the ID constant.

  Lifecycle is the field most likely to be silently stale, because nothing
  in the repo goes red when a vendor announces a date. Work the whole
  deprecation page, not just the models you suspect:
  - **Read each vendor's date wording before copying it**, and note that
    each vendor words it differently:
    - Anthropic — "Tentative retirement date / Not sooner than X" is a
      floor → leave `Retires` unset.
    - OpenAI — announces exact shutdown dates → `Retires`.
    - Google — on `model-versions`, a bare date is firm → `Retires`, while
      "or later" is a floor → unset. On `ai.google.dev/docs/deprecations`
      the whole column is hedged as "the *earliest possible* dates" even
      where it is effectively firm, so confirm those against the live
      models list (below) rather than trusting the wording.
  - **Take the whole row, and check which batch it belongs to.** Every
    OpenAI deprecation lives under a dated `### YYYY-MM-DD:` heading — that
    heading is `Life.Deprecated`, the row's date is `Life.Retires`. Models
    are easy to mis-file: `o4-mini` reads like the o3 batch but ships in the
    earlier legacy-snapshot batch with a two-month-earlier shutdown.
  - **Match name granularity.** OpenAI rows list `snapshot | alias`. Only
    the aliases we key entries on count: `gpt-4o-2024-05-13` is deprecated
    while the `gpt-4o` alias is not, so that entry stays clean.
  - **Re-check `ReplacedBy` on every pass, not just new deprecations.**
    Vendors revise recommendations as newer models ship (the gpt-5 snapshots
    moved from the 5.4/5.5 line to 5.6). A stale target is invisible until a
    user follows it.
  - **Probe when a date has passed.** For Gemini, the models list is
    authoritative about what still exists:
    ```bash
    curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY&pageSize=200" \
      | jq -r '.models[].name' | sed 's|^models/||' | sort
    ```
    A model absent here is retired regardless of how the column is worded
    (this is how `gemini-3-pro-preview` was confirmed retired).
  - **Bedrock sets its own schedule and publishes only Legacy/EOL models**
    on the model-lifecycle page — that table omits Active models, so an
    absence there means "no EOL announced", not "unchecked". `Available` is
    genuinely unknown per inference profile without
    `ListFoundationModels`; leave it unset rather than guessing.
  - Two guards in `cmd/catalog-snapshot/lifecycle_test.go` catch the
    mechanical half (deprecated-without-`ReplacedBy`, and `ReplacedBy`
    pointing at an already-retired offering). They cannot catch a wrong or
    missing date — that still needs the vendor page.
- **Facts**: `Knowledge` is the *reliable* cutoff when the vendor publishes
  both reliable and training dates; `Released` is the first-ship date
  anywhere; `Description` is a short current blurb.
- **Capabilities/modalities**: provider-documented model facts (JSON mode
  means a schemaless mode — Anthropic has none; document input means the
  API accepts document parts). Bedrock geo ratio stays 1.10x
  (`TestGeoGlobalRatio` trips if AWS changes it).

## When the docs are ambiguous, probe

- Context window: send a prompt just over the claimed threshold with the
  raw SDK (~4 chars/token, overshoot with margin); a 200K model returns
  "prompt is too long", a genuine 1M model accepts without beta headers.
- Bedrock IDs: a one-token `Converse` call distinguishes working /
  unpublished (`ValidationException`) / unsubscribed
  (`AccessDeniedException`).

## Finish

```bash
task catalog:snapshot   # regenerate; review the JSON diff line by line
task test:unit          # validation + pricing guards
task catalog:check      # same check TestCommittedSnapshotIsFresh runs under task test:unit
```

Every rate/date changed should carry a comment naming the provider source
it came from.
