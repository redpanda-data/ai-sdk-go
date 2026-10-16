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

Every run, hand runs included, follows `.github/reconcile-models/RULES.md`,
which decides what to change, skip or ask. The host pages in
`.claude/skills/reconcile-models/hosts/` say where each host publishes its
facts: start with `shared.md`, then the page for each package. This skill is
the workflow.

## Sources

Facts come only from the hosts' own pages, and code cites only those.

- **Anthropic API, Claude on Vertex, Bedrock:** their host pages
  (`hosts/anthropic.md`, `hosts/vertex.md`, `hosts/bedrock.md`).
- **OpenAI** (`providers/openai`):
  - Pricing: `https://developers.openai.com/api/docs/pricing`
  - **Lifecycle** (the only source for deprecation and shutdown dates):
    `https://developers.openai.com/api/docs/deprecations`
  - Per-model specs: `https://developers.openai.com/api/docs/models/<id>`
    gives window, **maximum input tokens**, max output, per-tier pricing notes

  Append `.md` to any of these to get clean markdown
  (`.../deprecations.md`). Do this for the deprecations page especially:
  it is long, and summarizing it lossily has produced flatly
  contradictory answers about whether `o3`/`o3-pro` are deprecated at
  all. Fetch the `.md`, then grep it: the tables are the data.
- **Google** (Gemini API, `providers/google`):
  - Pricing: `https://ai.google.dev/gemini-api/docs/pricing`
  - Per-model limits: `https://ai.google.dev/gemini-api/docs/models`
  - **Lifecycle**: `https://ai.google.dev/gemini-api/docs/deprecations`
    (fetch `.../deprecations.md.txt`) plus the changelog
    (`.../docs/changelog`). `ReplacedBy` comes from the replacement column
    or a changelog recommendation.

  **Never copy lifecycle between Google surfaces.** Vertex retired the
  Gemini 2.5 models on 2026-10-20; the Gemini API still serves them.
- **Gemini on Vertex** (`providers/vertex`; `hosts/vertex.md` covers only
  its Claude models):
  - Pricing: the Gemini sections of
    `https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing`,
    one tab per region. The self-check in `hosts/vertex.md` applies to
    Gemini too.
  - Lifecycle (release, retirement and recommended replacement in one
    table): `https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions`.
  - Per-model pages carry launch stage and release date:
    `https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/<slug>`
    (e.g. `3-6-flash`).
- **Aggregators:** see `hosts/shared.md` ("Hand runs: LiteLLM leads").

## Per-entry checklist

- **Rates**: base input, output and cache rates against the host's table,
  and every override card (fast, speed, tier, region) and bracket
  (price-follow). `hosts/shared.md` ("Writing prices") says how to encode
  them. Watch for intro-pricing notes ("introductory through …"); vendors
  sometimes make intro prices permanent.
- **Limits**: `MaxInputTokens` and `MaxOutputTokens` (limit-follow).
- **Lifecycle**: dates follow lifecycle-exact-date, and `ReplacedBy`
  follows replacement-host-pick or replacement-publisher-pick. When the
  catalog doesn't carry the replacement, follow replacement-not-carried.
  Retirement is append-only: keep the entry, and add a `// Deprecated:` doc
  comment on the ID constant.

  Lifecycle is the field most likely to be silently stale, because nothing
  in the repo goes red when a vendor announces a date. Work the whole
  deprecation page, not just the models you suspect. The host pages say
  how Anthropic, Vertex's Claude pages and Bedrock word their dates; for
  the rest:
  - **Read each vendor's date wording before copying it**, and note that
    each vendor words it differently:
    - OpenAI: announces exact shutdown dates → `Retires`.
    - Gemini API: every date is a floor → unset until Google announces an
      exact date (lifecycle-exact-date).
    - Gemini on Vertex (`model-versions`): a bare date is firm →
      `Retires`; "or later" is a floor → unset.
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
  - **Probe when a date has passed (hand runs only; it needs a key).** For
    Gemini, the models list shows what still exists:
    ```bash
    curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY&pageSize=200" \
      | jq -r '.models[].name' | sed 's|^models/||' | sort
    ```
    A model absent here no longer serves (this is how `gemini-3-pro-preview`
    was confirmed retired). `Retires` still comes only from a page's exact
    date (lifecycle-exact-date); tell the person what the list shows.
  - Two guards in `cmd/catalog-snapshot/lifecycle_test.go` catch the
    mechanical half (deprecated-without-`ReplacedBy`, and `ReplacedBy`
    pointing at an already-retired offering). They cannot catch a wrong or
    missing date — that still needs the vendor page.
- **Facts**: `Knowledge` follows knowledge-month-end, `Released` follows
  released-first-callable, and an offering's launch `Available` follows
  available-host-launch. A month-only release date follows
  release-month-only. A preview under its own ID is a separate model; see
  add-model. An existing `Description` stays as it is.

  **Fixed facts vs. mutable data.** `Released`, `Knowledge`, and an
  offering's launch `Available` describe a past event: once correct they
  never change. Everything else (rates, brackets, limits, lifecycle,
  `ReplacedBy`, capabilities) legitimately moves with the vendor; update it
  and cite the source.

  So a fixed fact that disagrees with its page follows label-fact-unproven
  or label-fact-differs, which turn on its source comment. A newer model in
  the same series is a new entry, never a reason to edit the old one's
  dates. In a hand run, also find where the current value came from before
  you change it, and tell the person what you found:
  `git log -S'MustDate("YYYY-MM-DD")' -- catalog/ providers/`.
- **Wire APIs**: verify each offering's `WireAPIs` protocol/transport pairs
  against vendor API-compatibility matrices and the exact model/profile's
  programmatic-access page. Review API support separately from the SDK's
  selected transport: InvokeModel, direct runtime Messages, Mantle, and Vertex
  envelopes are not interchangeable. Reconcile exceptions per offering, not
  by publisher or a uniform provider default. Check OpenAI Responses-only
  models and Bedrock bare-vs-profile IDs explicitly. Do not advertise
  degraded/experimental compatibility layers as native equivalents or infer
  account access, feature parity, new IDs, or runtime routing from this list.
  Empty metadata is unknown, never permission to guess a provider default.
  Regenerate the snapshot; unrelated prices, lifecycle, and capability fields
  must stay unchanged during a wire-only reconciliation.
- **Capabilities, modalities and reasoning** (thinking mode, effort
  levels): provider-documented model facts (capability-follow). JSON mode means a schemaless mode (Anthropic has
  none); document input means the API accepts document parts.

## When the docs are ambiguous, probe (hand runs only)

`hosts/shared.md` ("Probes (hand runs only)") has the context-window probe,
and each host page lists its own, such as Bedrock's `Converse` ID check in
`hosts/bedrock.md`.

## Finish

```bash
task catalog:snapshot   # regenerate; review the JSON diff line by line
task test:unit          # validation + pricing guards
task catalog:check      # same check TestCommittedSnapshotIsFresh runs under task test:unit
```

Every rate/date changed should carry a comment naming the provider source
it came from.
