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

- **Anthropic API, OpenAI API, Gemini API, Vertex, Bedrock:** their host
  pages (`hosts/anthropic.md`, `hosts/openai.md`, `hosts/google.md`,
  `hosts/vertex.md`, `hosts/bedrock.md`).
- **Aggregators:** see `hosts/shared.md` ("Hand runs: LiteLLM leads").

## Per-entry checklist

- **Rates**: base input, output and cache rates against the host's table,
  and every override card (fast, speed, tier, region) and bracket
  (price-follow). `hosts/shared.md` ("Writing prices") says how to encode
  them. Watch for intro-pricing notes ("introductory through …"); vendors
  sometimes make intro prices permanent.
- **Limits**: `MaxInputTokens` and `MaxOutputTokens` (limit-follow).
- **Lifecycle**: dates and `ReplacedBy` follow RULES.md
  (lifecycle-exact-date, replacement-pick, out-of-reach).
  Retirement is append-only: keep the entry, and add a `// Deprecated:` doc
  comment on the ID constant.

  Lifecycle is the field most likely to be silently stale, because nothing
  in the repo goes red when a vendor announces a date. Work the whole
  deprecation page, not just the models you suspect. Each host page says
  which of its date wordings is exact and which is a floor
  (lifecycle-exact-date).
  - **Take the whole row, and check which batch it belongs to.** Each host
    page says how its deprecation tables are laid out and which IDs a row
    counts for (`hosts/openai.md` for OpenAI's dated sections and snapshots).
  - **Re-check `ReplacedBy` on every pass, not just new deprecations.**
    Vendors revise recommendations as newer models ship (the gpt-5 snapshots
    moved from the 5.4/5.5 line to 5.6). A stale target is invisible until a
    user follows it.
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
  by vendor or a uniform provider default. Check OpenAI Responses-only
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
`hosts/bedrock.md` and the Gemini API's models list in `hosts/google.md`.

## Finish

```bash
task catalog:snapshot   # regenerate; review the JSON diff line by line
task test:unit          # validation + pricing guards
task catalog:check      # same check TestCommittedSnapshotIsFresh runs under task test:unit
```

Every rate/date changed should carry a comment naming the provider source
it came from.
