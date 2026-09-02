---
name: add-model
description: >-
  Add a new LLM model to this repo's provider catalogs: register facts in
  catalog/facts_data.go, author the catalog.Entry in providers/*/models.go
  (Bedrock: a family declaration), encode pricing, and regenerate the
  snapshot. Use when registering a new model, a new model snapshot, or new
  Bedrock region/profile variants. For correcting existing entries against
  vendor sources, use reconcile-models instead.
---

# Add a new model

Each provider authors `catalog.Entry` values in its `models.go` (`entries()`),
frozen into a `catalog.Catalog` at init — `catalog.MustNew` validates every
entry, so a broken entry fails every test run with a path-qualified error.
Adding a model must never require editing an existing entry: generation
ordering, succession, and price tiers are derived at read time.

## Step 1 — Register facts (once per logical model)

In `catalog/facts_data.go`: canonical `ModelID` constant
(`"anthropic/claude-opus-5"` — vendor segment names the creator, not the
host), `DisplayName`, `Description` (short UI blurb), `Series`, `Released`,
`Knowledge`. Two providers offering the same model reference the same
`ModelID`; never author facts twice.

- **Series is a non-branching succession line**, not a brand: gpt, gpt-mini,
  and gpt-nano are three series — a mini is not the successor of a base
  model. Vendor renames map onto the line they succeed.
- **Knowledge is the reliable cutoff.** When a vendor publishes both a
  "reliable knowledge" and a broader "training data" date, use the reliable
  one. Month-only cutoffs normalize to the last day of that month.
- **Released** is the vendor's announcement date when findable. Vendors
  often publish only a month; OpenRouter's `created` field is a
  day-precision fallback (it matches launch day for models listed at
  launch) — but it tracks the *slug's* listing, so it misses the preview
  launch of models that re-slugged preview→GA. LiteLLM has no release
  dates. Comment the sourcing when using the fallback.
- Dates are `catalog.MustDate("YYYY-MM-DD")` (midnight UTC; validated).

## Step 2 — Author the provider entry

Exported ID constant (greppable), capabilities, constraints, modalities,
`Reasoning` (efforts/adaptive/budget), `Pricing`, `Life`.

- **Capabilities and modalities describe the model as the provider documents
  it**, not what this SDK's request mappers wire yet.
- **`Constraints.MaxInputTokens` is the vendor's enforceable input cap**:
  when the vendor publishes a separate "maximum input tokens" (OpenAI's
  GPT-5: 272K within its 400K window), use that — the API rejects larger
  inputs. Use the stated context window only when input is not separately
  capped (Anthropic's 1M).

### Pricing encoding

- Constructors take **US dollars per million tokens**: `$2.50/M` is written
  `2.50` (`pricing.NewRates`, `pricing.FlatInfo`,
  `.WithCacheCreation(w5m, w1h, unknown)`). They convert to microcents
  internally — never pass microcent literals; `FlatInfo(250_000_000, …)`
  compiles and overprices by 10⁸.
- A `0` input/output rate means *unpriced* and is rejected; a published $0
  rate is `pricing.RateFree`.
- Context-tiered rates use `pricing.TieredInfo(base, pricing.Bracket{...})`.
  A bracket's rates **fully replace** the base above `MinContextTokens` —
  author every field, not just the ones that change. Speed/tier/region
  variation uses `Pricing.WithOverride(selector, RateCard{...})` — never
  provider-specific fields. An override card has its **own** brackets: a
  tiered model with a fast mode needs the bracket on the fast card too, or
  large fast requests underbill.
- **Encode only the rate in effect today.** Vendor tables print future rates
  in the same cell; `pricing.Info` has no effective date, so a scheduled
  rate is a comment, never data. Only add a bracket for a threshold the
  vendor currently publishes.
- Anthropic cache rates derive from published multipliers off base input:
  5m-write 1.25x, 1h-write 2x, cache-read 0.10x.

### Lifecycle

- `Life.Retires` is an **exact announced shutdown date only** (inclusive:
  retired *on* that date). A published "not sooner than" floor is a lower
  bound, not a shutdown date — leave `Retires` unset until an exact date is
  announced.
- `Life.ReplacedBy` must name an offering in the same catalog; skip it when
  the provider's recommendation isn't one we carry.
- **The catalog is append-only**: never delete a retired model's entry — set
  `Retires`, add a `// Deprecated:` doc comment on the ID constant naming
  the replacement, and leave the entry so historical usage stays priceable.

## Step 3 — Snapshot

`task catalog:snapshot` regenerates the committed `catalog/snapshot.json`
(a stale file fails `TestCommittedSnapshotIsFresh` under `task test:unit`; `task catalog:check`
is the CLI form). **The snapshot diff is
the review surface** — confirm it shows exactly what you meant to change and
nothing else.

## Sources

Authoritative (always reconcile against these):

- Anthropic: platform.claude.com/docs/en/about-claude/pricing ·
  /model-deprecations
- OpenAI: developers.openai.com/api/docs/pricing · /docs/deprecations ·
  per-model pages (/docs/models/<id>) for window, max input, max output
- Google: ai.google.dev/gemini-api/docs/pricing · /docs/deprecations ·
  /docs/models
- Bedrock: aws.amazon.com/bedrock/pricing/ · per-model cards ·
  `ListFoundationModels` (`modelLifecycle`)

Cross-checks (fetch, compare, never cite in code comments — code references
provider sources only):

```bash
# OpenRouter: per-model objects; /endpoints lists per-host variants.
curl -s https://openrouter.ai/api/v1/models | jq '.data[] | select(.id=="anthropic/claude-opus-4.5")'
# LiteLLM: one flat JSON; values are USD PER TOKEN — multiply by 1e6 for $/M.
curl -sL https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json |
  jq '."claude-opus-4-5"'
```

Cross-check traps: LiteLLM sometimes files "not sooner than" retirement
floors under `deprecation_date` — never trust it for lifecycle; OpenRouter's
API truncates `description` mid-sentence (trim to the last full sentence);
both may lag brand-new models. When aggregators and this repo disagree, the
provider page wins.

## Bedrock specifics

Bedrock models are one `family` declaration in `models.go`, expanded by
`families.go` into bare + profile entries (`us.`, `eu.`, `global.`, …).

- Check the model card's Programmatic Access and Regional Availability
  tables for exact IDs. Register the bare ID (`BareInvokable`) only when the
  bedrock-runtime row publishes an In-Region endpoint URL — otherwise the
  bare ID returns `ValidationException` ("on-demand throughput isn't
  supported"). A bedrock-mantle Anthropic Messages URL is NOT bedrock-runtime
  support: the mantle transport implements only the OpenAI-compatible
  Responses surface.
- Confirm uncertain IDs with a one-token `Converse` request: working IDs
  return output; unpublished ones return `ValidationException` ("model
  identifier is invalid"); published-but-unsubscribed ones return
  `AccessDeniedException`. New models are often US- and global-only at first.
- Pricing is per-profile: `global.` is cheapest; every geo/in-region rate is
  exactly **1.10x** the global rate (pinned by `TestGeoGlobalRatio` as a
  tripwire — a future exception is a data edit, not a schema change).
- Declare `ProfileRegions` when the model card publishes an exact
  source-region→profile map; add lookup and region-allow tests for every
  published ID.
- Invocation also requires the **account** to have accepted the model's AWS
  Marketplace agreement in every region the profile routes across. Until
  then invocation returns `AccessDeniedException`; the conformance suite's
  access gate skips such models so CI stays green.

## Verify

- `task test:unit` — catalog validation, pricing guards
  (`TestGeoGlobalRatio`, fast-mode guards), and conformance metadata checks
  pick the entry up automatically through `Catalog()`.
- Don't trust the published window — probe it: send a prompt just over the
  threshold with the raw SDK (~4 chars/token, overshoot with margin). A
  genuine 1M model accepts >200K with no beta header; a 200K model returns
  "prompt is too long". Disable caching when measuring, or sum
  `BilledInputTokens()`.
