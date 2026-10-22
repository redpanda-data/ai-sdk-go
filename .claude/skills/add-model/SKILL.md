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
Adding a model never edits an existing entry (RULES.md principle 6,
*Append-only*).

Every run, hand runs included, follows `.github/reconcile-models/RULES.md`,
which decides what to change, skip or ask. The host pages in
`.claude/skills/reconcile-models/hosts/` say where each host publishes its
facts: start with `shared.md`, then the page for each package. This skill is
the workflow.

## Step 1 — Register facts (once per logical model)

In `catalog/facts_data.go`: canonical `ModelID` constant
(`"anthropic/claude-opus-5"` — vendor segment names the creator, not the
host), `Publisher`, `DisplayName`, `Description` (a short blurb from the
vendor's own pages),
`Series`, `Released`, `Knowledge`. Two providers offering the same model reference the same
`ModelID`; never author facts twice.

- **Publisher is the vendor that published the model**, independent of any
  provider serving it. Use the `catalog.Publisher*` const rather than a
  string literal; a vendor with no const yet adds one to
  `catalog/publisher.go` first. Consumers read a model's brand from it, so
  `catalog.New` rejects facts that leave it empty, and
  `TestEveryFactsRecordDeclaresItsPublisher` fails when it disagrees with
  the `ModelID`'s own vendor segment.
- **Series is a non-branching succession line**, not a brand: gpt, gpt-mini,
  and gpt-nano are three series — a mini is not the successor of a base
  model. Vendor renames map onto the line they succeed.
- **Knowledge** follows knowledge-month-end.
- **Released** follows released-first-callable, and the provider's API
  launch day goes on that offering's `Lifecycle.Available`
  (available-host-launch). GPT-5.5: Released Apr 23 in ChatGPT and Codex,
  Available Apr 24 in the API. Usually Released is the announcement day;
  when an announcement promises a later rollout, use the rollout day. A
  preview shipped under its own ID (`gemini-3.1-flash-lite-preview`) is a
  separate model with its own date, so the GA ID's Released is its own
  first-callable day, not the preview's. When the vendor gives only a
  month, follow release-month-only. Comment the source on every Released.
- Dates are `catalog.MustDate("YYYY-MM-DD")` (midnight UTC; validated).

## Step 2 — Author the provider entry

Exported ID constant (greppable), capabilities, constraints, modalities,
`WireAPIs`, `Reasoning` (efforts/adaptive/budget/thinking prefix check),
`Pricing`, `Life`.

- **`Reasoning.ThinkingPrefixCheck`** marks a model whose provider checks
  replayed thinking against an unchanged prefix (Anthropic's preserved
  thinking: platform.claude.com/docs/en/build-with-claude/preserved-thinking,
  "Keeping the prefix unchanged"). The Anthropic request mapper then sends
  `block_binding: drop_block`. Set it only on a surface whose docs list the
  model, and only with adaptive thinking (`TestThinkingPrefixCheckComesFromTheCatalog`).
- **An entry declares no publisher** — it comes from the model's facts, so
  read it back with `o.Facts().Publisher`.
- A new provider package must add itself to two hand-kept lists: the
  `snapshot.Encode` call in `cmd/catalog-snapshot/main.go` (what the
  generator writes) and `allCatalogs` in
  `cmd/catalog-snapshot/lifecycle_test.go` (what the invariants walk).
  Nothing links the two, and adding only the second makes
  `TestCommittedSnapshotIsFresh` fail with advice that cannot fix it.

- **Wire APIs belong to each offering, not shared model facts.** Author
  `WireAPIs` as `catalog.WireAPI{Protocol, Transport}` pairs verified for that
  exact provider-native ID. A protocol does not identify its transport:
  Anthropic Messages through Bedrock InvokeModel needs a different envelope
  from direct Messages, runtime Messages, Mantle, or Vertex rawPredict.
  Verify per-model API support from vendor model pages and API matrices; do
  not infer it from the publisher, model-name patterns, the SDK's current
  mapper, or another offering's geo or profile variant. OpenAI Pro models can
  be Responses-only while other OpenAI models support both APIs. Advertise
  production-grade bindings, not degraded or experimental compatibility
  layers. Availability does not promise feature parity or account
  entitlement. An empty list means unknown, not a provider default. The list
  is a set, not a preference order; catalog construction sorts and copies it.
  No endpoint URLs, credentials, regions, or model-ID remapping belong in
  this field.
- **Capabilities and modalities** follow capability-follow.
- **`Constraints.MaxInputTokens` and `MaxOutputTokens`** follow
  limit-follow. OpenAI's GPT-5 caps input at 272K within its 400K window,
  so its `MaxInputTokens` is 272K. Anthropic's 1M models have no separate
  input cap, so theirs is the window.

### Pricing encoding

- Encode prices as `reconcile-models/hosts/shared.md` ("Writing prices")
  says. Which rates to write follows price-follow.

### Lifecycle

- **Dates** follow lifecycle-exact-date. `Life.Retires` is inclusive: the
  model is retired *on* that date.
- **Lifecycle is a host fact** (RULES.md principle 2, *Host facts*): the
  Gemini API and Vertex schedule Gemini differently.
- **`Life.ReplacedBy`** follows replacement-host-pick or
  replacement-publisher-pick. When the catalog doesn't carry the
  replacement, follow replacement-not-carried.
- **A retired model keeps its entry** (RULES.md principle 6,
  *Append-only*); `reconcile-models/hosts/shared.md` ("Reading lifecycle
  pages") says how to mark it.

## Step 3 — Snapshot

`task catalog:snapshot` regenerates the committed `catalog/snapshot.json`
(a stale file fails `TestCommittedSnapshotIsFresh` under `task test:unit`; `task catalog:check`
is the CLI form). **The snapshot diff is
the review surface** — confirm it shows exactly what you meant to change and
nothing else.

## Sources

- The reconcile-models skill's Sources section lists each host's pages, and
  the host pages in `.claude/skills/reconcile-models/hosts/` say where on
  them each fact lives.
- Aggregators: see `reconcile-models/hosts/shared.md` ("Hand runs: LiteLLM
  leads").

## Bedrock specifics

Bedrock models are one `family` declaration in `models.go`, expanded by
`families.go` into bare + profile entries (`us.`, `eu.`, `global.`, …).

- **A family declares no publisher.** It comes from the model's facts, so
  the bare, geo-prefixed and global variants all report the same vendor;
  `TestPublisherMatchesBareIDVendor` pins that vendor against the namespace
  of `BareID` — `anthropic` for `anthropic.claude-opus-5`.
- Take exact IDs from the model card (`reconcile-models/hosts/bedrock.md`,
  "IDs and copies"). Register the bare ID (`BareInvokable`) only when the
  card gives it an endpoint this SDK can call on demand
  (bare-id-needs-endpoint); that host page says which endpoints count.
- `reconcile-models/hosts/bedrock.md` says how to confirm an uncertain ID
  ("Probes"), where each rate comes from and which rate card it fills
  ("Prices"), and how to check the geo-to-global ratio ("Self-check").
- A profile in a region the catalogue doesn't offer yet (a new prefix,
  e.g. `in.`) is a product decision (`region-not-offered` in
  `.github/reconcile-models/RULES.md`). Don't add one unless a person asked
  for it. Offering one takes every place below, in one PR:
  - `profileLabels` in `families.go` (`expandFamilies` panics on a profile
    that isn't there) and `geoProfilePrefixes` in `models.go`;
    `TestModelsDiscovery_InferenceGeoMetadata` fails until both have it;
  - `sourceRegionGeoPrefix` in `cross_region.go`, for the source regions
    the new geography covers, and each `ProfileRegions` map that should
    route those regions to it;
  - the family's `Profiles`, and the prefix list passed to `hasGeoPrefix`
    in `TestIsModelAllowedFromRegion_AllSupportedModels`
    (`cross_region_test.go`);
  - the *Offered regions* row in `.github/reconcile-models/RULES.md`;
    `TestOfferedRegionsMatchCode` fails until it matches the catalogue.
- Declare `ProfileRegions` when the model card publishes an exact
  source-region→profile map; add lookup and region-allow tests for every
  published ID.
- Author the family's wire API bindings and explicit per-profile exceptions
  where documented. A bare Mantle ID does not imply a runtime binding, and a
  registered runtime profile does not imply its ID works on Mantle. This
  SDK's Mantle adapter implementing only the Responses surface is an SDK
  limit, not the offering's wire-API inventory. Do not synthesize metadata
  for unpublished siblings, and keep the SDK's existing runtime routing
  unchanged when updating this metadata.
- Invocation also requires the **account** to have accepted the model's AWS
  Marketplace agreement in every region the profile routes across. Until
  then invocation returns `AccessDeniedException`; the conformance suite's
  access gate skips such models so CI stays green.

## Verify

- `task test:unit` — catalog validation, pricing guards
  (`TestGeoGlobalRatio`, fast-mode guards), and conformance metadata checks
  pick the entry up automatically through `Catalog()`.
- In a hand run, don't trust the published window: probe it as
  `reconcile-models/hosts/shared.md` ("Probes (hand runs only)") says, and
  run the probes the host's own page lists.
