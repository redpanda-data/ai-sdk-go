---
name: model-maintenance
description: >-
  Guidance for adding or updating LLM models in this repo's provider catalogs
  (providers/*/models.go). Use when registering a new model, a new model
  snapshot, or new region/profile variants, especially on Bedrock. Pairs with
  the "Adding New Models" pricing rules in CLAUDE.md.
---

# Model maintenance

When you add a model to any provider's `supportedModels` map you **must** set
`Pricing` (microcents per million tokens) — see the "Adding New Models" section
in `CLAUDE.md` for the conversion and the per-provider `TestAllModelsHavePricing`
requirement. For where to source rates, flat vs context-tiered pricing, and the
long-context surcharge, see the `model-catalog` skill. Below is the extra process
that is specific to Bedrock.

## Adding a new Bedrock model

Bedrock exposes each Claude model only through **cross-region inference
profiles** (`us.`, `eu.`, `global.`, and so on), never the bare
`anthropic.<model>` id — invoking the bare id returns a `ValidationException`
("on-demand throughput isn't supported"), so register one entry per *published
profile* in `providers/bedrock/models.go` and do **not** add a bare entry.
Confirm which profiles actually exist by invoking each candidate with a one-token
`Converse` request rather than guessing: a working profile returns output, an
unpublished one returns `ValidationException` ("the provided model identifier is
invalid") even from its home region, and a published-but-unsubscribed one returns
`AccessDeniedException`. A newly released model is often US- and global-only at
first, with other geos published later. Pricing is per-profile: the `global.`
profile is the cheapest and each geo profile is exactly `1.10x` the global rate in
every column (enforced by `TestGeoGlobalRatio`) — spell every rate out in place,
and reuse the shared capability/constraint vars when the context window matches an
existing generation. Add a lookup test (the new profiles present; the bare and any
unpublished profile absent) plus a region-allow test; the integration conformance
suite then picks the new entries up automatically through `Models()`.

Separately, a catalog entry is not invokable until the **account** has accepted
the model's Bedrock agreement (an AWS Marketplace subscription) in **every region
the profile routes across** — a `us.` cross-region profile needs the agreement
enabled in all of its member regions, not just the entry region. Enable it from
the Bedrock "Model access" console page, or with
`aws bedrock create-foundation-model-agreement` (after
`list-foundation-model-agreement-offers` to get the offer token); newly released
Anthropic models are usually already authorized at the account level, so this is a
one-time acceptance with no use-case form. Until access is granted, invocation
returns `AccessDeniedException` mentioning the required `aws-marketplace` actions.
The conformance suite's `isModelAccessGate` treats that as "in the catalog but not
subscribed by this account" and **skips** the model, so CI stays green for entries
the test account hasn't enabled yet; once the account is subscribed, the same test
exercises the model for real with no code change.
