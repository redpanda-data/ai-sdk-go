---
name: model-maintenance
description: >-
  Guidance for adding or updating LLM models in this repo's provider catalogs
  (providers/*/models.go). Use when registering a new model, a new model
  snapshot, or new region/profile variants, especially on Bedrock. Pairs with
  the "Adding New Models" pricing rules in CLAUDE.md.
---

# Model maintenance

When you add a model you register its facts once in `catalog/facts_data.go`
(canonical ModelID, Series, release date) and author a `catalog.Entry` in the
provider's `entries()` — with `Pricing` in **USD per million tokens** (the
constructors convert; never microcent literals) and `Life` lifecycle dates. See
the "Adding New Models" section in `CLAUDE.md` for the full flow, and the
`model-catalog` skill for rate sourcing, flat vs context-tiered pricing, and the
long-context surcharge. Regenerate `catalog/snapshot.json` with
`task catalog:snapshot` and review the diff. Below is the extra process that is
specific to Bedrock.

## Adding a new Bedrock model

Bedrock may expose a Claude model through a bare **in-region** ID, cross-region
inference profiles (`us.`, `eu.`, `global.`, and so on), or both. Check the
model card's Programmatic Access and Regional Availability tables for the exact
IDs. Register the bare ID only when the bedrock-runtime row publishes an
In-Region endpoint URL; a bare Model ID alone is not sufficient. Otherwise,
invoking it returns `ValidationException` ("on-demand throughput isn't
supported"). Do not treat a bedrock-mantle Anthropic Messages URL as
bedrock-runtime support: this provider's mantle transport implements only the
OpenAI-compatible Responses surface. Confirm uncertain IDs with a one-token
`Converse` request rather than guessing: a working ID returns output, an
unpublished one returns `ValidationException` ("the provided model identifier
is invalid"), and a published-but-unsubscribed one returns
`AccessDeniedException`. A newly released model is often US- and global-only at
first, with other geos published later. Pricing is per-profile: the `global.`
profile is the cheapest and each geo or in-region entry is exactly `1.10x` the
global rate in every column (pinned by `TestGeoGlobalRatio` as a tripwire).
Bedrock models are authored as one `family` declaration in `models.go`
(expanded by `families.go`): declare the published `Profiles`, the geo `Rates`
and `GlobalRates` cards, and reuse the shared capability/constraint vars when
the context window matches an existing generation. Add lookup tests for every
published ID and region-allow tests; the integration conformance suite then
picks new entries up automatically through `Catalog()`.

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
