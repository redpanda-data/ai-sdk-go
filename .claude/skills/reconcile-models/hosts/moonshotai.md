# Moonshot AI models

Moonshot AI is the publisher (`moonshotai`), not Fireworks. This run currently owns its Kimi copies in `providers/fireworks`; there is no direct Moonshot provider package. Start with `shared.md` and `fireworks.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Creator release evidence | `https://platform.kimi.ai/docs/platform-changelog.md` | Kimi K3 availability and the precision of its launch date |
| Native Kimi K3 model and reasoning | `https://platform.kimi.ai/docs/guide/kimi-k3-quickstart.md` | modalities, context and reasoning controls |
| Creator announcement | `https://www.kimi.com/blog/kimi-k3` | app/API availability; an announcement alone is not an exact launch day when it gives only a month |
| Hosted model | `https://fireworks.ai/models/fireworks/kimi-k3` | exact invocation ID, serving context, supported features and FAQ |
| Fireworks reasoning mapping | `https://docs.fireworks.ai/api-reference/post-chatcompletions.md` | Kimi K3 model-specific effort map, not the shared reasoning enum |

## Reading them

- Canonical identity is `moonshotai/kimi-k3`; the Fireworks invocation ID is `accounts/fireworks/models/kimi-k3`.
- Creator evidence currently establishes July 2026, not a verified exact first-callable day. The existing model-wide release is 2026-07-31 under RULES.md (release-month-only); its source comment ends `day defaulted`. Fireworks resource creation, a weight release and first app/API availability are different events.
- Fireworks' K3 mapping has low/high/max prompt tiers, promotes medium to high and xhigh to max, and defaults to max. A shared API accepting an effort string does not establish a distinct model tier.
- Published Standard, Priority, Fast and US-only rates are separate billing selections. A Fast or US-only router has its own invocation ID and price card and follows RULES.md (fireworks-router-unscoped).
- Fireworks' serving limits, prices, lifecycle and API bindings are host facts. Use Fireworks' own documentation, not creator endpoint support, for those host facts.
