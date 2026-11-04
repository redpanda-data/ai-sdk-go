# Fireworks (`providers/fireworks`)

Fireworks is a hosting provider, not the publisher of OpenAI, Z.ai, MiniMax, DeepSeek or Moonshot AI models. Each creator's run owns its Fireworks copies. Start with `shared.md`, the creator's page and this page.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Serverless rates and Priority availability | `https://docs.fireworks.ai/serverless/pricing.md` | Standard and Priority columns give input / cached input / output USD per million tokens; a dash means Priority is unavailable |
| Serverless lifecycle | `https://docs.fireworks.ai/updates/changelog.md` | dated deprecation announcements, exact shutdown dates and their serverless/global/US-only scope |
| Model IDs, serving limits and features | `https://fireworks.ai/models/<library-owner>/<model-id>` | model path, Serverless status, specifications and FAQ; use this public page rather than the login-gated `app.fireworks.ai` page |
| Modes and router IDs | `https://docs.fireworks.ai/serverless/serverless-modes.md` | Standard, Priority, Fast and Ultra-fast are different billing selections; speed routers have their own IDs |
| Messages API and accounting | `https://docs.fireworks.ai/tools-sdks/anthropic-compatibility.md` | supported fields, reasoning mapping, unsupported operations and Token usage |
| Chat reasoning controls | `https://docs.fireworks.ai/api-reference/post-chatcompletions.md` | model-specific reasoning effort behaviour, not only the shared enum |
| Responses API | `https://docs.fireworks.ai/guides/response-api.md` | endpoint, supported options and retention |

## Reading them

- Preserve the exact `accounts/fireworks/models/...` invocation ID. A resource existing in the library does not establish serverless availability. The public GPT-OSS 20B page says serverless is unsupported.
- Standard prices form the default card. Published Priority prices form a service-tier override; do not assume a uniform premium. Cached reads use Fireworks' published rate. Messages reports cache creation as zero, not an Anthropic write multiplier.
- Pricing the Priority card does not establish effective-tier attribution. Messages and Chat document `service_tier: "priority"`; Responses support and effective-tier response metadata require separate qualification. Never silently treat an unknown effective tier as Standard.
- The changelog's explicit serverless shutdown is exact. For example, GPT-OSS 120B retires on 2026-11-07; dedicated availability does not extend its serverless life. Retained historical entries stay priceable.
- Model resource `Created on` is not the creator's first release day. The creator's page supplies model-wide facts. Fireworks' own launch evidence supplies its offering's availability.
- A native context window is not automatically the hosted input cap. Public cards may format limits as `1040k` or `512k`; do not invent an exact served integer from that formatting. Cite a verified exact limit when available. A sample `max_tokens` value is not an output cap.
- Current pinned qualification candidates are `glm-5p3`, `glm-5p3-flash`, `minimax-m3`, `deepseek-v4p1-flash`, `deepseek-v4p1-flash-1017`, `kimi-k3` and the historical `gpt-oss-120b`. DeepSeek's public library owner is `deepseek-ai`, while its invocation IDs still begin `accounts/fireworks/models/`. Read each model's own page. Z.ai's graded efforts, MiniMax M3's adaptive mode and GPT-OSS' always-on reasoning are not interchangeable.

## Catalog-only qualification

Fireworks deliberately has no SDK request transport. Its existing qualified text/chat entries remain maintained for gateway discovery and pricing; the scoped exceptions and expansion gate live only in RULES.md (model-not-callable, reasoning-unsendable, fireworks-contract-qualification).

Each offering's API bindings, per-API features and modalities require fresh committed live-contract evidence for that exact offering. The provider's three endpoint documents, another model's capture and a basic HTTP success do not establish those contracts. Keep new offerings and contract expansions for a trusted qualification run; do not fabricate captures or credentials in reconciliation CI.

A moving alias, Fast/Ultra-fast router, Nexus `auto`, US-only endpoint, private deployment or dedicated GPU-time rate is not a fixed-price alias of a qualified base offering. These require their own reviewed scope and qualification. The bot does not add an outbound transport or choose a new gateway/client default.
