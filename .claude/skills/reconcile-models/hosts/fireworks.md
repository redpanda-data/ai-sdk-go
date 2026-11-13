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
- Pricing the Priority card does not establish effective-tier attribution. Check each API's documentation for `service_tier: "priority"` support and effective-tier response metadata; Messages and Chat document the request field. Never silently treat an unknown effective tier as Standard.
- The changelog's explicit serverless shutdown is exact. For example, GPT-OSS 120B retires on 2026-11-07; dedicated availability does not extend its serverless life. Retained historical entries stay priceable.
- Model resource `Created on` is not the creator's first release day. The creator's page supplies model-wide facts. Fireworks' own launch evidence supplies its offering's availability.
- A native context window is not automatically the hosted input cap. Public cards may format limits as `1040k` or `512k`; do not invent an exact served integer from that formatting. Cite a verified exact limit when available. A sample `max_tokens` value is not an output cap.
- Current documented offerings are `glm-5p3`, `glm-5p3-flash`, `minimax-m3`, `deepseek-v4p1-flash`, `deepseek-v4p1-flash-1017`, `kimi-k3` and the historical `gpt-oss-120b`. DeepSeek's public library owner is `deepseek-ai`, while its invocation IDs still begin `accounts/fireworks/models/`. Read each model's own page. Z.ai's graded efforts, MiniMax M3's adaptive mode and GPT-OSS' always-on reasoning are not interchangeable.

## Catalog metadata and runtime support

Fireworks provides a selected-binding SDK runtime alongside native catalog metadata for discovery and pricing. The factory uses exactly the offering's first normalized wire API, without fallback; runtime capabilities and admitted controls come from that binding, not the catalog union. Catalog `SupportedParams` stays empty and does not describe runtime admission. The SDK representability rules live in RULES.md (model-not-callable, reasoning-unsendable).

Use Fireworks' documentation for each offering's API bindings, per-API features and modalities under the ordinary RULES.md rules. Recordings are regression fixtures, not catalog qualification certificates: adding documented models or features does not require live calls or committed captures. Native support does not imply SDK support: image input is not representable by `llm.Part`, and configured MiniMax Messages models with explicitly disabled thinking withhold schema output. Preserve those runtime constraints separately from native metadata.

A moving alias, Fast/Ultra-fast router, Nexus `auto`, US-only endpoint, private deployment or dedicated GPU-time rate is not a fixed-price alias of a base offering: preserve its documented invocation identity and billing selection. Routers require their own reviewed scope under RULES.md (fireworks-router-unscoped). The bot does not add an outbound transport or choose a new gateway/client default.
