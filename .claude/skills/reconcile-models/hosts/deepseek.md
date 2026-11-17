# DeepSeek models

DeepSeek is the publisher (`deepseek`), not Fireworks. This run currently owns DeepSeek's copies in `providers/fireworks`; there is no direct DeepSeek provider package. Start with `shared.md` and `fireworks.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Base V4.1 Flash release | `https://api-docs.deepseek.com/news/news260910` | Release 2026/09/10 and availability on the website, app and API |
| V4.1 Flash 1017 release | `https://api-docs.deepseek.com/news/news261017` | Release 2026/10/17; this is a distinct released version, not the September release day |
| Native limits and reasoning | `https://api-docs.deepseek.com/api/list-models` | sample model object: exact context and output limits, model version, owner and effort levels |
| Current native model version | `https://api-docs.deepseek.com/quick_start/pricing` | current model version, modalities, context, output and thinking/non-thinking support; these prices are not Fireworks rates |
| Hosted model | `https://fireworks.ai/models/deepseek-ai/deepseek-v4p1-flash` and `https://fireworks.ai/models/deepseek-ai/deepseek-v4p1-flash-1017` | public library specifications, exact invocation IDs and FAQ |
| Fireworks reasoning mapping | `https://docs.fireworks.ai/api-reference/post-chatcompletions.md` | DeepSeek V4.1 model-specific mapping, distinct from V4 |

## Reading them

- Canonical IDs are `deepseek/deepseek-v4.1-flash` and `deepseek/deepseek-v4.1-flash-1017`. The Fireworks invocation IDs use `v4p1`, and the public model-library owner is `deepseek-ai`. Do not construct its library URL under `/models/fireworks/`.
- Keep the base and 1017 entries distinct. A creator pricing page naming the currently served version does not prove Fireworks redirects the base ID or shares a fixed pricing identity with an alias.
- The creator publishes native 1,048,576 context and 393,216 maximum output for the current model version. Fireworks' hosted limits remain host facts, subject to RULES.md (capped-at-vendor, limit-follow).
- Fireworks maps low to low; medium/high/xhigh to high; max to max; omitted effort to high. None disables thinking. Its numeric 1-100 effort scale is not a token budget; a documented budget is a different control. Do not inherit the earlier DeepSeek V4 mapping.
- API compatibility and capabilities on the creator's endpoint do not establish Fireworks' Messages, Chat or Responses bindings. Use Fireworks' own documentation for those host facts.
