# Meta Model API (`providers/meta`)

Where Meta's Model API publishes each fact. Meta is also its models' vendor, so its pages also give the model-wide facts. Meta's Llama models are sold on Bedrock, and their cards give their facts (`bedrock.md`). Start with `shared.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Models | `https://dev.meta.ai/docs/models.md` | one section per model line (Muse Wonder, Muse Spark and others), each with its versions, their IDs, inputs and outputs, context window, and a table of where each ID is available |
| Rates | `https://dev.meta.ai/docs/pricing-rate-limits.md` | Model pricing: one table per model line, with a Standard mode and a Contributor mode column; Rate limits: one row per ID |
| Reasoning | `https://dev.meta.ai/docs/reasoning.md` | the effort levels each model takes, and Constraints |
| Features | `https://dev.meta.ai/docs/structured-output.md`, `tool-calling.md`, `image-understanding.md`, `video-understanding.md`, `prompt-caching.md` | which models support each feature |
| Release days | the model's launch post on `https://research.meta.ai/blog/` (Muse Spark 1.3's is `introducing-muse-spark-1-3`), and its model card, which the models page links | the post's date |
| Every docs page | `https://dev.meta.ai/llms.txt` | an index |

## Reading them

- Add `.md` to a docs page's path to get Markdown.
- IDs: the models page writes each API ID in backticks, dots included (`muse-spark-1.3`); the catalogue keys on it.
- Rates: the Standard mode column is the base card. The Contributor mode IDs (`muse-spark-1.3-contributor` and the like) are separate IDs with their own rates, and Meta may use requests to them to train its models (the pricing page's Pricing modes).
- Limits: the models page gives each line's context window, and Meta publishes no separate output cap, so `MaxOutputTokens` stays the SDK's bound, as the entry's comment says (limit-follow).
- Capabilities: the models page's inputs and outputs, the feature pages, and the reasoning page's effort levels (capability-follow).
- Lifecycle: Meta publishes no deprecation page or changelog (`/docs/deprecations` and `/docs/changelog` were 404 on 2026-11-02), so an offering has dates only when a page gives them (lifecycle-exact-date).
- `Knowledge`: the model card, when it gives a cutoff (knowledge-month-end).
- Muse Spark Open and Muse Glimmer are open-weight releases that Model API doesn't serve.
