# OpenAI API (`providers/openai`)

Where the OpenAI API publishes each fact. OpenAI is also its models' publisher, so its pages also give the model-wide facts and the publisher's replacement pick that every host uses. Start with `shared.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Rates | `https://developers.openai.com/api/docs/pricing.md` | Standard pricing data (short- and long-context columns), then the Batch, Flex, Fast and Ultrafast tables |
| Models | `https://developers.openai.com/api/docs/models.md` | a link to each model's page |
| One model | `https://developers.openai.com/api/docs/models/{id}.md` | Model details (context window, maximum input tokens, max output tokens, knowledge cutoff), Pricing, Endpoints, Supported features, Snapshots |
| Deprecations | `https://developers.openai.com/api/docs/deprecations.md` | one dated section per deprecation, with its shutdown date and recommended replacement |
| Release days | `https://developers.openai.com/api/docs/changelog.md` | one heading per day; a "Released" entry names the model |

## Reading them

- Append `.md` to any page to get Markdown, then grep its tables: the tables are the data. Don't summarize the deprecations page: it is long, and summaries of it have contradicted each other about which models are deprecated.
- IDs: the catalogue keys on the API ID as the pages write it, dots included (`gpt-6.1-sol`). A model page's Snapshots section lists its dated snapshots and aliases.
- Rates: the Standard table is the base card. Where it has long-context columns, the whole request above the threshold uses them, so they are a bracket. The Batch, Flex, Fast and Ultrafast tables each give an override card, and a model carries one only when that table lists the model's own row. A rate a table leaves "-" stays 0 (unpriced). Copy rates as printed, rounded ones included (price-follow). The comments above `entries()` in `providers/openai/models.go`, and an entry with every column such as GPT-6 Astra's, show how each column is written; follow them.
- Limits: the model page's context window, maximum input tokens and max output tokens. `MaxInputTokens` is the documented maximum input, not the window (limit-follow).
- `Knowledge`: the model page's knowledge cutoff (knowledge-month-end).
- Lifecycle: the deprecations page is the only source for deprecation and shutdown dates, and its dates are exact. A model with no announcement carries no dates (lifecycle-exact-date). `ReplacedBy` is the recommended replacement only when the catalogue carries it (replacement-pick).
- Release days: the changelog's day for a model is the day it reached the API, the OpenAI offering's `Available`. A model can reach ChatGPT or Codex a day or more earlier (GPT-5.5: 23 April in ChatGPT, 24 April in the API), and ChatGPT's changelog isn't a named host (released-first-callable).
- Capabilities: the model page's Endpoints and Supported features lists (capability-follow).
- OpenAI models on Bedrock take their facts from Bedrock's own cards (`bedrock.md`).
