# Mistral AI (vendor of Mistral models)

Mistral AI's own pages for its models. Mistral has no package of its own: Bedrock sells its models, so this run owns them in `providers/bedrock` (RULES.md, Report values, says how to name them). Start with `shared.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Models | `https://docs.mistral.ai/models.md` | one row per model: name, API name, status, context |
| One model | the `https://docs.mistral.ai/models/<page>.md` the models page links | API names, Status, Release date, Version, Context length, Input, Output, Features, Deprecation date, Retirement date, Replacement |
| Every docs page | `https://docs.mistral.ai/llms.txt` | an index |

## Reading them

- Mistral's pages describe Mistral's own API, which no package in this SDK calls. Most models there aren't on Bedrock, and some are other vendors' (Z.ai, DeepSeek).
- For a model on Bedrock, take two facts from them: `Released`, the model page's Release date (released-first-callable), and the vendor's replacement pick, its Replacement, for a copy that Bedrock's pages mark Legacy or EOL (replacement-pick). Every other fact comes from the Bedrock card (`bedrock.md`).
- **Never copy lifecycle from Mistral's API to Bedrock.** On 2026-11-02, Mistral Large 3's page said Deprecated (deprecation date 2026-10-31, retirement 2027-04-30), while its Bedrock card gave no EOL date (lifecycle-exact-date).
- Match a Bedrock card to its Mistral page by model name, since the IDs differ: Mistral Large 3 is `mistral.mistral-large-3-675b-instruct` on Bedrock, `mistral-large-2512` on Mistral's API, and its page is `mistral-large-3-25-12.md`.
- Neither Mistral's pages nor the Bedrock card give a knowledge cutoff.
