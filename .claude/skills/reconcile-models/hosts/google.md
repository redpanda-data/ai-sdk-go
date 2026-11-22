# Gemini API (`providers/google`)

Where the Gemini API publishes each fact. Google is also its models' vendor, so its pages also give the model-wide facts and the vendor's replacement pick that every host uses. Start with `shared.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Rates | `https://ai.google.dev/gemini-api/docs/pricing.md.txt` | one section per model, each with a Standard, Batch, Flex and Priority table |
| Models | `https://ai.google.dev/gemini-api/docs/models.md.txt` | one section per family, each with an "All … models" table of model codes, linked to each model's page |
| One model | `https://ai.google.dev/gemini-api/docs/models/{id}.md.txt` | Model code, Supported data types, Token limits, Capabilities, Consumption options, Versions, Latest update |
| Deprecations | `https://ai.google.dev/gemini-api/docs/deprecations.md.txt` | one table per family: model, release date, shutdown date, recommended replacement |
| Release days | `https://ai.google.dev/gemini-api/docs/changelog.md.txt` | one heading per day |
| PDF input | `https://ai.google.dev/gemini-api/docs/document-processing.md.txt` | the opening paragraph |
| Decisions-only models | `https://ai.google.dev/gemini-api/docs/decisions-api.md.txt` | Limitations: the only model and API version that serve the Decisions API; its rates are the pricing page's Gemini Decision 1 section (input only, output "Not applicable") and its limits the model page's Token limits |
| Decisions request and usage | `https://ai.google.dev/api/decisions.md.txt` | POST `/v1beta/decisions` and the Usage fields (`total_input_tokens`, `input_tokens_by_modality`, `total_output_tokens`) |
| Gemma releases | `https://ai.google.dev/gemma/docs/releases` | one heading per release day (HTML: the `.md.txt` export is gone since 2026-11-03) |

## Reading them

- Append `.md.txt` to any page to get Markdown.
- Rates: the Standard table is the base card. A cell that splits at "prompts <= 200k tokens" gives a bracket. Output prices include thinking tokens. `providers/google` has no service-tier selector, so the Batch, Flex and Priority tables have no field (no-catalog-field).
- Thinking levels: the Gemini API rejects a level a model doesn't take with an error; Gemini 4 Hydrogen "rejects `minimal` with an error" (`https://ai.google.dev/gemini-api/docs/whats-new-gemini-4-hydrogen`, read 2026-11-16). So a documented level `geminiThinkingLevels` lacks is a small-gap-fill case. genai v1.51.0 names no XHIGH constant, so that entry is `genai.ThinkingLevel("XHIGH")`, beside a new `ReasoningEffort` constant. The effort lists in `models.go` are shared by several models (small-gap-fill).
- Limits: the model page's Token limits, "Input token limit" and "Output token limit" (limit-follow).
- Capabilities: the model page's Supported data types, Capabilities and Consumption options rows (capability-follow). The PDF page documents PDF input for Gemini models, so it covers a model whose page leaves PDF out of its inputs (gemini-2.5-flash's does).
- Lifecycle: the deprecations page plus the changelog. `ReplacedBy` comes from the replacement column or a changelog recommendation (replacement-pick). `Deprecated` is the day of the changelog entry that announces it (lifecycle-exact-date). Every shutdown date on the deprecations page is a floor (lifecycle-exact-date).
- Gemma: the Gemma releases page gives each release day and Bedrock's card its launch day; `Released` is the first day callable anywhere (released-first-callable).
- **Never copy lifecycle between Google surfaces.** Vertex retired the Gemini 2.5 models on 2026-10-20; the Gemini API still serves them. Gemini on Vertex takes its host facts from `vertex.md`.

## Probes (hand runs only)

- When a date has passed, the models list shows what still exists (it needs a key):
  ```bash
  curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY&pageSize=200" \
    | jq -r '.models[].name' | sed 's|^models/||' | sort
  ```
  A model absent here no longer serves (this is how `gemini-3-pro-preview` was confirmed retired). `Retires` still comes only from a page's exact date (lifecycle-exact-date); tell the person what the list shows.
