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
