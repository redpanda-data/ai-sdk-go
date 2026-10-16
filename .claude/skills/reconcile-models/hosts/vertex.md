# Vertex (`providers/vertex`)

Where Google Cloud's Vertex (the Gemini Enterprise Agent Platform) publishes facts for the Claude models it serves. Google prices and schedules Claude on Vertex itself. Start with `shared.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Rates | `https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing?hl=en` | the Anthropic Claude models section (`#anthropics-claude-models`), one tab per region |
| One model | `https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/{slug}?hl=en` | Model ID, Launch stage, Supported inputs and outputs, Token limits, Versions (each with a launch stage and release date), Model availability, Retirement Date |
| All Claude models | `https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude?hl=en` | the list of models, with a link to each model page |
| Locations | `https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations?hl=en` | the model-by-location table |

## Reading them

- IDs: the model page's Model ID. Older Claude models carry a version (`claude-opus-4-5@20251101`); newer ones are bare (`claude-sonnet-5-5`).
- `Available`: the release date the model page gives for that version is Vertex's own launch day (available-host-launch). `Released` stays the model-wide date from Anthropic's pages.
- Launch stage: the model page's Launch stage, GA or Preview (stage-follow).
- Retirement: the model page words it "Not sooner than" a date, which is a floor (lifecycle-exact-date). Quote it in the entry's source comment.
- Limits: the model page's Token limits, "Maximum input tokens" and "Maximum output tokens" (limit-follow).
- Locations: `servedLocations` in `providers/vertex/locations.go` is copied from a model page's Model availability section, or from the locations table for a model whose page has none (location-follow). The table lists some locations the page leaves out, such as asia-east1 for Haiku 4.5, where calls fail. A location the page lists only under "ML processing", with no availability and no price, isn't served, so leave it out. Note the source and the day you read it in the comment above it.
- Rates: each region tab holds that region's rates; read global and each served region. A region the model is served in with no tab: region-premium-no-tab. The code writes the two cards with `regionalPricing(global, nonGlobal, regions...)`, or with one `WithOverride` per region for a model that needs it.
- Tool search: Google's docs never mention it. A new entry starts with it off (`claudeCaps`), whatever Anthropic's page says. A model's flag goes off when Anthropic's tool-search page drops it, and a person turns it on once a call on Vertex returns a tool search result (vertex-tool-search-one-way).

## Self-check

- Every non-global rate, brackets included, should be 1.10x the global rate. `TestGlobalToNonGlobalRatio` checks base rates only, so check brackets by hand. Where Google publishes another ratio, follow Google (price-follow) and update the test (owned-test-update).
