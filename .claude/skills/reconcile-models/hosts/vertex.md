# Vertex (`providers/vertex`)

Where Google Cloud's Vertex (the Gemini Enterprise Agent Platform) publishes facts for the Claude and Gemini models it serves. Google prices and schedules both on Vertex itself. Start with `shared.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Rates | `https://cloud.google.com/model-garden/partner-models/pricing?hl=en` | the Anthropic Claude models section (`#anthropics-claude-models`), one tab per region |
| One model | `https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/{slug}?hl=en` | Model ID, Launch stage, Supported inputs and outputs, Token limits, Versions (each with a launch stage and release date), Model availability, Retirement Date |
| All Claude models | `https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude?hl=en` | the list of models, with a link to each model page |
| Locations | `https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations?hl=en` | the model-by-location table (Claude and Gemini) |
| Gemini rates | `https://cloud.google.com/model-garden/pricing?hl=en` | the Gemini 4 (`#gemini-4`) and Gemini 3 (`#googlemodels`) sections, each with a Standard, a Priority and a Flex/Batch table, in that order |
| Gemini lifecycle | `https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions.md.txt` | one table per family: Model ID, Release date, Retirement date, Replacement model ID |
| One Gemini model | `https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/{slug}.md.txt` | Model ID, Modalities, Token limits, Capabilities, Tools, APIs, Consumption options, and one row of locations each for ML processing, Provisioned Throughput and Standard PayGo; the slug drops `gemini-` and writes dots as dashes (`3-6-flash`) |

## Reading them

- IDs: the model page's Model ID. Older Claude models carry a version (`claude-opus-4-5@20251101`); newer ones are bare (`claude-sonnet-5-5`).
- `Available`: the release date the model page gives for that version is Vertex's own launch day (available-host-launch). `Released` stays the model-wide date from Anthropic's pages.
- Launch stage: the model page's Launch stage, GA or Preview (stage-follow).
- Retirement: the model page words it "Not sooner than" a date, which is a floor (lifecycle-exact-date). Quote it in the entry's source comment.
- Limits: the model page's Token limits, "Maximum input tokens" and "Maximum output tokens" (limit-follow).
- Locations: `servedLocations` in `providers/vertex/locations.go` is copied from a model page's Model availability section (on a Gemini page, its Provisioned Throughput and Standard PayGo rows), or from the locations table for a model whose page has neither (location-follow). The table lists some locations the page leaves out, such as asia-east1 for Haiku 4.5, where calls fail. A location the page lists only under "ML processing", with no availability and no price, isn't served, so leave it out. Note the source and the day you read it in the comment above it.
- Rates: each region tab holds that region's rates; read global and each served region. A region the model is served in with no tab: sibling-ratio, with the ratio from Anthropic's pricing page, since Anthropic sets Claude's regional premium. The code writes the two cards with `regionalPricing(global, nonGlobal, regions...)`, or with one `WithOverride` per region for a model that needs it.
- Thinking prefix check: Anthropic's preserved-thinking page covers Google Cloud too, so `ThinkingPrefixCheck` follows the same model list as in `anthropic.md` (capability-follow). No Vertex test pins it, so check it by hand.
- Capabilities, thinking and tool search included: Anthropic's pages cap what a Vertex entry claims (capped-at-vendor). Google's docs never mention tool search, so a new Claude entry starts with it off (`claudeCaps`), and a person turns it on once a call on Vertex returns a tool search result.
- Gemini rates: the Standard table is the base card, with a Global and a Non-global row per model; the Priority and Flex/Batch tables have no field (no-catalog-field). Gemini 3's columns are input up to 200K, input above 200K (a bracket), then the same two for cached input. The page's HTML holds the sections on one very long line, so cut one out before you read it. This cuts Gemini 3 (`googlemodels` to `gemini-25`); for Gemini 4, use `gemini-4` to `googlemodels`:
  ```bash
  curl -sL 'https://cloud.google.com/model-garden/pricing?hl=en' \
    | jq -Rrs 'capture("id=\"googlemodels\"[^>]*>(?<s>.*)id=\"gemini-25\"").s | gsub("<[^>]+>"; " ") | gsub("&lt;"; "<") | gsub("&gt;"; ">") | gsub("\\s+"; " ")'
  ```
- Gemini lifecycle: in the model-versions table a bare date is exact and "or later" is a floor (lifecycle-exact-date). Its Release date is Vertex's own launch day (available-host-launch); the model page gives none. Never copy lifecycle from the Gemini API (`google.md`).

## Self-check

- Every non-global rate, Claude's and Gemini's, brackets included, should be 1.10x the global rate. `TestGlobalToNonGlobalRatio` checks base rates only, so check brackets by hand. Where Google publishes another ratio, follow Google (price-follow) and update the test (owned-test-update).
