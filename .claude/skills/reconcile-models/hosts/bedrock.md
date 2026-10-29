# Bedrock (`providers/bedrock`)

Where Amazon Bedrock publishes facts for the models it serves. AWS sets its own IDs, regions, prices and lifecycle. Start with `shared.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Model cards | `https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-anthropic.md` | a link to each Anthropic model's card |
| One model | `https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-{slug}.md` | Model Details, Endpoints and APIs supported, Capabilities and Features, Programmatic Access, Service Tiers, Regional Availability |
| Rates | `https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/<region>/index.json` | AWS's public price list, one file per source region |
| Lifecycle, older models | `https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle-legacy.md` | the table of Legacy and EOL models |
| Lifecycle, newer models | `https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.md` | the policy for models launched on or after 2026-09-07: Legacy and EOL, and when a card gains its EOL date |
| OpenAI's model cards | `https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-openai.md` | a link to each OpenAI model's card |
| One OpenAI model | the card the cards page links | Model Details (launch date, EOL, context window, max output), Endpoints and APIs supported, Capabilities and Features, Pricing, Programmatic Access (the IDs), Service Tiers, Regional Availability |

The table names Anthropic's and OpenAI's cards. The other vendors `providers/bedrock` carries (Amazon Nova, Mistral, Google Gemma) have cards linked from each vendor's model-cards page. An OpenAI card's file name drops dots unevenly (`model-card-openai-gpt-56-sol.md` for GPT-5.6 Sol, `model-card-openai-gpt-6-1-sol.md` for GPT-6.1 Sol), so follow the cards page's link instead of building the name. Most OpenAI cards use these section names; GPT-6 Sol's and GPT-6 Luna's call the ID table "Call the model" and the region tables "Supported Regions".

## IDs and copies

- A model is a family declaration in `providers/bedrock/models.go`, or two when it also has a `bedrock-mantle` copy, as GPT-6's models do. `families.go` expands a family into one entry per profile in its `Profiles`, and adds a bare entry only when `BareInvokable` is set (bare-id-unregistered).
- The card's Programmatic Access table gives the exact IDs: the bare Model ID with its In-Region endpoint URL, the Geo inference IDs (`us.`, `eu.`, `au.`, `jp.` and the like) and the Global inference ID (`global.`).
- The card's Regional Availability tables say, for each source region, whether In-Region, Geo and Global work, and list the source regions each geo profile serves.
- Before declaring a profile, compare its prefix with RULES.md's Offered regions (new-copy-offered-region, region-not-offered). New models often start with only US and global profiles.
- Endpoints that make a bare ID callable on demand (bare-id-unregistered): an In-Region endpoint URL on `bedrock-runtime`, or `bedrock-mantle`'s Responses API. A `bedrock-mantle` Anthropic Messages URL (`.../anthropic/v1/messages`) doesn't count: this SDK's mantle transport speaks only the Responses API.
- Routing: a family's `ProfileRegions` map, in `cross_region.go`, sends bare-name calls from each source region to one profile. Its entries come from the card's per-geo source-region tables. A new copy points its source regions there (new-copy-offered-region); any other change to the map moves or blocks existing callers, so it goes to a person (no-rule).
- A copy the card no longer lists shows up when you compare the card's tables with the family's `Profiles` (copy-gone-no-eol).

## Prices

- The pricing page (`https://aws.amazon.com/bedrock/pricing/`) fills in its numbers with JavaScript, so a plain fetch sees none. Use the price list, and cite its URL.
- List one model's on-demand rates:
  ```bash
  curl -sL https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/us-east-1/index.json | jq -r '. as $l | .products[] | select(.attributes.servicename == "Claude Sonnet 5.5 (Amazon Bedrock Edition)") | .sku as $s | [.attributes.usagetype, ($l.terms.OnDemand[$s][]?.priceDimensions[] | .unit, .pricePerUnit.USD)] | @tsv'
  ```
- That file (`AmazonBedrockFoundationModels`) holds the Claude models and some OpenAI ones (GPT-6 Astra), each under its own `servicename`; `jq -r '[.products[].attributes.servicename] | unique[]'` lists them. Amazon's own models and most other vendors' (Nova, Mistral, Gemma, GPT OSS) are in the `AmazonBedrock` offer instead (`.../offers/v1.0/aws/AmazonBedrock/current/<region>/index.json`), where `servicename` is "Amazon Bedrock" for every row and the `model` attribute names the model (read 2026-10-15).
- Each file holds one source region's rates, per 1M tokens; `region_index.json` in the same folder lists the regions. Read `us.` and `global.` from `us-east-1`, and each other profile from one of its source regions (for example `eu-central-1`, `ap-southeast-2`, `ap-northeast-1`).
- The usage type names the rate. Newer models use `input_tokens_standard`, `output_tokens_standard`, `cache_read_tokens_standard`, `cache_write_tokens_standard` (5 minutes) and `cache_write_tokens_1h_standard`; older ones use `InputTokenCount`, `OutputTokenCount`, `CacheReadInputTokenCount`, `CacheWriteInputTokenCount` and `CacheWrite1hInputTokenCount`. A name with `global` in it is the `global.` rate; without it, the geo and in-region rate. Ignore `Batch`, `Reserved` and provisioned-throughput rows.
- `long_ctx` in a usage type marks a long-context tier. Neither the price list nor an Anthropic card gives its threshold; OpenAI's cards give 272K input tokens.
- Rate cards: `GlobalRates` is the `global.` card and `Rates` every other profile's. Speed or tier variation goes in the family's `Overrides` and `GlobalOverrides`, which a family with a global profile sets in pairs. The card's Service Tiers table shows which tiers the model has.
- Copy cache rates from the price list too (price-follow).
- OpenAI's cards print their own rates in a Pricing section: per endpoint and profile, short and long context, with the 10% premium already added. The price list holds only GPT-6 Astra (read 2026-10-28), so the card is the rate source for the other OpenAI models.

## Lifecycle

- AWS gives one Legacy date and one EOL date per model, and its tables have no replacement column (lifecycle-exact-date, replacement-pick).
- Which page holds them depends on the card's Model launch date:
  - launched before 2026-09-07: `model-lifecycle-legacy.md` lists only Legacy and EOL models, so a model missing from it has no announced end of life;
  - launched on or after 2026-09-07: the card's Model EOL date, which AWS adds when the Legacy period starts. `N/A` means none is announced, and "EOL no sooner than" is a floor.
- A profile's own launch day comes only from the authenticated `ListFoundationModels` API (`modelLifecycle`), not from any page (profile-available-unset).

## Limits and capabilities

- The card's Model Details give the context window and max output tokens (limit-follow). OpenAI's cards give no input cap.
- Its modality table and its Capabilities and Features lists give what Bedrock supports (capability-follow).

## Self-check

- Every geo or in-region rate, brackets and overrides included, should be 1.10x the global rate. `TestGeoGlobalRatio` checks base rates only, so check the rest by hand. Where the host publishes another ratio, follow the host (price-follow) and update the test (owned-test-update).

## Probes (hand runs only)

- A one-token `Converse` request tells a working ID (it answers) from an unpublished one (`ValidationException`, "model identifier is invalid") and one the account hasn't subscribed to (`AccessDeniedException`). A bare ID with no on-demand endpoint returns `ValidationException` ("on-demand throughput isn't supported"). Cards and the API have disagreed before, so hand runs confirm each new ID this way.
- In CI there are no AWS credentials, so new copies come from the card alone (new-copy-offered-region). The full Test job later calls copies only from its own region.
