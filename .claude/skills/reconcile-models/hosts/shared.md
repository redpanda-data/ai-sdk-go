# Every host: finding and recording facts

How to find a fact and write it down, whatever the host. `.github/reconcile-models/RULES.md` decides what to change, skip or ask; a name in parentheses below is a rule there. Each host's own page says where that host publishes each fact.

## Pages

- `.github/reconcile-models/named-sources.json` lists each run's packages, its named hosts and the pages to start from. Its `per_model` entries give each host's page for one model, with `{slug}` standing for an ID such as `sonnet-5-5`.
- Any page on a named host may be read, not only those.
- On `platform.claude.com` and `docs.aws.amazon.com`, add `.md` to a page's path to get clean markdown; on `docs.cloud.google.com`, add `.md.txt`. Pages on `cloud.google.com` are HTML; add `?hl=en`.
- Deprecation pages and price lists are long tables. Fetch the page and filter it with `grep` or `jq`; a summary of a long table loses rows.

## Where facts live in the code

- `catalog/facts_data.go` holds one facts record per model, shared by every host: `Publisher`, `DisplayName`, `Description`, `Series`, `Released`, `Knowledge`.
- `providers/<package>/models.go` holds one entry per offering: capabilities, modalities, constraints, reasoning, pricing and `Life` (`Available`, `Deprecated`, `Retires`, `ReplacedBy`).
- The add-model skill shows how to author both. "Writing prices" below says how to encode prices.

## Source comments

- Put a comment beside each value you change, naming the page's URL and the day you read it, for example `// Rates from https://platform.claude.com/docs/en/about-claude/pricing (read 2026-10-15).`
- Before you change a fixed fact (`Released`, `Knowledge`, a launch `Available`), read the comment beside it: label-fact-unproven and label-fact-differs turn on what it names.

## Writing prices

- Rates are US dollars per million tokens: `$2.50/M` is `2.50` in `pricing.NewRates`, `pricing.FlatInfo` and `.WithCacheCreation(w5m, w1h, 0)`. Never write microcent literals: `FlatInfo(250_000_000, ...)` compiles and overprices by 10^8.
- A 0 rate means unpriced, and `catalog.New` rejects a 0 base input or output rate: a published $0 one is `pricing.RateFree`. A 0 cache rate, bracket rate or override rate is allowed.
- Context tiers are `pricing.TieredInfo(base, pricing.Bracket{...})`. A bracket's rates replace the base above `MinContextTokens`, so write every field of the bracket, not only the ones that change.
- Variation by speed, tier or region is `Pricing.WithOverride(selector, pricing.RateCard{...})`, never a provider-specific field. An override card has its own brackets: a tiered model with a fast mode needs the bracket on the fast card too, or large fast requests underbill.
- When a host raises a model's context window, look on its pricing page for a long-context surcharge that came with it.
- Vendors often print a scheduled rate next to today's in the same cell; read the cell's note before copying a number. `pricing.Info` has no effective date, so only today's rate can be data (price-follow).

## Reading lifecycle pages

- Read each host's whole deprecation or lifecycle page on every run, and re-check every `ReplacedBy`, not only the models you suspect. Vendors revise their recommended replacements as newer models ship, and nothing in the repo fails when a vendor announces a date.
- Each host words its dates its own way. Its page here says which wording is an exact date and which is a floor (lifecycle-exact-date).
- A retiring model keeps its entry. Add a `// Deprecated:` doc comment on its ID constant that names the replacement.

## Probes (hand runs only)

- Context window: send a prompt just over the claimed size with the raw SDK (about 4 characters per token, with a margin). A real 1M model accepts it with no beta header; a 200K model answers "prompt is too long". Turn caching off while measuring, or sum `BilledInputTokens()`.
- Each host page lists its own probes.

## Hand runs: LiteLLM leads

- CI reads no aggregator. A hand run may compare LiteLLM's numbers with an entry: prices and input and output limits, nothing else.
- A number that differs is a lead. Show it to the person, then settle the value on the host's own page and cite that page. Never write a LiteLLM number as a value, never use LiteLLM for a release date or a lifecycle date, and never copy its text into `Description`.
- Pull only those numbers for one model, here `claude-opus-4-5`:
  ```bash
  curl -sL https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json | jq '."claude-opus-4-5" | with_entries(select((.key | test("cost|^max_(input|output)_tokens$")) and (.value | type == "number")))'
  ```
- Cost keys with `per_token` or `token_cost` in their name are US dollars per token: multiply by 1e6 for dollars per million. Other cost keys name their own unit. `max_input_tokens` and `max_output_tokens` compare with `MaxInputTokens` and `MaxOutputTokens`.
- LiteLLM files host copies under their own IDs: Vertex copies under `vertex_ai/...`, Bedrock copies under their Bedrock model IDs. Look up the copy for the package you are checking.
- OpenRouter isn't read in any run until someone has read its terms.

## In CI

- The run has no git history, so the skills' `git log -S` check for where a value came from is skipped; the source comment is the only record.
- The run has no host credentials, so no probe runs. A rule marked *Needs a probe* says what the bot does instead.
- Fetch pages exactly as `.github/reconcile-models/prompt.md` says.
