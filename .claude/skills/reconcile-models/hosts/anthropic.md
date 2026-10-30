# Anthropic API (`providers/anthropic`)

Where the Claude API publishes each fact. Anthropic is also Claude's vendor, so its pages also give the model-wide facts and the vendor's replacement pick that every host uses. Start with `shared.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Rates | `https://platform.claude.com/docs/en/about-claude/pricing.md` | the model table (base input, 5-minute and 1-hour cache writes, cache hits, output), and the long-context, fast-mode and batch sections |
| IDs, limits, cutoffs | `https://platform.claude.com/docs/en/models/overview.md` | the Compare models table: Claude API ID and alias, context window, max output, reliable knowledge cutoff, training data cutoff, retirement |
| One model | `https://platform.claude.com/docs/en/models/{slug}/overview.md` | the "Released" line at the top, then Specifications (model IDs, pricing, capabilities), including its Thinking and Default effort rows |
| Deprecations | `https://platform.claude.com/docs/en/about-claude/model-deprecations.md` | the Model status table, then one dated section per deprecation under Deprecation history |
| Release days | `https://platform.claude.com/docs/en/release-notes/overview.md` | one heading per day |
| Tool search | `https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool.md` | the model compatibility list |
| Features by platform | `https://platform.claude.com/docs/en/build-with-claude/overview.md` | which platforms have each feature, and where it is beta |
| Effort | `https://platform.claude.com/docs/en/build-with-claude/effort.md` | the effort levels each model accepts |
| Preserved thinking | `https://platform.claude.com/docs/en/build-with-claude/preserved-thinking.md` | the model list under "Keeping the prefix unchanged" |

## Reading them

- IDs: from the 4.6 generation on, each ID is dateless and is itself a pinned snapshot. Older models have a dated snapshot ID (`claude-opus-4-5-20251101`) and an alias the catalogue keys on (`claude-opus-4-5`).
- The deprecations page lists dated IDs for older models. A row for a dated snapshot counts for the entry whose ID points at that snapshot, but not for an alias that now points at a newer snapshot.
- Model status table: the Deprecated column is the deprecation date. In the retirement column, "Not sooner than" a date is a floor, a bare date is exact, and "To be announced" is no date (lifecycle-exact-date).
- Each Deprecation history section's Recommended replacement column is the vendor's pick, which every host uses (replacement-pick).
- `Released`: the model page's "Released" line, or the release notes heading for the day the ID first shipped (released-first-callable, release-month-only).
- `Knowledge`: the reliable knowledge cutoff, not the training data cutoff (knowledge-month-end).
- Limits: the overview's context window and max output (limit-follow). Max output there is the synchronous Messages API limit; the larger batch limit is a beta.
- Cache rates: the pricing page lists each model's cache write and cache hit rates. The cache hit share of the input rate differs between models, so copy each rate (price-follow).
- Thinking: each model page's Thinking row, checked for every model the catalogue carries, not only new ones (capability-follow). "Adaptive" or "Adaptive (always on)" sets `Adaptive`. "Extended" sets `Budget` and adds `thinking_budget` to `SupportedParams`. "Adaptive (extended deprecated)" sets both, with `thinking_budget`. A Default effort of "Not supported" means no effort levels; otherwise the effort page lists them.
- Thinking prefix check: `Reasoning.ThinkingPrefixCheck` must match the preserved-thinking page's model list (capability-follow). A newly listed model needs the flag, or its replayed thinking can 400 on newer accounts. `TestThinkingPrefixCheckComesFromTheCatalog` pins the set, so update it with the entries (owned-test-update).
- Tool search: a model on the compatibility list has hosted tool search; one marked unsupported, or absent, doesn't. On Vertex the same list only turns tool search off (capped-at-vendor).
- The overview and model pages also list Bedrock and Vertex IDs and say those hosts set their own dates. Take those hosts' facts from their own pages.
