# Fireworks catalog

This package supplies `catalog.Provider` and published serverless token prices under the provider key `fireworks`. It does **not** implement an outbound `llm.Model`, gateway proxy, Claude Code/Codex launcher, or spending pipeline. Accepting Anthropic Messages does not make an offering Anthropic-published.

## Scope and identity

Use exact invocation IDs. `catalog.WireAPI` is the existing shared protocol/transport contract; each supported Fireworks binding uses the `direct` transport. API-specific capabilities and reasoning controls are explicit `WireAPISupport` records, not inferred from provider names, creators, model prefixes, or the model-level capability superset. Missing API feature records mean unknown, not full feature parity. Messages has schema-based structured output but no separate schemaless JSON mode.

`Constraints.SupportedParams` is intentionally empty because no SDK request transport/control-admission guarantee is supplied. `llm.ModelConstraints.ValidateParameterSupport` therefore rejects every SDK option; nil is not a tri-state validator and is never supports-all. This is **not** a claim that the upstream APIs support no controls: their truthful capabilities and per-API reasoning metadata remain present. Catalog/UI consumers must not project this absent SDK admission list into an upstream supports-none or supports-all claim. Native dialect controls require their own qualified API contract.

Resolution is exact, including exact authored aliases. An unlisted date suffix, deployment, router, shorthand, or `*-latest` alias must not inherit the base model's API support or price. No moving aliases are currently registered. Router variants, US-only capacity, custom/private/fine-tuned deployments, batch, embeddings, and decision-only models are outside this catalog's supported contract. Messages support does not advertise native count-tokens: the observed `/inference/v1/messages/count_tokens` endpoint returned 404. Hosted `ToolSearchProtocol` remains empty; ordinary tools and Fireworks' client-side deferred-tool compatibility do not establish provider-hosted tool execution.

Candidate qualification inventory (checked 2026-11-04). A catalog row is not a passing acceptance result; only a complete passing manifest qualifies the declared contract:

| Invocation ID suffix (`accounts/fireworks/models/…`) | Creator | Role | Standard input/cache/output USD/M | Priority input/cache/output USD/M |
| --- | --- | --- | --- | --- |
| `glm-5p3` | Z.ai | Coding main | 1.40 / 0.26 / 4.40 | 1.75 / 0.325 / 5.50 |
| `glm-5p3-flash` | Z.ai | Inexpensive background; vision-capable | 0.15 / 0.03 / 0.50 | 0.1875 / 0.0375 / 0.625 |
| `minimax-m3` | MiniMax | Lower-cost coding main | 0.30 / 0.06 / 1.20 | 0.45 / 0.09 / 1.80 |
| `deepseek-v4p1-flash` | DeepSeek | Coding main; independently qualified original version | 0.30 / 0.006 / 1.20 | 0.375 / 0.0075 / 1.50 |
| `deepseek-v4p1-flash-1017` | DeepSeek | Current pinned coding update; not a price-only alias | 0.30 / 0.006 / 1.20 | 0.375 / 0.0075 / 1.50 |
| `kimi-k3` | Moonshot AI | Multimodal coding main | 3.00 / 0.30 / 15.00 | 3.75 / 0.375 / 18.75 |
| `gpt-oss-120b` | OpenAI | Historical/smoke only; **retires 2026-11-07** | 0.15 / 0.015 / 0.60 | 0.18 / 0.018 / 0.72 |

These are published estimates, not negotiated invoice rates. Cache creation is explicitly free, not an Anthropic cache-write multiplier. Priority rate cards describe published prices only: effective tier attribution is **blocked** (`priority_tier_status=blocked_missing_effective_tier`). A downstream request implementation must reject Priority until it can establish billing semantics; a successful Priority HTTP response alone is not evidence. The Responses schema does not currently document `service_tier`.

Input limits use exact `context_length` values returned by the live inference model listing (GLM: 1,048,576; MiniMax: 512,000; GPT-OSS: 131,072). Prompt and output share that window. GLM output is capped at the creator's 131,072-token limit under the repository's `capped-at-vendor` rule; the Fireworks generic API allows output up to remaining context. No full-window paid probe was performed. Unknown knowledge cutoffs and exact host launch dates remain unset. GLM's creator release is August 14, not the conflicting Fireworks FAQ's August 18 or the hosting resource's creation date. DeepSeek output is capped at the creator's 393,216-token limit. Kimi's creator gives only July 2026 for API launch; its July 31 release is explicitly day-defaulted under `release-month-only`, not asserted as a verified exact day. This user-requested current coding inventory deliberately includes models older than the unattended bot's 90-day new-model default.

## Inventory reconciliation and exclusions

`GET /inference/v1/models` returned 30 IDs with this account on 2026-11-04. Listing is reachability/discovery evidence, not proof of invocation, entitlement, fixed prices, or per-API features. The public price table and creator/host model pages are independent inputs.

| Listed/public offering or group | Status/reason |
| --- | --- |
| `deepseek-v4p1-flash`, `deepseek-v4p1-flash-1017`, `kimi-k3` | Included in the expanded candidate matrix, each with its own API/feature evidence. Inclusion is not a passing acceptance claim. |
| `qwen3p8-max` | Published rates and live listing, but exact served context was absent in the listing and its public card says Context Length N/A. This is an unknown-limit evidence/qualification blocker, not evidence its APIs are unsupported. Catalog validation requires a positive known limit, so no guessed limit is registered. |
| `nemotron-lightning-3p5-30b-a3b`, `nemotron-3-ultra-nvfp4` | Published rates and listing; both retire 2026-11-07. Withheld from this new coding rollout under `model-on-way-out`. Retirement does not prohibit historical catalog representation: GPT-OSS is deliberately retained for regression of the earlier real three-protocol usage probes, not as a durable default. |
| `ember-1` | Published vision pricing; outside the initial coding/background qualification batch; no per-API qualification claim. |
| `muse-spark-open` | Published rates but US-only capacity; deferred deployment/billing surface. The listing also reports `supports_chat=false`. |
| `minimax-m3p1-flash`, `qwen3p8-2p4t-a95b`, `inkling` | Listed, but absent as exact headline rows in the checked public price table; no guessed architecture/size/cache/Priority rates. |
| `gpt-oss-20b` | Not in the live listing; official card says serverless unsupported, deprecated 2026-08-27. Not an inexpensive serverless fallback. |
| `glm-5p2-fast`, `glm-5p3-fast`, `kimi-k3-fast`, `deepseek-v4p1-flash-1017-ultrafast` | Distinct router identities, not equivalent aliases. Fast/Ultra-fast and their independently priced contracts are deferred. The older GLM router has no current exact headline rate row. |
| All `*-spot` routers, including `spot-deepseek-v4-flash` | Separate routing/billing contracts, deferred; do not inherit Standard rates by prefix. |
| `auto`, `firerouter/auto` | Nexus routing, including potentially external credentials; not fixed-price model offerings. |
| `qwen3-embedding-8b`, `qwen3-reranker-8b` | Embedding/reranking APIs, outside text generation. |
| `rune` model/router | Decision API, no generated output tokens; not chat/Responses/Messages generation. |
| US-only variants not returned in the default listing | Separate capacity and prices; deferred. |

## Required direct API acceptance

Live API tests are local-only and explicitly opt-in; CI runs credential-free catalog, unit, and offline replay tests without a Fireworks key. Explicit acceptance is billable and must fail on a missing credential, missing evidence, absent required test, skipped required case, unsupported newly advertised feature, catalog drift, or any failed contract. It is not a gateway or CLI acceptance claim.

```sh
# Credential-free strict replay + catalog-driven completeness:
task fireworks:replay

# Explicitly replace reviewed live evidence, never automatic in CI:
task fireworks:record -- -fireworks-key-file /path/to/credential-file -fireworks-budget-usd=1

# Required release acceptance: replay followed by every live required case:
task fireworks:accept -- -fireworks-key-file /path/to/credential-file -fireworks-budget-usd=1
```

The key is read inside the test process and used only in its HTTP authorization header. Do not put its contents in shell arguments, logs, fixtures, or reports. Responses requests always send `store: false`. Calls are sequential, output-token capped, and reserve conservative worst-case cost before sending. An explicit finite positive per-run `-fireworks-budget-usd` limit is required; it is not a persistent SDK spending policy. Recording must checkpoint sanitized exchanges before parsing so parser failures cannot lose evidence or spend reservations. Never convert an incomplete run to successful evidence by deleting failing cases.

The machine-readable manifest pins exact offerings, protocol/transport, stream mode, request options, contract implementation, catalog fingerprint, capture time, and SDK revision. For these intentionally uncommitted implementation runs, the Git revision is the clean **base**, not a claim that upstream already contains Fireworks; catalog/test/request hashes identify the actual candidate. A corrected verifier is recorded separately from each immutable capture's original revision and fingerprint, and incremental repairs are labelled partial corrections rather than fresh full sweeps. Strict replay rejects unexpected method/path/headers/model/body and unmatched calls, and reruns executable assertions against real response/SSE bytes. Synthetic chunking, omitted/zero usage, and identity regressions are labelled synthetic extensions, not represented as observed Fireworks behavior. Final accounting uses presence-aware cumulative usage rather than summing repeated counters; reasoning is included in total output once.

Direct SDK contracts do not certify ADP user/tenant isolation, budgets/guardrails, credential forwarding, persisted spending rollups, native CLI behavior, or end-to-end gateway behavior. Those require the separate gateway acceptance suite.

## Maintenance

Fireworks is a **host**, not a new publisher for OpenAI, Z.ai, MiniMax, DeepSeek, or Moonshot AI models. The existing scheduled/manual reconciliation workflow owns these entries through their creator vendor legs. Its named-source registry, authoritative Fireworks host guide, egress allowlist, package/provider mapping, snapshot/report/guard checks, and CI coverage are the update path; there is no second importer. CI has no Fireworks inference credential. It may refresh existing factual prices/limits/lifecycle, but API/feature/alias expansion requires fresh exact-offering live qualification and review, not automatic widening from the inventory listing.

Refresh the inventory, rates, aliases and lifecycle whenever those authoritative inputs change; regenerate `task catalog:snapshot`, review the JSON diff, and run required acceptance. The SDK model-catalog maintainers own that review. Do not dispatch the public reconciliation workflow with a local credential or give untrusted PR code provider secrets.

Sources (checked 2026-11-04):

- [Fireworks pricing](https://docs.fireworks.ai/serverless/pricing), [serverless modes](https://docs.fireworks.ai/serverless/serverless-modes), [recommended models](https://docs.fireworks.ai/guides/recommended-models), [changelog](https://docs.fireworks.ai/updates/changelog).
- [Messages compatibility/accounting](https://docs.fireworks.ai/tools-sdks/anthropic-compatibility), [Responses](https://docs.fireworks.ai/guides/response-api), [Chat API](https://docs.fireworks.ai/api-reference/post-chatcompletions).
- [GLM 5.3](https://fireworks.ai/models/fireworks/glm-5p3), [Flash](https://fireworks.ai/models/fireworks/glm-5p3-flash), [MiniMax M3](https://fireworks.ai/models/fireworks/minimax-m3), [GPT-OSS 120B](https://fireworks.ai/models/fireworks/gpt-oss-120b).
- [GLM creator release](https://z.ai/blog/glm-5.3), [Flash creator release](https://z.ai/blog/glm-5.3-flash), [GLM limits](https://docs.z.ai/guides/llm/glm-5.3), [Flash limits](https://docs.z.ai/guides/llm/glm-5.3-flash), [GPT-OSS creator specification](https://developers.openai.com/api/docs/models/gpt-oss-120b).
