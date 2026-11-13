# Fireworks provider

This package supplies Fireworks serverless `llm.Model` instances, `catalog.Provider`, and published token prices under the provider key `fireworks`. It reuses the existing Messages, Responses, and Chat Completions implementations; it is not a gateway proxy, Claude Code/Codex launcher, or spending pipeline. Accepting Anthropic Messages does not make an offering Anthropic-published.

```go
provider, err := fireworks.NewProvider(ctx,
    fireworks.WithAPIKey(key), // Fireworks key or a gateway service-account token
    // fireworks.WithBaseURL("https://gateway.example.com/fireworks/v1"),
    // fireworks.WithHTTPClient(client),
)
if err != nil {
    return err
}
model, err := provider.NewModel(fireworks.ModelGLM53Flash,
    fireworks.WithMaxTokens(4096),
    fireworks.WithReasoningEffort(fireworks.ReasoningEffortLow),
)
```

`NewProvider(ctx)` and the provider's zero value still support credential-free catalog discovery. Credentials are explicit; the provider never reads environment variables or credential files. `NewModel` requires a bearer credential and rejects unregistered IDs. A custom base URL is the inference API root ending in `/v1`; gateway prefixes are preserved.

The factory uses exactly the resolved offering's `WireAPIs[0]`, in the catalog's normalized order, without reordering, preference knobs, or fallback. An empty, unsupported, or undeclared first binding is an error even if a later binding is usable. All seven current offerings select Messages/direct. The selected binding determines capabilities and reasoning controls, not the offering-wide union.

**Runtime image restriction:** native catalog vision support remains true where documented, but constructed SDK models report `Capabilities().Vision == false`: `llm.Part` has no image input representation. This is an SDK limitation, not an upstream unsupported claim; SDK vision is not qualified.

## Scope and identity

Use exact invocation IDs. `catalog.WireAPI` is the existing shared protocol/transport contract; each supported Fireworks binding uses the `direct` transport. API-specific capabilities and reasoning controls are explicit `WireAPISupport` records, not inferred from provider names, creators, model prefixes, or the model-level capability superset. Missing API feature records mean unknown, not full feature parity. Messages has schema-based structured output but no separate schemaless JSON mode.

Catalog `Constraints.SupportedParams` remains intentionally empty: it is not a provider-wide SDK control guarantee. Each constructed model gets a separate, narrow admission list for its selected API: `max_tokens`, `temperature`, advertised `reasoning_effort` values, and `thinking` only where adaptive reasoning is declared. The catalog is not mutated. Nil is not a tri-state validator and is never supports-all; catalog/UI consumers must not project absent SDK admission into an upstream supports-none or supports-all claim.

`WithMaxTokens` defaults to a bounded 4096 tokens (including reasoning), capped at the offering's output limit. Temperature is validated against the offering's range. `WithReasoningEffort` accepts only the selected binding's exact list; DeepSeek `none` maps to disabled thinking on Messages/Chat. MiniMax exposes `WithThinking(true)` for adaptive thinking and `WithThinking(false)` to disable it, without invented budgets or Claude-specific controls. MiniMax Messages with explicit `WithThinking(false)` reports `StructuredOutput == false` and rejects schema requests before HTTP: native testing observed broken schema enforcement with that combination. Omit the thinking override or use adaptive thinking for schema output; both configurations passed actual SDK tests in both modes. The native catalog/API schema capability remains true. Foreign `Request.Options` are rejected rather than silently ignored; configure these controls at model construction. Responses always sends `store: false`. No automatic Anthropic cache markers are added; authoritative native cache usage is preserved. Priority, speed, thinking budgets, top-p, stop controls, and provider-hosted tool search are not exposed by this runtime.

Resolution is exact, including exact authored aliases. An unlisted date suffix, deployment, router, shorthand, or `*-latest` alias must not inherit the base model's API support or price. No moving aliases are currently registered. Router variants, US-only capacity, custom/private/fine-tuned deployments, batch, embeddings, and decision-only models are outside this catalog's supported contract. Messages support does not advertise native count-tokens: the observed `/inference/v1/messages/count_tokens` endpoint returned 404. Hosted `ToolSearchProtocol` remains empty; ordinary tools and Fireworks' client-side deferred-tool compatibility do not establish provider-hosted tool execution.

Initial inventory (checked 2026-11-04). This historical table describes the original rollout; the catalog is the current source for entries and rates. Recordings cover observed behavior, not every catalog model:

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

## SDK runtime tests

The original local SDK run exercised `NewProvider(...).NewModel(...)`, `Generate`, and `GenerateEvents` for all seven exact offerings through the default Messages binding. It covers system/history, forced tools and tool-result replay, observable reasoning and reasoning-history replay, native schema enforcement against a conflicting plain control, output-limit termination, and authoritative SDK/native usage. Captured native exchanges are replayed offline; replay is not represented as a new live result. The factory's other two protocol branches are separately covered at the wire seam, not claimed as new all-model SDK live coverage.

The initial 86-request SDK batch passed all native usage comparisons and the complete feature sweep for six models, but its two MiniMax schema cases failed upstream with explicit disabled thinking. Those failed native exchanges remain immutable in `testdata/sdk-runtime/messages-live.json`, with an explicit failed-run outcome alongside them; `native_verified` means usage/price agreement, not feature success. Four separately captured corrective SDK schema calls passed with default/omitted or adaptive thinking, in both modes. The restriction above rejects the failed configuration without silently changing it. These corrections are not represented as a new full sweep. The original recorder repeatedly sanitized request tool IDs; offline conversation replay applies an explicitly labelled derived ID correction for that historical defect while retaining the original capture and every response byte unchanged.

SDK live calls are explicitly opt-in and local-only. They are not dispatched by CI and must not be given a Fireworks credential through workflows. SDK tests do not include images.

```sh
# Build before reading credentials; execute locally only.
go test -c -o /tmp/fireworks-sdk-live.test ./providers/fireworks
(cd providers/fireworks && /tmp/fireworks-sdk-live.test \
  -test.run='^TestFireworksSDKLiveFirstBinding$' -test.v \
  -fireworks-sdk-live -fireworks-sdk-key-file=/absolute/path/to/credential-file)
# Optional: -fireworks-sdk-cache adds at most three bounded cache probes.
```

The SDK test client enforces one upstream HTTP attempt per invocation (including vendor-SDK retries), bounds each request body and output cap, and checkpoints sanitized native exchanges into a unique persistent run directory, never over the retained fixtures. `-fireworks-sdk-output-dir` selects the parent directory; otherwise the system temporary directory is used. The logged `*-incomplete.json` artifact requires explicit review and manual promotion. Repeated checkpoints sanitize fresh copies, preserving stable IDs. Approve a conservative paid-call bound before opting in; the test client is not a production spending policy.

## Recorded native API tests

The three-protocol native tests are offline regression tests, separate from SDK replay. They replay recorded requests and responses rather than requiring every catalog offering to have live evidence. No catalog, test-source or fixture fingerprint gates are used.

```sh
task fireworks:replay
```

Live calls and recording are optional, local-only and billable. Select one model with `-fireworks-model`; recording requires a separate `-fireworks-record-output` path and explicit finite positive `-fireworks-budget-usd`. See [contract.md](contract.md) for commands, fixture promotion and spend safeguards. CI never needs a Fireworks key.

## Maintenance

Fireworks is a **host**, not a new publisher for OpenAI, Z.ai, MiniMax, DeepSeek, or Moonshot AI models. The reconciliation workflow owns these entries through their creator vendor legs. Add documented serverless models and features through the normal reviewed catalog workflow; live recordings are additional coverage, not an admission requirement. Existing replay fixtures must still pass. Keep SDK representability and request-mapping limits separate from native metadata.

The catalog is the only model-rate table. Update authoritative rates/limits/lifecycle, regenerate `task catalog:snapshot`, review the JSON diff, and run unit/replay tests. Numerical price changes do not invalidate recorded HTTP responses. Do not give untrusted PR code provider secrets or dispatch public reconciliation workflows with local credentials.

Sources (checked 2026-11-04):

- [Fireworks pricing](https://docs.fireworks.ai/serverless/pricing), [serverless modes](https://docs.fireworks.ai/serverless/serverless-modes), [recommended models](https://docs.fireworks.ai/guides/recommended-models), [changelog](https://docs.fireworks.ai/updates/changelog).
- [Messages compatibility/accounting](https://docs.fireworks.ai/tools-sdks/anthropic-compatibility), [Responses](https://docs.fireworks.ai/guides/response-api), [Chat API](https://docs.fireworks.ai/api-reference/post-chatcompletions).
- [GLM 5.3](https://fireworks.ai/models/fireworks/glm-5p3), [Flash](https://fireworks.ai/models/fireworks/glm-5p3-flash), [MiniMax M3](https://fireworks.ai/models/fireworks/minimax-m3), [GPT-OSS 120B](https://fireworks.ai/models/fireworks/gpt-oss-120b).
- [GLM creator release](https://z.ai/blog/glm-5.3), [Flash creator release](https://z.ai/blog/glm-5.3-flash), [GLM limits](https://docs.z.ai/guides/llm/glm-5.3), [Flash limits](https://docs.z.ai/guides/llm/glm-5.3-flash), [GPT-OSS creator specification](https://developers.openai.com/api/docs/models/gpt-oss-120b).
