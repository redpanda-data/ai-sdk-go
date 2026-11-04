# Direct Fireworks contract evidence

This package is catalog-only. These tests call Fireworks' native HTTP APIs directly; they do not introduce an SDK model transport or test an ADP gateway.

## Commands

From the SDK root:

```sh
# Offline evidence, complete active catalog coverage and strict replay.
task -t taskfiles/fireworks.yaml replay

# Explicit, billable full recapture. The credential file is read only inside
# the test process; its contents must never be supplied in command arguments.
task -t taskfiles/fireworks.yaml record -- -fireworks-key-file=/path/to/credential-file -fireworks-budget-usd=1

# Required release acceptance: existing replay plus every active live case.
task -t taskfiles/fireworks.yaml accept -- -fireworks-key-file=/path/to/credential-file -fireworks-budget-usd=1
```

`FIREWORKS_API_KEY` is also supported. Without credentials, live acceptance **fails**, not skips. Each task compiles the test binary and verifies the exact required test names before running it, so an empty `-run` match cannot pass. Required cases cannot skip. Ordinary `go test -short` does not invoke the paid live suite.

Recording is explicit. A failed or interrupted run writes `testdata/contracts/incomplete.json`, never an accepted manifest. Each returned exchange is sanitized and checkpointed **before** protocol assertions or usage parsing. SIGINT/SIGTERM cancel the live request and flush the incomplete report. Transport/read failures have safe error classes; HTTP request objects and credentials are never printed.

## Matrix and replay

The matrix comes from active catalog offerings, exact aliases, declared direct wire APIs, per-binding capabilities and reasoning controls. It includes both stream modes, complete/incomplete results, system and conversation history, forced tools and tool-result replay, JSON/schema output with conflicting unformatted controls, invalid-schema validation, explicit adaptive on/off behavior, advertised efforts and reasoning-history replay, advertised image input, cache warm/repeat, cancellation, and native authentication/model/malformed-request failures. Unknown enum-value probes remain exploratory diagnostics: published valid controls do not imply a promise to reject every unknown value. Unknown features or bindings cannot silently inherit a happy path.

Cache probes send a stable per-case `X-Session-Affinity` header, following [Fireworks' prompt-caching guide](https://docs.fireworks.ai/guides/prompt-caching). A zero cache count is valid native behavior, not an API failure. Qualification separately requires an actually observed final cache-hit count and fresh-input count from current required captures for each exact offering/API/stream-mode binding. An affinity hint never substitutes for those counts.

Cancellation probes use a 16-output-token cap and cancel 25 ms after the HTTP transport confirms the request was written. This proves a real, started client request was aborted; it does **not** prove inference had already started on the server or that the server stopped billing at the same instant. Complete final usage is retained if it arrives. Otherwise the full conservative reservation remains held, and final cancellation billing is unresolved rather than invented.

The evidence records immutable per-case capture versions, test fingerprints, dates, SDK revisions and sanitized request hashes, separately from the current verifier fingerprint. Original failed attempts remain byte-identical source artifacts, referenced by SHA256; corrections retain superseded evidence and are labelled partial corrections, not another full sweep. The capture and current catalog fingerprints are separate only for the explicitly reviewed withdrawal of SDK `SupportedParams`: reconstructing that single withdrawn tuple must reproduce the entire old snapshot hash. No API, capability, fact, numerical limit or price drift is silently accepted. Whitespace and comments do not change the test-token fingerprint; assertions, request construction and other code tokens do. Lifecycle-only clock passage does not invalidate historical captures: every currently active case remains mandatory, and only known, now-retired offerings may supply extra historical cases. Retired offerings are never invoked by the live matrix.

Strict replay rejects unexpected method, path, authorization class, required headers or JSON request shape, and unmatched/unconsumed exchanges. It runs the same direct-protocol assertions and verifies exact sanitized response/SSE bytes, including a separately labelled three-byte chunking extension. Synthetic identity/usage/lifecycle/cancellation-mechanics tests are labelled as synthetic; they are not represented as observed upstream failures or production transport coverage.

Only allowlisted request/response headers are recorded. New diagnostic captures retain actual presence of allowlisted per-request prompt/cache/performance headers and pseudonymize opaque correlation identifiers; authorization, cookies and account-quota/rate-limit values are excluded. Historical captures retained only Content-Type, so they cannot establish effective affinity or replica routing. Streaming performance-header absence is not assumed to mean zero cached usage. Authorization is represented by a safe class, never a value. Opaque response/tool identifiers are consistently replaced with deterministic capture IDs, retaining tool-result linkage. Responses requests always use `store: false`.

`full-matrix-v2-incomplete.json` preserves the complete seven-offering, 680-case sweep and its original assertions. `revalidated-evidence.json` combines compatible retained captures with explicitly versioned corrections and reports hard qualification gaps; it is not a fabricated successful fresh sweep.

`original-incomplete.json` is immutable characterization from the first, superseded harness run. It includes known harness-invalid cases and is **not accepted evidence**. Its retained successful responses remain regression inputs, while missing captures and failures cannot satisfy the current completeness gate.

## Pricing and spend bounds

`published-rates.json` is an independently sourced worksheet, not the matrix inventory. Every active invocation ID requires a worksheet row. Expected costs are calculated directly from its published USD-per-million rates and each response's own raw usage, independently of `pricing.Calculate`. OpenAI-shaped input includes cached reads; Messages input excludes them. Reasoning is a subset of native total output and must not be charged twice.

The live suite is sequential, has bounded request/response sizes and HTTP/suite timeouts, and reserves the complete output-token cap plus conservative input/template headroom **before** each call. Reservations use the larger of independently published and catalog-derived estimates. A successful authoritative usage record reconciles that reservation. Unknown usage keeps its reservation. Budget exhaustion fails incomplete acceptance; it never samples away required cases.

Live execution requires an explicit finite positive `-fireworks-budget-usd` per-run limit. The in-memory request reservations bound that run only; they are not an SDK spending policy, durable billing ledger, or cross-run accounting system. Captures retain their original spend estimates and unresolved reservations as execution provenance, not exact Fireworks invoices.

Published Priority prices are retained independently, but `priority_tier_status=blocked_missing_effective_tier` explicitly withholds Priority requests. An accepted request is not evidence of effective billing-tier attribution.

## Separate release gates

Passing this direct SDK suite does not certify gateway credential injection, operation/model/tenant admission, guardrails, budgets, persisted spending, real Claude Code/Codex behavior, or deployment readiness. Those require the separate cloudv2 gateway replay and live end-to-end suites. Missing final usage on cancellation, unsupported Priority attribution and any incomplete direct matrix remain explicit limitations; they must not be presented as successful billing evidence.
