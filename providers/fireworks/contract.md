# Fireworks recorded contract tests

The native HTTP and SDK replay suites use recorded upstream exchanges as regression fixtures. They check requests, responses, streaming, errors and token accounting without calling Fireworks. They do not certify the current backend or require recordings for every catalog entry.

## Offline tests

```sh
task fireworks:replay
go test -short ./providers/fireworks
```

Replay exercises the cases present in the fixtures, including historical retired offerings. It rejects unexpected methods, paths, authorization classes, headers or JSON bodies, unmatched/unconsumed calls, and changed response/SSE bytes. The same assertions also run with three-byte response chunking. Failed recordings cannot be promoted by deleting failing assertions or cases.

There are no catalog, test-source, request or whole-file fingerprint gates. Old fixture provenance fields are ignored; capture dates and SDK revisions are descriptive metadata. A catalog date, limit or price correction does not require re-recording HTTP exchanges. Changing request construction or response handling is checked by the executable tests themselves.

New models and documented capabilities can be added through the normal reviewed catalog workflow without live calls or new recordings. Missing recordings mean missing live-derived coverage, not that the model is unavailable. Existing fixtures must still pass. Synthetic wire tests remain useful for request mapping that has not been recorded.

## Optional live checks and recording

Live calls are local-only, explicitly opt-in and billable. CI has no Fireworks inference credential. A missing key or finite positive budget fails an explicitly requested native live run; ordinary offline tests never require either.

```sh
# Check one exact active model, retaining the existing fixtures unchanged.
task fireworks:accept -- -fireworks-model=accounts/fireworks/models/glm-5p3-flash \
  -fireworks-key-file=/path/to/credential-file -fireworks-budget-usd=1

# Record one model to a separate file for review.
task fireworks:record -- -fireworks-model=accounts/fireworks/models/glm-5p3-flash \
  -fireworks-key-file=/path/to/credential-file -fireworks-budget-usd=1 \
  -fireworks-record-output=/tmp/fireworks-capture.json
```

Omitting `-fireworks-model` selects all active catalog offerings for the native live suite. The matrix covers supported direct APIs and their known feature cases in both stream modes. It is a test generator, not a catalog admission policy. A cache miss is valid behavior; unobserved hit/fresh branches are reported rather than requiring repeated paid calls until they appear.

Recording writes sanitized checkpoints to `<output>.incomplete.json` before parsing responses. Only a successful run writes the requested output file. Use a separate output path, review the exchanges, and manually promote new-model fixtures to `testdata/contracts/recordings/*.json`; replay loads these alongside the retained base fixture. For an existing case, update its fixture rather than adding a duplicate. Never overwrite a historical failed attempt to claim it succeeded.

Credentials are read inside the test process and used only in the HTTP authorization header. Never put their contents in arguments, logs, fixtures or reports. Only allowlisted headers are recorded; opaque IDs are consistently pseudonymized to preserve tool-result linkage. Responses requests send `store: false`.

Requests run sequentially with bounded bodies, output caps and timeouts. Each request reserves conservative input/output cost before sending, using current catalog prices. Authoritative final usage reconciles the reservation; absent final usage retains it. An explicit per-run budget bounds estimates for that run, not vendor invoices or production spending. Cancellation proves a written request was aborted, not that upstream inference or billing stopped immediately.

## Pricing

The catalog is the only model-rate table. Contract tests check native usage semantics and disjoint SDK buckets: OpenAI-shaped input includes cached reads, Messages input excludes them, and reasoning belongs to total output exactly once. Fixed-rate synthetic tests cover accounting independently of mutable vendor rates. Any replay cost calculation uses current catalog rates; historical capture spend estimates remain historical metadata, not exact invoices.

Published Priority rates do not establish effective-tier attribution. Priority requests remain withheld where the runtime cannot determine their billing tier.

## Scope

Native HTTP replay is separate from actual SDK `Generate`/`GenerateEvents` replay; see [README.md](README.md#sdk-runtime-tests). Neither certifies gateway credential injection, tenant isolation, guardrails, persisted spending or real CLI clients. Those require separate end-to-end tests.

The retained `revalidated-evidence.json` combines historical captures and corrective runs. Its name and old verifier metadata do not impose a revalidation workflow. `original-incomplete.json` remains characterization of an earlier harness, including known harness-invalid cases; it is not a successful live run.
