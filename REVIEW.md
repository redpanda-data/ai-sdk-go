# Review guidelines

These rules apply to every pull request. The last section adds checks for the
reconcile bot's catalog PRs.

Report every diff-introduced, impactful defect, not preferences. Each finding
needs a caller or input, consequence, and source location. A claim about
behavior needs a `file:line` citation, not an inference from naming.

## Severity

Important means the change breaks a caller, the wire format, persisted
sessions, cost accounting, or a trust boundary. Other real defects are Nits.
Style-only comments and optional future work are not findings.

## Do not report

- What CI enforces: golangci-lint (formatting, imports, forbidden functions),
  license headers, govulncheck and OSV, and catalog snapshot freshness.
- Generated code under `store/session/kvstore/proto/gen/`. Review the `.proto`.
- `catalog/snapshot.json` line by line. Review its delta against the
  `models.go` change that produced it, and flag any delta no source change
  explains.

## SDK failure modes

- Verify provider-native model IDs, regions, capabilities, and request envelopes.
  A first-party mapping does not prove that Bedrock or Vertex sends the same
  controls. Catalog flags must be exercised through the serving provider.
- Preserve opaque reasoning signatures, redacted payloads, tool IDs, and ordering
  across plain responses, streaming, persistence, and replay. Never replay a
  different provider's opaque payload or export it through telemetry.
- Check pricing selectors, context boundaries, cache-write accounting, regional
  premiums, negotiated overrides, and unknown-model behavior against independent
  provider evidence. Do not infer partner prices from first-party prices. When
  no source is reachable, say so instead of reporting a finding.
- Preserve error classification, cancellation, stream completion, and retry
  bounds. Request rewrites must retain the rewritten body on redirects and
  retries, without forwarding ambient credentials to another provider.
- Every tool call in session history gets exactly one result before the next
  model call, including after cancellation and crash recovery. Recovery answers
  an incomplete call as interrupted and never re-runs it.
- Tool schemas pass through each provider's adapter. OpenAI and openaicompat
  send strict mode, which makes optional properties required and nullable, so a
  tool must accept `null` for them.
- Inbound A2A messages, MCP server output, and fetched web content are
  untrusted. Flag changes that let them inject non-user turns, widen the
  environment passed to stdio MCP servers, or weaken the webfetch SSRF guard.
- Exported identifiers and documented behavior are the public contract. Flag
  breaking changes, and behavior changes for existing callers that the PR
  summary doesn't state or no test pins.

## Evidence

- A regression test must fail before the fix for the intended reason; a compile
  failure is not that proof.
- A fake HTTP peer proves wire behavior, not live provider acceptance. Flag a PR
  that counts a skipped integration test as a passed live check.
- A live test takes its model from a `<provider>test` setting, which
  `TestLiveTestModelsAreCurrent` dates against the catalog, or calls
  `testutil.SkipIfRetired`. Flag a model ID written into a live test: it goes
  red when the vendor shuts the model down.
- Flag hand edits to generated files, and source-text tests that stand in for a
  reviewed snapshot delta.
- Keep unresolved owner decisions visible. Do not substitute an agent's
  assertion for hosted checks, independent approval, or permission to merge.

## Bot catalog PRs

This section applies only to PRs from `vbotbuildovich`.

The reconcile-models workflow opens draft PRs from `vbotbuildovich` that change
the model catalog, following `.github/reconcile-models/RULES.md`. The PR job
checked the report's format and that every source is a named source, and ran
the tests without secrets. It didn't check the values.

- Check each row in "Changes" against the page it cites, and each offering in
  "Changed with no row" in the snapshot diff, since no row explains it. Mark the
  PR ready only when every value matches.
- Settle each question in "Resolve before merging": fix it on the base branch
  or record a decision. You may also merge with a question open; the next run
  asks it again.
- In "Held by decisions", each value should still match its entry in
  `.github/reconcile-models/decisions.yaml`.
- Approve a move in "Routing changes" only if every caller of that bare name
  can take the new profile and its price.
- Read the files in "Code changed" before marking the PR ready: that runs them
  with secrets.
- When `TestLiveTestModelsAreCurrent` fails once the PR is ready, push the swap
  it prints onto this same PR, so the date and the swap merge together. Later
  runs then stop updating the PR until it merges or closes.
