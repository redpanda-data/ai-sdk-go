# Review guidelines

Report every diff-introduced, impactful defect, not preferences. Each finding
needs a caller or input, consequence, and source location. Omit style-only nits
and optional future work.

## SDK failure modes

- Verify provider-native model IDs, regions, capabilities, and request envelopes.
  A first-party mapping does not prove that Bedrock or Vertex sends the same
  controls. Catalog flags must be exercised through the serving provider.
- Preserve opaque reasoning signatures, redacted payloads, tool IDs, and ordering
  across plain responses, streaming, persistence, and replay. Never replay a
  different provider's opaque payload or export it through telemetry.
- Check pricing selectors, context boundaries, cache-write accounting, regional
  premiums, negotiated overrides, and unknown-model behavior against independent
  provider evidence. Do not infer partner prices from first-party prices.
- Preserve error classification, cancellation, stream completion, and retry
  bounds. Request rewrites must retain the rewritten body on redirects and
  retries, without forwarding ambient credentials to another provider.

## Evidence

- Prefer a regression at the public model API or wire seam. Confirm it fails
  before the fix for the intended reason; a compile failure is not that proof.
  Use table-driven tests and parallel cases where state is isolated.
- Run the affected Go packages and the linter version declared in
  `.github/workflows/golangci-lint.yaml`. The global or Taskfile version may differ.
- For catalog changes, run `task catalog:snapshot` and `task catalog:check`; review
  unrelated snapshot drift. Build affected examples after public API or dependency
  changes. Run `task security` for dependency and security-sensitive work.
- State exact commands, tested versions, unavailable credentials, and skipped
  checks. A fake HTTP peer proves wire behavior, not live provider acceptance.
  Never count a skipped integration test as a passed live check.
- A live test takes its model from a `<provider>test` setting, which
  `TestLiveTestModelsAreCurrent` dates against the catalog, or calls
  `testutil.SkipIfRetired` when it tests one model on purpose. A model ID written
  into a live test goes red without warning when the vendor shuts it down.

## Generated files and delivery

Review authored owners rather than hand-editing protobuf output, generated
connectors, or files marked generated. Regenerate `catalog/snapshot.json` through
its command. Require reviewed snapshot deltas, not source-text proxy tests.

Keep unresolved owner decisions visible. Do not substitute an agent's assertion
for hosted checks, independent approval, or permission to merge.

## Bot catalog PRs

The reconcile-models workflow opens draft PRs from `vbotbuildovich` that change
the model catalog, following `.github/reconcile-models/RULES.md`. The PR job
checked the report's format and that every source is a named source, and ran
the tests without secrets. It didn't check the values.

- Check each row in "Changes" against the page it cites, and each offering in
  "Changed with no row" in the snapshot diff, since no row explains it. Mark the
  PR ready only when every value matches.
- Settle each question in "Resolve before merging" before you merge: fix it on
  the base branch or record a decision.
- In "Held by decisions", each value should still match its entry in
  `.github/reconcile-models/decisions.yaml`.
- Approve a move in "Routing changes" only if every caller of that bare name
  can take the new profile and its price.
- Read the files in "Code changed" before marking the PR ready: that runs them
  with secrets.
- When `TestLiveTestModelsAreCurrent` fails once the PR is ready, push the swap
  it prints onto this same PR, so the date and the swap merge together. Later
  runs then stop updating the PR until it merges or closes.
