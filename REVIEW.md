# Review guidelines

Review diff-introduced defects, not preferences. Report at most five actionable
findings, each with a caller or input, consequence, and source location. Keep
style-only nits out of the review.

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

## Generated files and delivery

Review authored owners rather than hand-editing protobuf output, generated
connectors, or files marked generated. Regenerate `catalog/snapshot.json` through
its command. Require reviewed snapshot deltas, not source-text proxy tests.

Keep unresolved owner decisions visible. Do not substitute an agent's assertion
for hosted checks, independent approval, or permission to merge.
