# Review guidelines

Report every diff-introduced, impactful defect, not preferences. Each finding
needs a caller or input, consequence, and source location. Omit style-only nits
and optional future work.

Each section starts with the PRs it applies to. Apply a section only when the PR
matches it and ignore it otherwise. A PR can match several sections.

## Evidence

Applies to every PR.

- Prefer a regression at the public API or wire seam. Confirm it fails before
  the fix for the intended reason; a compile failure is not that proof. Use
  table-driven tests and parallel cases where state is isolated.
- Run the affected Go packages and the linter version declared in
  `.github/workflows/golangci-lint.yaml`. The global or Taskfile version may differ.
- State exact commands, tested versions, unavailable credentials, and skipped
  checks. A fake HTTP peer proves wire behavior, not live provider acceptance.
  Never count a skipped integration test as a passed live check.

## Generated files and delivery

Applies to every PR.

Review authored owners rather than hand-editing protobuf output, generated
connectors, or files marked generated.

Keep unresolved owner decisions visible. Do not substitute an agent's assertion
for hosted checks, independent approval, or permission to merge.

## Model catalog and pricing

Applies when the PR changes model metadata: `catalog/`, `pricing/`,
`providers/*/models.go`, Bedrock profiles and routing
(`providers/bedrock/families.go`, `providers/bedrock/cross_region.go`), or
Vertex locations (`providers/vertex/locations.go`).

- Verify provider-native model IDs, regions, locations, and capabilities
  against the vendor's model card or docs. Catalog flags must be exercised
  through the serving provider.
- Moving a Bedrock source region to another profile moves the bare-name
  callers in that region. Each must be able to take the new profile and its
  price.
- Check pricing selectors, context boundaries, cache-write accounting, regional
  premiums, negotiated overrides, and unknown-model behavior against independent
  provider evidence. Do not infer partner prices from first-party prices.
- Run `task catalog:snapshot` and `task catalog:check`; review unrelated
  snapshot drift. Regenerate `catalog/snapshot.json` through its command, and
  require reviewed snapshot deltas, not source-text proxy tests.

## Provider runtime

Applies when the PR changes how requests or messages are built, sent,
streamed, retried, persisted, converted, or replayed: `llm/`, `agent/`,
`runner/`, `store/`, `adapter/`, `plugins/`, and files under `providers/` other
than the model metadata files above.

- Verify request envelopes per serving provider. A first-party mapping does not
  prove that Bedrock or Vertex sends the same controls.
- Preserve opaque reasoning signatures, redacted payloads, tool IDs, and ordering
  across plain responses, streaming, persistence, and replay. Never replay a
  different provider's opaque payload or export it through telemetry.
- Preserve error classification, cancellation, stream completion, and retry
  bounds. Request rewrites must retain the rewritten body on redirects and
  retries, without forwarding ambient credentials to another provider.

## Live tests

Applies when the PR adds or changes a test that calls a live provider.

A live test takes its model from a `<provider>test` setting, which
`TestLiveTestModelsAreCurrent` dates against the catalog, or calls
`testutil.SkipIfRetired` when it tests one model on purpose. A model ID written
into a live test goes red without warning when the vendor shuts it down.

## Dependencies and public API

Applies when the PR changes a `go.mod` or `go.sum` in any module, or an exported
identifier.

Build affected examples. Run `task security` for dependency and
security-sensitive work.

## Bot catalog PRs

Applies only to PRs from `vbotbuildovich`, in addition to the model catalog
section above.

The reconcile-models workflow opens draft PRs from `vbotbuildovich` that change
the model catalog, following `.github/reconcile-models/RULES.md`. No automated
check compares a value with its source: only the agent did. For its own
commits, the bot checks the report's format, each row's rule and that each
source is on a named host. It also checks that only allowed files change, and
runs unit tests and lint without secrets. A commit a person pushes gets fewer
checks: the run builds it, checks the catalog and tests the packages whose Go
files it changes.

Words the PR body uses:

- An offering is one model as one host sells it: its ID there, its price and
  its limits.
- A copy is another offering of a model the catalog already has: on another
  host, or in another region, such as a Bedrock `us.` profile.
- An open question is a case no rule settles, one a rule hands to a person, or
  a region the catalog doesn't offer yet. The bot changes nothing for it.
- A region question asks whether the catalog offers a region at all. Until
  someone decides, the bot leaves that region's copies out.
- A rule draft is the agent's wording of a rule that would settle a case seen on
  two or more offerings. It does nothing until a person adds it to RULES.md.
- A reused commit is a vendor's commit from an earlier agent run, applied again
  with no model because it still makes exactly the same change. Its values were
  checked against the pages as they were on that run's day.
- A commit not from the bot is one a person pushed onto the bot PR. Each run
  rebuilds the branch and puts it back on top while it still applies exactly
  and passes its build and tests. If it doesn't, or the PR gets another merge
  commit or a `.github/` change, the run stops updating the PR and comments
  why. Later runs don't read PR comments.

To review one:

- Check each row in "Changes" against the page it cites, and each offering in
  "Changed with no row" in `catalog/snapshot.json` under Files changed, since no
  row explains it. Mark the PR ready only when every value matches.
- Settle each question in "Open questions" as its Decide line says, in a PR to
  the base branch. You may also merge with a question open; the next run asks
  it again.
- In "Held by decisions", each value should still match its entry in
  `.github/reconcile-models/decisions.yaml`.
- Approve a move in "Routing changes" only if every caller of that bare name
  can take the new profile and its price.
- Read the files under "Code changed" as code before marking the PR ready: that
  runs them with secrets.
- When `TestLiveTestModelsAreCurrent` fails once the PR is ready, push the swap
  it prints onto this same PR, so the date and the swap merge together. Later
  runs carry it onto each rebuild while it still applies exactly.
