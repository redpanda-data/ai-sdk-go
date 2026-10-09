#!/usr/bin/env bash
# Applies the reconcile-models agent's patch to a clean checkout and checks
# it with no model involved. Used by the workflow's PR job; runs locally too.
# Usage: pr-job.sh PROVIDER OUT_DIR
set -euo pipefail

provider="${1:?usage: pr-job.sh PROVIDER OUT_DIR}"
out="$(cd "${2:?usage: pr-job.sh PROVIDER OUT_DIR}" && pwd)"
patch="$out/patch.diff"
report="$out/report.json"
models="providers/${provider}/models.go"
facts="catalog/facts_data.go"
hosts="$(jq -r --arg p "$provider" '.[$p] // empty | join(",")' .github/reconcile-models/named-sources.json)"
emit() { if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "$1" >>"$GITHUB_OUTPUT"; fi; }

[[ -n "$hosts" ]] || { echo "pr-job: no named sources for $provider" >&2; exit 1; }
[[ -s "$report" ]] || { echo "pr-job: no report from the agent job" >&2; exit 1; }

# A clean run lists nothing. If every offering was skipped because no named
# source could be read, the agent checked nothing, so fail instead of passing.
if jq -e '(.changes | length) == 0 and (.needs_human | length) == 0 and (.skipped | length) > 0
          and all(.skipped[]; .reason == "not_on_named_sources")' "$report" >/dev/null; then
  echo "pr-job: the agent read no named source (every offering skipped as not_on_named_sources)" >&2
  exit 1
fi

if [[ ! -s "$patch" ]]; then
  go run ./cmd/catalog-report -report "$report" -provider "$provider" -hosts "$hosts" -out "$out/body.md"
  emit changed=false
  echo "pr-job: no changes"
  exit 0
fi

# Reject file creation, deletion, renames and mode changes, then any path outside the allowlist.
if git apply --summary "$patch" | grep -E '^ (create|delete|rename|mode)'; then
  echo "pr-job: the patch creates, deletes or renames files" >&2
  exit 1
fi
while IFS=$'\t' read -r _ _ path; do
  if [[ "$path" != "$models" && "$path" != "$facts" ]]; then
    echo "pr-job: the patch touches $path" >&2
    exit 1
  fi
done < <(git apply --numstat "$patch")

git apply --check "$patch"
git show "HEAD:$models" >"$out/base-models.go"
git show "HEAD:$facts" >"$out/base-facts.go"
git apply "$patch"
go run ./cmd/catalog-guard -base "$out/base-models.go" -patched "$models"
go run ./cmd/catalog-guard -base "$out/base-facts.go" -patched "$facts"

# A patch must come with reported changes, so the PR body never shows less than the diff.
if ! jq -e '.changes | length > 0' "$report" >/dev/null; then
  echo "pr-job: the patch changes files but the report lists no changes" >&2
  exit 1
fi

task catalog:snapshot
task catalog:check
task test:unit
# The agent job runs with hooks off, so lint here. Only the patched lines (--new),
# and without --fix, which would change files after the checks.
task install:golangci-lint
GOROOT="$(go env GOROOT)" .build/.bin/golangci-lint run --new --timeout 10m "./providers/${provider}/..." ./catalog/...
task license:check

unexpected="$(git diff --name-only | grep -vxF -e "$models" -e "$facts" -e catalog/snapshot.json || true)"
[[ -z "$unexpected" ]] || { echo "pr-job: unexpected changes: $unexpected" >&2; exit 1; }

git diff -- catalog/snapshot.json >"$out/snapshot.diff"
go run ./cmd/catalog-report -report "$report" -provider "$provider" -hosts "$hosts" -diff "$out/snapshot.diff" -out "$out/body.md"
emit changed=true
echo "pr-job: patch accepted"
