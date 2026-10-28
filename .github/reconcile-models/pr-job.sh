#!/usr/bin/env bash
# Applies the reconcile-models agent's patch to a clean checkout and checks
# it with no model involved. Used by reconcile-vendor.yml's check job; runs
# locally too.
# Patched code runs only inside a sandbox with no network, an empty
# environment and a read-only checkout (only catalog/snapshot.json, which it
# regenerates, is writable), so code the agent wrote can't reach secrets or
# change what runs after the checks.
# Usage: pr-job.sh PROVIDER OUT_DIR
set -euo pipefail

provider="${1:?usage: pr-job.sh PROVIDER OUT_DIR}"
out="$(cd "${2:?usage: pr-job.sh PROVIDER OUT_DIR}" && pwd -P)"
repo="$(pwd -P)"
facts="catalog/facts_data.go"
# A run owns its vendor's models in every provider package that sells them
# (e.g. Claude in anthropic, vertex and bedrock), and reads those docs sites.
sources=.github/reconcile-models/named-sources.json
owned="$(jq -r --arg p "$provider" '.[$p].providers // empty | join("|")' "$sources")"
emit() { if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "$1" >>"$GITHUB_OUTPUT"; fi; }

[[ "$owned" =~ ^[a-z0-9|]+$ ]] || { echo "pr-job: no named sources for $provider" >&2; exit 1; }
[[ -s "$out/report.json" ]] || { echo "pr-job: no report from the agent job" >&2; exit 1; }

# Work from private copies of the inputs and the tools, outside the checkout
# and read-only to the sandbox, so patched code can't rewrite them.
priv="$(mktemp -d)"
priv="$(cd "$priv" && pwd -P)"
trap 'rm -rf "$priv"' EXIT
cp "$out/report.json" "$priv/report.json"
cp .github/reconcile-models/decisions.yaml .github/reconcile-models/RULES.md "$sources" "$priv/"
if [[ -f "$out/patch.diff" ]]; then cp "$out/patch.diff" "$priv/patch.diff"; else : >"$priv/patch.diff"; fi
report="$priv/report.json"
patch="$priv/patch.diff"
go build -o "$priv/catalog-report" ./cmd/catalog-report
go build -o "$priv/catalog-guard" ./cmd/catalog-guard
# The PR body links repo files at the base commit.
base_url=""
if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
  base_url="${GITHUB_SERVER_URL:-https://github.com}/$GITHUB_REPOSITORY/blob/$(git rev-parse HEAD)"
fi
# The body reads the report, and the catalogue before and after the patch
# once both are there, from $priv: the one vendor's folder.
render() {
  "$priv/catalog-report" -vendor "$provider=$priv" -sources "$priv/named-sources.json" \
    -rules "$priv/RULES.md" -decisions "$priv/decisions.yaml" -base-url "$base_url" -base-branch "${BASE:-main}" -out "$out/body.md" "$@"
}

# A clean run lists nothing. If every offering was skipped because no named
# source could be read, the agent checked nothing, so fail instead of passing.
if jq -e '(.changes | length) == 0 and (.needs_human | length) == 0 and (.skipped | length) > 0
          and all(.skipped[]; .reason == "not_on_named_sources")' "$report" >/dev/null; then
  echo "pr-job: the agent read no named source (every offering skipped as not_on_named_sources)" >&2
  exit 1
fi

if [[ ! -s "$patch" ]]; then
  render
  # The agent said it changed values its patch doesn't contain: fail, so the
  # drift doesn't hide behind a green run.
  if jq -e '.changes | length > 0' "$report" >/dev/null; then
    echo "pr-job: the report lists changes but the patch is empty" >&2
    exit 1
  fi
  emit changed=false
  echo "pr-job: no changes"
  exit 0
fi

# Allowed: any Go file in the owned providers' packages (new files too) and
# catalog/facts_data.go. Workflows, dependencies and other packages stay out of
# reach: they would take effect before review, or for every provider.
go_re="^providers/(${owned})/[a-z0-9_]+\.go$"
allowed() { [[ "$1" == "$facts" || "$1" =~ $go_re ]]; }

summary="$(git apply --summary "$patch")"
refused="$(grep -E '^ (delete|rename|copy|mode)' <<<"$summary" | sed 's/^ //' | paste -s -d ';' - || true)"
if [[ -n "$refused" ]]; then
  echo "pr-job: the patch deletes, renames or copies files, or changes a mode: $refused" >&2
  exit 1
fi
while read -r _ _ _ path; do
  if [[ ! "$path" =~ $go_re ]]; then
    echo "pr-job: the patch creates $path" >&2
    exit 1
  fi
done < <(grep -E '^ create mode' <<<"$summary" || true)

patched=()
while IFS=$'\t' read -r _ _ path; do
  if ! allowed "$path"; then
    echo "pr-job: the patch touches $path" >&2
    exit 1
  fi
  patched+=("$path")
done < <(git apply --numstat "$patch")

# Everything that needs the network happens now, before any patched code runs.
task install:go install:tparse install:golangci-lint install:addlicense
go mod download

# Patched code runs only in this sandbox, which also keeps OUT_DIR and the
# private copies read-only.
SANDBOX_RO=("$out" "$priv")
# shellcheck source=sandbox.sh
source "$(dirname "$0")/sandbox.sh"
if ! sandbox_check; then
  echo "pr-job: the sandbox check failed" >&2
  exit 1
fi

git apply --check "$patch"
# The catalogue and Bedrock routes before the patch, for each change's effect
# note and the body's routing changes. The base holds earlier vendors'
# unreviewed commits in a run that covers several, so it runs in the sandbox too.
git show HEAD:catalog/snapshot.json >"$priv/base-snapshot.json"
sandbox go run ./cmd/catalog-routes >"$priv/base-routes.txt"
for path in "${patched[@]}"; do
  if git cat-file -e "HEAD:$path" 2>/dev/null; then git show "HEAD:$path" >"$priv/base-${path//\//_}"; fi
done
git apply "$patch"

# The data-only check labels files instead of gating them: tests, new files
# and any change that is more than data are listed in the PR body as code.
code=()
for path in "${patched[@]}"; do
  base="$priv/base-${path//\//_}"
  if [[ "$path" == *_test.go || ! -f "$base" ]] || ! "$priv/catalog-guard" -base "$base" -patched "$path"; then
    echo "pr-job: $path changes code; listed for review" >&2
    code+=("$path")
  fi
done

# A patch must come with reported changes, so the PR body never shows less than the diff.
if ! jq -e '.changes | length > 0' "$report" >/dev/null; then
  echo "pr-job: the patch changes files but the report lists no changes" >&2
  exit 1
fi

# Test the packages the patch can affect: the patched ones and everything that
# imports them. Tests needing Docker or the network can't run in the sandbox;
# the full suite runs in the PR's own CI. -count=1: never reuse cached results.
lint_pkgs="$(printf './providers/%s/... ' ${owned//|/ })"
affected="$(affected_packages "./providers/$provider" ./catalog)"

# The only place patched code runs. The agent job runs with hooks off, so lint
# here: only the patched lines (--new), and without --fix, which would change
# files after the checks.
sandbox bash -c "
  set -euo pipefail
  task catalog:snapshot
  task catalog:check
  go test -short -count=1 $affected
  .build/.bin/golangci-lint run --new --timeout 10m $lint_pkgs ./catalog/...
  task license:check"

# Where bare Bedrock names go now: patched code, so in the sandbox. Its
# output is checked line by line before the body uses it.
sandbox go run ./cmd/catalog-routes >"$priv/routes.txt"

# Nothing but the patched files and the snapshot may have changed, tracked or
# untracked (OUT_DIR excepted when it sits inside the checkout).
out_rel=""
if [[ "$out" == "$repo"/* ]]; then out_rel="${out#"$repo"/}"; fi
while IFS= read -r line; do
  path="${line:3}"
  if [[ -n "$out_rel" && ( "$path" == "$out_rel" || "$path" == "$out_rel"/* ) ]]; then continue; fi
  if ! printf '%s\n' "${patched[@]}" catalog/snapshot.json | grep -qxF -- "$path"; then
    echo "pr-job: unexpected change after the checks: $path" >&2
    exit 1
  fi
done < <(git status --porcelain --untracked-files=all)

# The files a PR commits: what the patch touched, plus the regenerated snapshot.
printf '%s\n' "${patched[@]}" catalog/snapshot.json >"$out/files.txt"
git diff -- catalog/snapshot.json >"$out/snapshot.diff"
# Non-data files go into the body's "Code changed" list, so reviewers read them as code.
for path in "${code[@]+"${code[@]}"}"; do echo "$path"; done >"$priv/code-files.txt"
cp catalog/snapshot.json "$priv/snapshot.json"
# The vendor's folder for the run's combined PR body, as the one-vendor body
# below reads it.
cp "$priv/base-snapshot.json" "$priv/snapshot.json" "$priv/code-files.txt" "$priv/base-routes.txt" "$priv/routes.txt" "$out/"
render -diff "$out/snapshot.diff" -code-files "$priv/code-files.txt" -base-routes "$priv/base-routes.txt" -routes "$priv/routes.txt"
emit changed=true
echo "pr-job: patch accepted"
