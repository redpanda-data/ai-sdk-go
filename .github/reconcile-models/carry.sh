#!/usr/bin/env bash
# Carries the commits other people pushed onto the open bot PR, such as a
# fix or a swap of a live test model, onto this run's tip, in their order,
# with no model and no agent fallback. Each one is rebuilt with a 3-way
# merge from its own parent (a conflict outside the regenerated snapshot
# stops it), keeps its author and message, gets its snapshot regenerated in
# the sandbox, and must then be exactly the original change (catalog-replay,
# with no report). It never carries a merge commit (main reaches the PR when
# the run rebuilds it) or a commit that changes .github/. Last, the carried
# code's checks run in the sandbox. Any refusal exits 1 and prints the
# reason as refused= for the PR's note, so the run pushes nothing.
# Prints tip and carried (the new commits, as JSON) for $GITHUB_OUTPUT.
# Usage: carry.sh CATALOG_REPLAY SHA...
# Env: SANDBOX_SH (default: sandbox.sh next to this script); CARRY_SNAPSHOT
#      and CARRY_CHECK, the sandboxed commands that regenerate the snapshot
#      and check the result (defaults below).
set -euo pipefail

replay_bin="${1:?usage: carry.sh CATALOG_REPLAY SHA...}"
shift
emit() { if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "$1" >>"$GITHUB_OUTPUT"; fi; }
refuse() {
  emit "refused=$1"
  echo "carry: $1" >&2
  exit 1
}

for sha in "$@"; do
  [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || refuse "$sha isn't a commit SHA"
done

start="$(git rev-parse HEAD)"
git fetch -q --no-tags --depth=2 origin "$@"

# shellcheck source=sandbox.sh
source "${SANDBOX_SH:-$(dirname "$0")/sandbox.sh}"
sandbox_check || { echo "carry: the sandbox check failed" >&2; exit 1; }

empty="$(mktemp)"
trap 'rm -f "$empty"' EXIT
echo '{"changes": [], "needs_human": [], "skipped": []}' >"$empty"

carried=()
for sha in "$@"; do
  short="${sha:0:7}"
  [[ "$(git rev-list --no-walk --parents "$sha" | wc -w)" -eq 2 ]] ||
    refuse "$short is a merge commit, which is never carried: main reaches the PR when the run rebuilds it"
  parent="$(git rev-parse "$sha^")"
  [[ -z "$(git diff --name-only "$parent" "$sha" -- .github)" ]] || refuse "$short changes .github/, which is never carried"

  base="$(git rev-parse HEAD)"
  status=0
  merged="$(git merge-tree --write-tree --name-only --no-messages --merge-base="$parent" "$base" "$sha")" || status=$?
  (( status <= 1 )) || refuse "git merge-tree failed on $short"
  conflicts="$(tail -n +2 <<<"$merged" | grep -vx 'catalog/snapshot.json' || true)"
  [[ -z "$conflicts" ]] || refuse "$short doesn't merge onto this run's commits: $(paste -s -d ' ' - <<<"$conflicts")"

  # Their commit on the new base, by its author, with its message.
  commit="$(git log -1 --format=%B "$sha" |
    GIT_AUTHOR_NAME="$(git log -1 --format=%an "$sha")" GIT_AUTHOR_EMAIL="$(git log -1 --format=%ae "$sha")" \
      GIT_AUTHOR_DATE="$(git log -1 --format=%aD "$sha")" git commit-tree "$(head -n 1 <<<"$merged")" -p "$base")"
  git checkout -q --detach "$commit"
  sandbox bash -c "${CARRY_SNAPSHOT:-task catalog:snapshot}"
  if ! git diff --quiet -- catalog/snapshot.json; then
    git add catalog/snapshot.json
    git -c core.hooksPath=/dev/null commit -q --amend --no-edit
  fi

  "$replay_bin" -parent "$parent" -bot "$sha" -base "$base" -replay HEAD -report "$empty" -vendor carried ||
    refuse "$short isn't carried exactly onto this run's commits"
  carried+=("$(git rev-parse HEAD)")
done

# The carried code with this run's: the catalogue and every package it touches.
packages="$(git diff --name-only "$start" HEAD -- '*.go' | while read -r file; do echo "./$(dirname "$file")"; done | sort -u | tr '\n' ' ')"
sandbox bash -c "${CARRY_CHECK:-set -e; task catalog:check; go build ./...; ${packages:+go test -short -count=1 $packages}}" ||
  refuse "the carried commits fail their checks on this run's commits"

emit "tip=$(git rev-parse HEAD)"
emit "carried=$(printf '%s\n' "${carried[@]+"${carried[@]}"}" | jq -R . | jq -sc 'map(select(. != ""))')"
echo "carry: carried ${#carried[@]} commits onto $start"
