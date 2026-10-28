#!/usr/bin/env bash
# Puts a reconcile run's vendor parts together into one PR, with no model:
# the last vendor commit is the PR's tip, checked out here for the push, and
# OUT_DIR gets the PR body for every vendor that ran, the files the vendors
# committed, the snapshot diff against the run's commit, and the record a
# later run replays from (record.json, when anything changed). A vendor whose
# part failed gets a note in the body; if it has a commit waiting in the PR,
# the run fails instead of pushing, or that change would drop out.
# Runs nothing from the patches. Prints tip and changed for $GITHUB_OUTPUT.
# Usage: combine.sh CATALOG_REPORT LEGS_DIR BUNDLES_DIR OUT_DIR
#   LEGS_DIR holds each vendor's reconcile-pr-<vendor> folder, BUNDLES_DIR its
#   reconcile-bundle-<commit> one.
# Env: NEEDS (the pr job's needs, as JSON), RUN and PENDING (from plan.sh),
#      GITHUB_SHA (the run's commit), BASE (the branch the PR goes into).
set -euo pipefail

usage="usage: combine.sh CATALOG_REPORT LEGS_DIR BUNDLES_DIR OUT_DIR"
report_bin="${1:?$usage}"
legs="${2:?$usage}"
bundles="${3:?$usage}"
out="${4:?$usage}"
emit() { if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "$1" >>"$GITHUB_OUTPUT"; fi; }
need() { jq -r --arg v "$1" ".[\$v]$2 // empty" <<<"$NEEDS"; }

mkdir -p "$out"
# The body's inputs from the run's commit, before the tip is checked out.
inputs="$(mktemp -d)"
trap 'rm -rf "$inputs"' EXIT
cp .github/reconcile-models/named-sources.json .github/reconcile-models/RULES.md .github/reconcile-models/decisions.yaml "$inputs/"
codes="$inputs/code-files.txt"
: >"$codes"
: >"$out/files.txt"

args=()
tip="$GITHUB_SHA" changed=false blocked=false first="" last="" recorded=()

for vendor in $(jq -r '.[]' <<<"$RUN"); do
  dir="$legs/reconcile-pr-$vendor"

  # A vendor's part counts once its check job finished, the only time it
  # hands on a head; a failed record job, which only keeps the gate's
  # baseline, doesn't drop a checked change.
  if [[ -z "$(need "$vendor" .outputs.head)" ]]; then
    args+=(-failed "$vendor")
    if jq -e --arg v "$vendor" 'index($v) != null' <<<"$PENDING" >/dev/null; then
      echo "combine: $vendor failed and has a commit waiting in the PR; pushing would drop it" >&2
      blocked=true
    fi
    continue
  fi

  args+=(-vendor "$vendor=$dir")
  if [[ "$(need "$vendor" .outputs.replayed)" == "true" ]]; then
    args+=(-replayed "$vendor=$(need "$vendor" .outputs.agent_run)@$(need "$vendor" .outputs.agent_at)")
  fi

  if [[ "$(need "$vendor" .outputs.changed)" == "true" ]]; then
    changed=true tip="$(need "$vendor" .outputs.head)" first="${first:-$dir}" last="$dir"
    recorded+=("$vendor")
    cat "$dir/files.txt" >>"$out/files.txt"
    cat "$dir/code-files.txt" >>"$codes"
  fi
done

if [[ "$blocked" == "true" ]]; then
  echo "::error::A vendor with a change waiting in the PR failed, so this run doesn't push; re-run the failed jobs."
  exit 1
fi

sort -u -o "$out/files.txt" "$out/files.txt"
sort -u -o "$codes" "$codes"

# The PR covers every vendor's commit, so its diff and routes run from the
# run's commit, before the first vendor's change, to the tip, after the last.
if [[ "$changed" == "true" ]]; then
  "$(dirname "$0")/checkout-base.sh" "$tip" "$bundles/reconcile-bundle-$tip/reconcile.bundle"
  git diff "$GITHUB_SHA" HEAD -- catalog/snapshot.json >"$out/snapshot.diff"
  args+=(-diff "$out/snapshot.diff" -code-files "$codes" -base-routes "$first/base-routes.txt" -routes "$last/routes.txt")

  # What the push holds: each vendor's commit, the commit it was made on,
  # its agent run and its report.
  vendors='{}'
  for vendor in "${recorded[@]}"; do
    head="$(need "$vendor" .outputs.head)"
    vendors="$(jq -c --arg v "$vendor" --arg commit "$head" --arg parent "$(git rev-parse "$head^")" \
      --arg run "$(need "$vendor" .outputs.agent_run)" --arg at "$(need "$vendor" .outputs.agent_at)" \
      --slurpfile report "$legs/reconcile-pr-$vendor/report.json" \
      '.[$v] = {commit: $commit, parent: $parent, agent_run: $run, agent_at: $at, report: $report[0]}' <<<"$vendors")"
  done
  jq -n --arg head "$tip" --argjson vendors "$vendors" '{head: $head, vendors: $vendors}' >"$out/record.json"
fi

# The PR body links repo files at the run's commit.
base_url=""
if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
  base_url="${GITHUB_SERVER_URL:-https://github.com}/$GITHUB_REPOSITORY/blob/$GITHUB_SHA"
fi

"$report_bin" "${args[@]}" -sources "$inputs/named-sources.json" -rules "$inputs/RULES.md" \
  -decisions "$inputs/decisions.yaml" -base-url "$base_url" -base-branch "${BASE:-main}" -out "$out/body.md"

emit "tip=$tip"
emit "changed=$changed"
