#!/usr/bin/env bash
# Replays a vendor's waiting commit from the open bot PR onto this checkout,
# the vendor's base in this run, with no model. It rebuilds the commit with
# a 3-way merge from the commit's own parent, so it stops at a conflict
# instead of landing on a nearby line, checks the result as pr-job.sh checks
# an agent's patch, and then runs catalog-replay, which passes only when the
# result is exactly the original change. On success OUT_DIR gets what an
# agent hands the check job (report.json, the saved report, and patch.diff),
# and it prints the original agent's run and time for $GITHUB_OUTPUT. Any
# failure exits 1, so the vendor's agent runs instead. Either way the
# checkout ends as it started.
# Usage: replay.sh VENDOR RECORD OUT_DIR CATALOG_REPLAY
#   RECORD is the record the last push left (plan.sh says when it counts).
# Env: PR_JOB, the checks to run (default: pr-job.sh next to this script).
set -euo pipefail

usage="usage: replay.sh VENDOR RECORD OUT_DIR CATALOG_REPLAY"
vendor="${1:?$usage}"
record="${2:?$usage}"
out="${3:?$usage}"
replay_bin="${4:?$usage}"
pr_job="${PR_JOB:-$(cd "$(dirname "$0")" && pwd)/pr-job.sh}"
emit() { if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "$1" >>"$GITHUB_OUTPUT"; fi; }
fail() { echo "replay: $*" >&2; exit 1; }

base="$(git rev-parse HEAD)"
work="$(mktemp -d)"
# A patch can only touch the providers and the catalogue, so only they need cleaning.
trap 'git reset -q --hard "$base"; git clean -fdq -- providers catalog; rm -rf "$work"' EXIT

entry="$(jq -c --arg v "$vendor" '.vendors[$v] // empty' "$record")"
[[ -n "$entry" ]] || fail "the record has no commit for $vendor"
bot="$(jq -r '.commit' <<<"$entry")"
parent="$(jq -r '.parent' <<<"$entry")"
[[ "$bot" =~ ^[0-9a-f]{40}$ && "$parent" =~ ^[0-9a-f]{40}$ ]] || fail "the record's commits for $vendor aren't SHAs"

git fetch -q --no-tags --depth=2 origin "$bot"
[[ "$(git rev-parse "$bot^")" == "$parent" ]] || fail "$bot isn't a commit on $parent"
# The snapshot is regenerated, so a conflict there alone doesn't count.
status=0
merged="$(git merge-tree --write-tree --name-only --no-messages --merge-base="$parent" "$base" "$bot")" || status=$?
(( status <= 1 )) || fail "git merge-tree failed"
tree="$(head -n 1 <<<"$merged")"
conflicts="$(tail -n +2 <<<"$merged" | grep -vx 'catalog/snapshot.json' || true)"
[[ -z "$conflicts" ]] || fail "$vendor's commit doesn't merge onto $base: $(paste -s -d ' ' - <<<"$conflicts")"

# The change as an agent hands it over: pr-job.sh regenerates the snapshot.
jq '.report' <<<"$entry" >"$work/report.json"
git diff --binary "$base" "$tree" -- . ':(exclude)catalog/snapshot.json' >"$work/patch.diff"
GITHUB_OUTPUT="" "$pr_job" "$vendor" "$work" || fail "$vendor's replayed change fails its checks"

# The replay as a commit, to compare with the original.
git add --pathspec-from-file="$work/files.txt"
git -c user.name=replay -c user.email=replay@localhost -c core.hooksPath=/dev/null commit -q -m "replay"
"$replay_bin" -parent "$parent" -bot "$bot" -base "$base" -replay HEAD -report "$work/report.json" -vendor "$vendor" ||
  fail "$vendor's replay isn't exactly its commit"

mkdir -p "$out"
cp "$work/report.json" "$work/patch.diff" "$out/"
emit "agent_run=$(jq -r '.agent_run' <<<"$entry")"
emit "agent_at=$(jq -r '.agent_at' <<<"$entry")"
echo "replay: $vendor's commit $bot replays exactly onto $base"
