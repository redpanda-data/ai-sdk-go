#!/usr/bin/env bash
# Replays a vendor's waiting commit from the open bot PR onto this checkout,
# the vendor's base in this run, with no model. It rebuilds the commit with
# merge_onto (merge.sh), checks the result as pr-job.sh checks an agent's
# patch, and then runs catalog-replay, which passes only when the result is
# exactly the original change. On success OUT_DIR gets what an agent hands
# the check job (report.json, the saved report, and patch.diff), and it
# prints the original agent's run and time for $GITHUB_OUTPUT. Any failure
# exits 1, so the vendor's agent runs instead, as does a commit whose agent
# ran more than MAX_AGE_HOURS (default 192) ago, the plan's bound. It starts
# and ends on the clean base. It runs only before the agent, while the job
# holds nothing the agent wrote.
# Usage: replay.sh VENDOR RECORD OUT_DIR
#   RECORD is the record the last push left (plan.sh says when it counts).
# Env: NOW (RFC 3339, UTC; default now); PR_JOB, the checks to run (default:
#      pr-job.sh next to this script); CATALOG_REPLAY, a built catalog-replay
#      (default: built here).
set -euo pipefail

usage="usage: replay.sh VENDOR RECORD OUT_DIR"
vendor="${1:?$usage}"
record="${2:?$usage}"
out="${3:?$usage}"
here="$(cd "$(dirname "$0")" && pwd)"
pr_job="${PR_JOB:-$here/pr-job.sh}"
now="${NOW:-$(date -u +%FT%TZ)}"
emit() { if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "$1" >>"$GITHUB_OUTPUT"; fi; }
fail() { echo "replay: $*" >&2; exit 1; }
# shellcheck source=merge.sh
source "$here/merge.sh"

base="$(git rev-parse HEAD)"
work="$(mktemp -d)"
# Start and end on the clean base, after an agent's edits too. A patch can
# only touch the providers and the catalogue, so only they need cleaning.
restore() { git reset -q --hard "$base" && git clean -fdq -- providers catalog; }
trap 'restore; rm -rf "$work"' EXIT
restore

entry="$(jq -c --arg v "$vendor" '.vendors[$v] // empty' "$record")"
[[ -n "$entry" ]] || fail "the record has no commit for $vendor"
bot="$(jq -r '.commit' <<<"$entry")"
[[ "$bot" =~ ^[0-9a-f]{40}$ ]] || fail "the record's commit for $vendor isn't a SHA"
jq -e --arg now "$now" --argjson max "${MAX_AGE_HOURS:-192}" \
  'try ((($now | fromdateiso8601) - (.agent_at | fromdateiso8601)) / 3600 <= $max) catch false' <<<"$entry" >/dev/null ||
  fail "$vendor's commit came from an agent that ran more than ${MAX_AGE_HOURS:-192} hours ago"

replay_bin="${CATALOG_REPLAY:-$work/catalog-replay}"
if [[ -z "${CATALOG_REPLAY:-}" ]]; then
  go build -o "$replay_bin" ./cmd/catalog-replay
fi

git fetch -q --no-tags --depth=2 origin "$bot"
parent="$(git rev-parse "$bot^")"
tree="$(merge_onto "$parent" "$base" "$bot" 2>"$work/conflicts")" ||
  fail "$vendor's commit doesn't merge onto $base: $(cat "$work/conflicts")"

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
