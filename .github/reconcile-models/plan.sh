#!/usr/bin/env bash
# Plans a reconcile run before any agent starts: which vendors run their
# agent, which replay the commit they have waiting in the open bot PR (named
# by its Reconcile-Run trailer) with no model, and whether someone else's
# commit on the PR stops the run. Every run rebuilds the PR from its base,
# so a vendor whose commit is waiting must run or replay, or that change
# would drop out. A weekly run runs every agent.
# A replay needs the record the last push left: the head it pushed and each
# vendor's commit and agent run. It counts only when the PR's commits up to
# that head are exactly its vendors' commits. A vendor whose agent ran more
# than MAX_AGE_HOURS (default 192) ago runs its agent again. With no record,
# a real run runs each waiting vendor's agent, and a dry run only what it
# was asked for. Commits after the record's head are someone else's.
# Reads GitHub's list of the PR's commits (pulls/N/commits) on stdin, [] when
# there's no PR. Prints run (agent and replay, in order), agent, replay,
# pending, carry, theirs, stop and record (whether the record counts) for
# $GITHUB_OUTPUT.
# Usage: plan.sh BOT SELECTED DRY_RUN VENDORS_JSON [RECORD]
#   SELECTED is a vendor or "all"; VENDORS_JSON lists every vendor in order;
#   RECORD is the record's file, if the plan job found one.
# Env: NOW (RFC 3339, UTC; default now), MAX_AGE_HOURS.
set -euo pipefail

usage="usage: plan.sh BOT SELECTED DRY_RUN VENDORS_JSON [RECORD]"
bot="${1:?$usage}"
selected="${2:?$usage}"
dry_run="${3:?$usage}"
vendors="${4:?$usage}"
record_file="${5:-}"
now="${NOW:-$(date -u +%FT%TZ)}"
commits="$(cat)"

theirs="$("$(dirname "$0")/others-commits.sh" "$bot" <<<"$commits")"
pending="$(jq -c --arg bot "$bot" --argjson vendors "$vendors" '
  [.[] | select(.author.login == $bot and .committer.login == $bot)
   | .commit.message | capture("(?m)^Reconcile-Run: [0-9]+/(?<vendor>[a-z0-9]+)$").vendor] as $named
  | [$vendors[] | select(IN($named[]))]' <<<"$commits")"

record=null
if [[ -n "$record_file" && -s "$record_file" ]]; then
  record="$(jq -c --argjson commits "$commits" '
    . as $record | ($commits | map(.sha)) as $shas | ($shas | index($record.head)) as $at
    | if $at != null and ($shas[0:$at + 1] - [$record.vendors[].commit]) == [] then $record else null end' "$record_file" 2>/dev/null || echo null)"
fi
carry="$(jq -c --argjson commits "$commits" '
  if . == null then [] else . as $record | ($commits | map(.sha)) as $shas | $shas[($shas | index($record.head)) + 1:] end' <<<"$record")"

if [[ "$selected" == "all" ]]; then
  agent="$vendors"
elif jq -e --arg v "$selected" 'index($v) != null' <<<"$vendors" >/dev/null; then
  agent="[\"$selected\"]"
else
  echo "plan: no named sources for \"$selected\"" >&2
  exit 1
fi

replay="[]"
if [[ "$record" != "null" ]]; then
  # Each waiting vendor not running its agent replays, while its agent run is recent.
  fresh="$(jq -c --argjson now "$(jq -n --arg t "$now" '$t | fromdateiso8601')" --argjson max "${MAX_AGE_HOURS:-192}" \
    '[.vendors | to_entries[] | select((try (($now - (.value.agent_at | fromdateiso8601)) / 3600) catch 1e9) <= $max) | .key]' <<<"$record")"
  replay="$(jq -c --argjson agent "$agent" --argjson fresh "$fresh" '[.[] | select((IN($agent[]) | not) and IN($fresh[]))]' <<<"$pending")"
  agent="$(jq -c --argjson agent "$agent" --argjson replay "$replay" '$agent + [.[] | select(IN($replay[]) | not)] | unique' <<<"$pending")"
elif [[ "$dry_run" == "false" ]]; then
  agent="$(jq -c --argjson agent "$agent" '$agent + . | unique' <<<"$pending")"
fi

# In the chain's order.
in_order() { jq -c --argjson these "$1" '[.[] | select(IN($these[]))]' <<<"$vendors"; }
agent="$(in_order "$agent")"
replay="$(in_order "$replay")"
run="$(in_order "$(jq -c --argjson replay "$replay" '. + $replay' <<<"$agent")")"

# Someone else's commit stops a real run here, before any agent spends
# money: the push would remove it.
stop=false
if [[ "$dry_run" == "false" && ( -n "$theirs" || "$carry" != "[]" ) ]]; then
  stop=true
  run="[]" agent="[]" replay="[]"
fi

printf 'run=%s\nagent=%s\nreplay=%s\npending=%s\ncarry=%s\ntheirs=%s\nstop=%s\nrecord=%s\n' \
  "$run" "$agent" "$replay" "$pending" "$carry" "$theirs" "$stop" "$([[ "$record" != "null" ]] && echo true || echo false)"
