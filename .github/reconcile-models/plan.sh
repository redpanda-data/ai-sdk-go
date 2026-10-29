#!/usr/bin/env bash
# Plans a reconcile run before any agent starts: which vendors run, which
# have a commit waiting in the open bot PR (named by its Reconcile-Run
# trailer), and whether someone else's commit on the PR stops the run.
# Every run rebuilds the PR from its base, so a real run for some vendors
# also runs each vendor whose commit is waiting, or that change would drop
# out. A dry run pushes nothing, so it runs only what it was asked for.
# Reads GitHub's list of the PR's commits (pulls/N/commits) on stdin, [] when
# there's no PR. Prints run, pending, theirs and stop for $GITHUB_OUTPUT.
# Usage: plan.sh BOT SELECTED DRY_RUN VENDORS_JSON
#   SELECTED is a vendor or "all"; VENDORS_JSON lists every vendor in order.
set -euo pipefail

bot="${1:?usage: plan.sh BOT SELECTED DRY_RUN VENDORS_JSON}"
selected="${2:?usage: plan.sh BOT SELECTED DRY_RUN VENDORS_JSON}"
dry_run="${3:?usage: plan.sh BOT SELECTED DRY_RUN VENDORS_JSON}"
vendors="${4:?usage: plan.sh BOT SELECTED DRY_RUN VENDORS_JSON}"
commits="$(cat)"

theirs="$("$(dirname "$0")/others-commits.sh" "$bot" <<<"$commits")"
pending="$(jq -c --arg bot "$bot" --argjson vendors "$vendors" '
  [.[] | select(.author.login == $bot and .committer.login == $bot)
   | .commit.message | capture("(?m)^Reconcile-Run: [0-9]+/(?<vendor>[a-z0-9]+)$").vendor] as $named
  | [$vendors[] | select(IN($named[]))]' <<<"$commits")"

if [[ "$selected" == "all" ]]; then
  wanted="$vendors"
elif jq -e --arg v "$selected" 'index($v) != null' <<<"$vendors" >/dev/null; then
  wanted="[\"$selected\"]"
  if [[ "$dry_run" == "false" ]]; then
    wanted="$(jq -c --arg v "$selected" '. + [$v]' <<<"$pending")"
  fi
else
  echo "plan: no named sources for \"$selected\"" >&2
  exit 1
fi

# The vendors to run, in the chain's order.
run="$(jq -c --argjson wanted "$wanted" '[.[] | select(IN($wanted[]))]' <<<"$vendors")"

# Someone else's commit stops a real run here, before any agent spends
# money: the push would remove it.
stop=false
if [[ "$dry_run" == "false" && -n "$theirs" ]]; then
  stop=true
  run="[]"
fi

printf 'run=%s\npending=%s\ntheirs=%s\nstop=%s\n' "$run" "$pending" "$theirs" "$stop"
