#!/usr/bin/env bash
# Writes the message the notify job posts in the team's channel, with no
# model: the bot PR this run pushed, the jobs that failed, a run that stopped
# before the push, or the open questions of a run that pushed nothing. A run
# with none of these writes no message. A run from a branch other than main is
# a test: it posts only failures and stops, unless POST_TEST_PR asks it to post
# its test PR too.
# Usage: notify.sh BODY_FILE MESSAGE_FILE
#   BODY_FILE is the PR body the pr job uploaded; it may be missing.
# Env: NEEDS (the notify job's needs, as JSON), POST_TEST_PR ("true" or not),
#      GITHUB_REF_NAME, GITHUB_SERVER_URL, GITHUB_REPOSITORY, GITHUB_RUN_ID.
#      Prints post for $GITHUB_OUTPUT.
set -euo pipefail

usage="usage: notify.sh BODY_FILE MESSAGE_FILE"
body="${1:?$usage}"
message="${2:?$usage}"
emit() { if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "$1" >>"$GITHUB_OUTPUT"; fi; }
need() { jq -r --arg j "$1" ".[\$j]$2 // empty" <<<"$NEEDS"; }
# Slack reads <, > and & as markup.
escape() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' <<<"$1"; }

repo_url="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}"
run="<${repo_url}/actions/runs/${GITHUB_RUN_ID}|run>"
pr_link() { echo "<${repo_url}/pull/$1|#$1>"; }

# The body's line that counts the changes and the open questions.
summary=""
if [[ -f "$body" ]]; then
  summary="$(grep -m 1 -E '^\*\*.+\*\* to the catalogue' "$body" | sed 's/\*\*//g' || true)"
fi

# Whether the PR and the questions go to the team, and how the PR is named.
team=true name="Catalog bot PR" from=""
if [[ "${GITHUB_REF_NAME:-}" != "main" ]]; then
  team="${POST_TEST_PR:-false}" name="Test bot PR" from=" from $(escape "${GITHUB_REF_NAME:-}")"
fi

lines=()

# A job that ran past its time limit ends cancelled, not failed.
failed="$(jq -r '[to_entries[] | select(.value.result == "failure" or .value.result == "cancelled") | .key] | join(", ")' <<<"$NEEDS")"
if [[ -n "$failed" ]]; then
  lines+=(":red_circle: The catalog reconcile ${run} failed in: ${failed}.")
fi

pr="$(need pr .outputs.pr)" pushed="$(need pr .outputs.pushed)"
# A commit on the PR that the run can't carry stops the push, as the pr job
# decided it.
if [[ "$(need pr .outputs.stop)" == "true" ]]; then
  target="the bot PR"
  if [[ -n "$pr" ]]; then target="$(pr_link "$pr")"; fi
  lines+=("The catalog reconcile ${run} didn't update ${target}, because $(escape "$(need pr .outputs.reason)").")
elif [[ "$team" != "true" ]]; then
  : # a test run's PR and questions stay off the channel
elif [[ "$pushed" == "true" && -z "$pr" ]]; then
  # The push worked and opening the PR didn't.
  lines+=("The catalog reconcile ${run} pushed to the bot branch, but no PR shows it yet.")
elif [[ "$pushed" == "true" ]]; then
  lines+=("${name} $(pr_link "$pr")${from} needs a reviewer. $(escape "$summary")")
  lines+=("To take it, react with :eyes: and add yourself as a reviewer. After two working days, the requested reviewer takes it.")
elif [[ "$summary" == *"open question"* ]]; then
  # pushed is false when the PR already holds the same changes, and empty
  # when the run had nothing to push.
  if [[ "$pushed" == "false" && -n "$pr" ]]; then
    lines+=("The catalog reconcile ${run} found its changes already on $(pr_link "$pr"). $(escape "$summary")")
  else
    lines+=("The catalog reconcile ${run} pushed nothing. $(escape "$summary") The questions are in the run's summary.")
  fi
fi

if [[ ${#lines[@]} -eq 0 ]]; then
  emit "post=false"
  exit 0
fi

printf '%s\n' "${lines[@]}" >"$message"
emit "post=true"
