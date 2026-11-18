#!/usr/bin/env bash
# Writes the message the notify job posts in the team's channel, with no
# model: the bot PR this run pushed, the jobs that failed, a run that stopped
# before the push, or the open questions of a run that pushed nothing. A run
# with none of these writes no message. A run from a branch other than main is
# a test: it posts only failures and stops, unless POST_TEST_PR asks it to post
# its test PR too, and every post it makes opens with a note saying so.
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

# Whether the PR and the questions go to the team.
team=true
if [[ "${GITHUB_REF_NAME:-}" != "main" ]]; then
  team="${POST_TEST_PR:-false}"
fi

lines=()

pr="$(need pr .outputs.pr)" pushed="$(need pr .outputs.pushed)"

# A job that ran past its time limit ends cancelled, not failed. Every job but
# plan, carry and pr is one vendor's.
failed="$(jq -c '[to_entries[] | select(.value.result == "failure" or .value.result == "cancelled") | .key]' <<<"$NEEDS")"
if [[ "$failed" != "[]" ]]; then
  # The failed jobs that pass a jq test of the job's name. A test can use
  # vendor, and checked: a vendor's job hands on a head once its check passed,
  # so one that failed later still has its changes in this run; one that
  # failed sooner doesn't.
  names() {
    jq -r --argjson failed "$failed" '. as $n
      | def vendor: IN("plan", "carry", "pr") | not;
        def checked: ($n[.].outputs.head // "") != "";
        [$failed[] | select('"$1"')] | join(", ")' <<<"$NEEDS"
  }
  unchecked="$(names 'vendor and (checked | not)')" late="$(names 'vendor and checked')"
  if jq -e 'any(. == "plan")' <<<"$failed" >/dev/null; then
    lines+=(":red_circle: The catalog reconcile ${run} failed before checking any vendor.")
  else
    line=":red_circle: The catalog reconcile ${run} failed"
    if [[ -n "$unchecked" ]]; then
      line+=" for: ${unchecked}. Their catalog entries may be out of date."
      if [[ -n "$late" ]]; then line+=" Failed after their changes were checked: ${late}."; fi
    elif [[ -n "$late" ]]; then
      line+=" for: ${late}, after their changes were checked."
    else
      line+="."
    fi
    # A vendor's failure mustn't hide one in the jobs that close the run.
    closing="$(names 'IN("carry", "pr")')"
    if [[ -n "$unchecked$late" && -n "$closing" ]]; then line+=" Also failed: ${closing}."; fi
    lines+=("$line")
  fi
  if [[ "$pushed" == "true" && -n "$unchecked$late" ]]; then
    # A re-run repeats the vendor but not the plan, which read the PR before
    # this run's push, so the pr job would refuse to push again. A new run
    # reads the PR as it is now.
    lines+=("To try again, <${repo_url}/actions/workflows/reconcile-models.yml|start a new run> for each failed vendor, with \"Stop before pushing\" unticked. Re-running this run's failed jobs can't update the PR again.")
  elif [[ "$(need pr .outputs.moved)" == "true" ]]; then
    # The pr job compares the PR with the head the plan read, and a re-run
    # keeps the plan's, so it would stop the same way.
    lines+=("Someone pushed to $(pr_link "${pr:-$(need plan .outputs.pr)}") during the run, so this run didn't update it. Re-running the failed jobs would stop the same way: <${repo_url}/actions/workflows/reconcile-models.yml|start a new run> with \"Stop before pushing\" unticked, or wait for the next scheduled run.")
  else
    # The open PR, as the pr job or else the plan found it, unless this run pushed to it.
    open="${pr:-$(need plan .outputs.pr)}" stale=""
    if [[ -n "$open" && "$pushed" != "true" ]]; then stale="$(pr_link "$open") wasn't updated. "; fi
    lines+=("${stale}Open the run and click \"Re-run failed jobs\" to try again.")
  fi
fi
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
  lines+=("Reconcile bot PR needs a reviewer: $(pr_link "$pr")" "")
  if [[ -n "$summary" ]]; then
    # A bullet for the changes, and one for each count of what didn't change.
    changed="${summary%% Not changed: *}"
    lines+=("• $(escape "${changed%.}")")
    if [[ "$summary" == *" Not changed: "* ]]; then
      unchanged="${summary#* Not changed: }"
      IFS=',' read -ra counts <<<"${unchanged%.}"
      for count in "${counts[@]}"; do lines+=("• $(escape "${count# }")"); done
    fi
    lines+=("")
  fi
  lines+=("To take it, react with :eyes: and add yourself as a reviewer.")
elif [[ "$(need pr .result)" == "success" && "$summary" == *"open question"* ]]; then
  # pushed is false when the PR already holds the same changes. It is empty
  # when the run had nothing to push, and also when the pr job left a closed
  # PR alone with changes it had: that run didn't push nothing, so it's quiet.
  if [[ "$pushed" == "false" && -n "$pr" ]]; then
    lines+=("The catalog reconcile ${run} found its changes already on $(pr_link "$pr"). $(escape "$summary")")
  elif [[ "$(need pr .outputs.changed)" != "true" ]]; then
    lines+=("The catalog reconcile ${run} pushed nothing. $(escape "$summary") The questions are in the run's summary.")
  fi
fi

if [[ ${#lines[@]} -eq 0 ]]; then
  emit "post=false"
  exit 0
fi

if [[ "${GITHUB_REF_NAME:-}" != "main" ]]; then
  lines=(":test_tube: *Test only:* this is a test run from \`$(escape "${GITHUB_REF_NAME:-}")\`, ignore this message." "${lines[@]}")
fi

printf '%s\n' "${lines[@]}" >"$message"
emit "post=true"
