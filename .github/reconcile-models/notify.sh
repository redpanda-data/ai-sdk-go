#!/usr/bin/env bash
# Writes the message the notify job posts in the team's channel, with no
# model: the bot PR this run pushed, the jobs that failed, a run that stopped
# before the push, or the open questions of a run that pushed nothing. A run
# with none of these writes no message. A run from a branch other than main is
# a test: it posts only failures and stops, unless POST_TEST_PR asks it to post
# its test PR too, and every post it makes opens with a note saying so.
# Every message has the same shape, so it reads at a glance: a bold heading
# that says what happened, a paragraph on what it means, the step to take when
# there is one, and the report's counts as a list.
# Usage: notify.sh BODY_FILE MESSAGE_FILE
#   BODY_FILE is the PR body the pr job uploaded; it may be missing.
# Env: NEEDS (the notify job's needs, as JSON), POST_TEST_PR ("true" or not),
#      GITHUB_REF_NAME, GITHUB_SERVER_URL, GITHUB_REPOSITORY, GITHUB_RUN_ID,
#      GITHUB_RUN_ATTEMPT. Prints post for $GITHUB_OUTPUT.
set -euo pipefail

usage="usage: notify.sh BODY_FILE MESSAGE_FILE"
body="${1:?$usage}"
message="${2:?$usage}"
emit() { if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "$1" >>"$GITHUB_OUTPUT"; fi; }
need() { jq -r --arg j "$1" ".[\$j]$2 // empty" <<<"$NEEDS"; }
# Slack reads <, > and & as markup.
escape() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' <<<"$1"; }

repo_url="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}"
run_url="${repo_url}/actions/runs/${GITHUB_RUN_ID}"
new_run="<${repo_url}/actions/workflows/reconcile-models.yml|start a new run>"
# This attempt's own summary, which a later re-run doesn't replace.
run_summary="<${run_url}/attempts/${GITHUB_RUN_ATTEMPT:-1}|run summary>"
pr_link() { echo "<${repo_url}/pull/$1|#$1>"; }

# The message's parts, in the order it shows them: a heading, which the first
# part to give one sets; the sentences on what happened, as one paragraph; the
# steps to take; and the report's counts.
heading=""
context=()
steps=()
counts=""
# Whether a failure's part already said what the run did to the bot PR.
told_pr=false
title() { if [[ -z "$heading" ]]; then heading="*Reconcile Bot - $1*${2:-}"; fi; }

# The body's line that counts the changes and the open questions.
summary=""
if [[ -f "$body" ]]; then
  summary="$(grep -m 1 -E '^\*\*.+\*\* to the catalogue' "$body" | sed 's/\*\*//g' || true)"
fi

# The words for a count of what didn't change. The PR body's short ones lean
# on its sections, which the channel doesn't show.
label() {
  sed -E -e 's/ held by a decision$/ report entry covered by a recorded decision/' \
    -e 's/ held by decisions$/ report entries covered by recorded decisions/' \
    -e 's/ skipped offering(s?)$/ model offering\1 skipped under catalog rules/' <<<"$1"
}
# The summary's counts as a list, below a line saying where the report is: a
# bullet for the changes, unless there are none, and one for each count of
# what didn't change. Rates a page only rounds stay in the report: they need
# its "Not asked: page rounding" list to make sense, and no one acts on them.
list() {
  if [[ -z "$summary" ]]; then return; fi
  counts="$1 has the report details."
  # The channel's word is catalog, as in the rest of the post.
  changed="${summary%% Not changed: *}" changed="${changed/ to the catalogue/ to the catalog}"
  if [[ "$changed" != "No changes"* ]]; then counts+=$'\n'"• $(escape "${changed%.}")"; fi
  if [[ "$summary" == *" Not changed: "* ]]; then
    unchanged="${summary#* Not changed: }"
    IFS=',' read -ra kept <<<"${unchanged%.}"
    for count in "${kept[@]}"; do
      count="${count# }"
      if [[ "$count" != *" the page rounds" ]]; then counts+=$'\n'"• $(escape "$(label "$count")")"; fi
    done
  fi
}

# Whether the PR and the questions go to the team.
team=true
if [[ "${GITHUB_REF_NAME:-}" != "main" ]]; then
  team="${POST_TEST_PR:-false}"
fi

pr="$(need pr .outputs.pr)" pushed="$(need pr .outputs.pushed)" stop="$(need pr .outputs.stop)"

# A job that ran past its time limit ends cancelled, not failed. Every job but
# plan, carry and pr is one vendor's.
failed="$(jq -c '[to_entries[] | select(.value.result == "failure" or .value.result == "cancelled") | .key]' <<<"$NEEDS")"
if [[ "$failed" != "[]" ]]; then
  # The failed jobs that pass a jq test of the job's name, as a JSON list. A
  # test can use vendor, and checked: a vendor's job hands on a head once its
  # check passed, so one that failed later still has its changes in this run;
  # one that failed sooner doesn't.
  names() {
    jq -c --argjson failed "$failed" '. as $n
      | def vendor: IN("plan", "carry", "pr") | not;
        def checked: ($n[.].outputs.head // "") != "";
        [$failed[] | select('"$1"')]' <<<"$NEEDS"
  }
  # Jobs as people say them, each with an optional ending such as "'s", joined
  # as "A", "A and B" or "A, B and C". A job not named here keeps its id.
  spoken() {
    jq -r --arg end "${2:-}" 'map(({anthropic: "Anthropic", openai: "OpenAI", google: "Google", amazon: "Amazon", meta: "Meta",
        mistral: "Mistral AI", zai: "Z.ai", minimax: "MiniMax", deepseek: "DeepSeek", moonshotai: "Moonshot AI",
        carry: "carrying people'"'"'s commits", pr: "updating the bot PR"}[.] // .) + $end)
      | if length > 1 then (.[:-1] | join(", ")) + " and " + .[-1] else join("") end' <<<"$1"
  }
  vendors="$(names vendor)" unchecked="$(names 'vendor and (checked | not)')" late="$(names 'vendor and checked')"
  who="$(spoken "$vendors")" retry="To retry" each=""
  if [[ -n "$who" ]]; then
    retry+=" the ${who} model catalog check"
    if [[ "$(jq length <<<"$vendors")" -gt 1 ]]; then retry+="s" each=" for each,"; fi
  fi
  plan_failed="$(jq 'any(. == "plan")' <<<"$failed")"
  what="catalog reconciliation failed before checking any vendor"
  if [[ "$plan_failed" != "true" ]]; then
    what="${who:+$who }catalog reconciliation failed"
    if [[ "$unchecked" != "[]" ]]; then context+=("$(spoken "$unchecked" "'s") catalog entries may be out of date."); fi
    if [[ "$late" != "[]" ]]; then context+=("$(spoken "$late" "'s") changes are in this run; a later step failed."); fi
    # A vendor's failure mustn't hide one in the jobs that close the run.
    closing="$(spoken "$(names 'IN("carry", "pr")')")"
    if [[ -n "$who" && -n "$closing" ]]; then context+=("Also failed: ${closing}."); fi
  fi
  title "$what" " :red_circle:"
  if [[ "$pushed" == "true" && -n "$who" ]]; then
    # A re-run repeats the vendor but not the plan, which read the PR before
    # this run's push, so the pr job would refuse to push again. A new run
    # reads the PR as it is now.
    steps+=("${retry}, ${new_run}${each} with \"Stop before pushing\" unticked. Re-running this run's failed jobs can't update the PR again.")
  elif [[ "$(need pr .outputs.moved)" == "true" ]]; then
    # The pr job compares the PR with the head the plan read, and a re-run
    # keeps the plan's, so it would stop the same way. It dropped every
    # vendor's changes, not only the failed ones', so the retry is a whole run.
    context+=("Someone pushed to $(pr_link "${pr:-$(need plan .outputs.pr)}") during the run, so this run didn't update it.")
    steps+=("To retry, ${new_run} with \"Stop before pushing\" unticked, or wait for the next scheduled run. Re-running the failed jobs would stop the same way.")
  else
    # What the run did to the open PR, as the pr job or else the plan found
    # it. A stopped run says it below, with its reason.
    if [[ "$plan_failed" != "true" && "$stop" != "true" ]]; then
      if [[ "$pushed" == "false" && -n "$pr" ]]; then
        context+=("This run's changes are already on $(pr_link "$pr").") told_pr=true
      elif [[ "$pushed" != "true" ]]; then
        open="${pr:-$(need plan .outputs.pr)}" line="The run pushed no catalog changes"
        if [[ -n "$open" ]]; then line+=", so $(pr_link "$open") wasn't updated"; fi
        context+=("${line}.") told_pr=true
      fi
    fi
    steps+=("${retry}, open the <${run_url}|GitHub Actions run> and choose \"Re-run failed jobs\".")
  fi
fi
# A commit on the PR that the run can't carry stops the push, as the pr job
# decided it.
if [[ "$stop" == "true" ]]; then
  target="the bot PR"
  if [[ -n "$pr" ]]; then target="$(pr_link "$pr")"; fi
  title "bot PR not updated"
  context+=("The run didn't update ${target}, because $(escape "$(need pr .outputs.reason)").")
  # The step the pr job's comment on the PR gives.
  steps+=("To let the next run update ${target}, merge or close it, or remove that commit if it was pushed after the bot's last update.")
elif [[ "$team" != "true" ]]; then
  : # a test run's PR and questions stay off the channel
elif [[ "$pushed" == "true" && -z "$pr" ]]; then
  # The push worked and opening the PR didn't.
  title "bot PR didn't open"
  context+=("The run pushed to the bot branch, but no PR shows it yet.")
elif [[ "$pushed" == "true" ]]; then
  take="react with :eyes: and add yourself as a reviewer."
  if [[ -z "$heading" ]]; then
    title "bot PR needs a reviewer"
    steps+=("To take $(pr_link "$pr"), ${take}")
  else
    steps+=("Bot PR $(pr_link "$pr") needs a reviewer. To take it, ${take}")
  fi
  list "The PR"
elif [[ "$(need pr .result)" == "success" && "$summary" == *"open question"* ]]; then
  # pushed is false when the PR already holds the same changes. It is empty
  # when the run had nothing to push, and also when the pr job left a closed
  # PR alone with changes it had: that run didn't push nothing, so it's quiet.
  if [[ "$pushed" == "false" && -n "$pr" ]]; then
    title "run has open questions"
    if [[ "$told_pr" != "true" ]]; then context+=("This run's changes are already on $(pr_link "$pr")."); fi
    list "The ${run_summary}"
  elif [[ "$(need pr .outputs.changed)" != "true" ]]; then
    title "run has open questions"
    if [[ "$told_pr" != "true" ]]; then context+=("The run pushed no catalog changes."); fi
    list "The ${run_summary}"
  fi
fi

if [[ -z "$heading" ]]; then
  emit "post=false"
  exit 0
fi

lines=()
if [[ "${GITHUB_REF_NAME:-}" != "main" ]]; then
  lines+=(":test_tube: *Test only:* this is a test run from \`$(escape "${GITHUB_REF_NAME:-}")\`, ignore this message.")
fi
lines+=("$heading")
if [[ ${#context[@]} -gt 0 ]]; then lines+=("" "${context[*]}"); fi
if [[ ${#steps[@]} -gt 0 ]]; then
  for step in "${steps[@]}"; do lines+=("" "$step"); done
fi
if [[ -n "$counts" ]]; then lines+=("" "$counts"); fi

printf '%s\n' "${lines[@]}" >"$message"
emit "post=true"
