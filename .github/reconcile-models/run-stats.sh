#!/usr/bin/env bash
# Writes the agent run's stats to the job summary: numbers, fixed reasons and
# the names of denied programs only, never text the agent wrote, because this
# repository's run logs and summaries are public.
# Usage: run-stats.sh EXECUTION_FILE REPORT
set -euo pipefail

# EXECUTION_FILE may be empty: the action sets no path when Claude didn't run.
[[ $# -eq 2 ]] || { echo "usage: run-stats.sh EXECUTION_FILE REPORT" >&2; exit 2; }
execution="$1"
report="$2"
summary="${GITHUB_STEP_SUMMARY:-/dev/stdout}"

if [[ ! -s "$execution" ]]; then
  echo "### Agent run"$'\n\n'"No execution log: the agent step didn't finish." >>"$summary"
  exit 0
fi

# The execution log is a JSON array of messages, or one message per line;
# the last "result" message holds the totals and the permission denials.
stats="$(jq -n -r '
  def safe($re): if type == "string" and test($re) then . else "other" end;
  def num: if type == "number" then (. * 100 | round / 100 | tostring) else "?" end;
  def program:
    (split(" ") | map(select(length > 0))) as $words
    | ($words[0] // "" | sub(".*/"; "") | safe("^[a-z0-9._-]{1,24}$")) as $first
    | if ($first | IN("git", "go", "task", "gh")) then $first + " " + ($words[1] // "" | safe("^[a-z0-9:._-]{1,24}$")) else $first end;
  def denial:
    (.tool_name | safe("^[A-Za-z0-9_]{1,40}$")) as $tool
    | if $tool == "Bash" then
        "Bash: " + ([.tool_input.command // "" | splits("\\s*(\\|\\||&&|\\||;|\\n)\\s*") | select(length > 0) | program][:6] | join(" + "))
      else $tool end;
  [inputs] | map(if type == "array" then .[] else . end)
  | (map(select(.type == "result")) | last) as $r
  | ($r.permission_denials // [] | map(denial)) as $denied
  | [
      "| Result | Turns | Cost (USD) | Duration (s) | Denied tool calls |",
      "|---|---|---|---|---|",
      "| \($r.subtype | safe("^[a-z_]{1,30}$")) | \($r.num_turns | num) | \($r.total_cost_usd | num) | \(($r.duration_ms // null) | if type == "number" then (. / 1000 | round | tostring) else "?" end) | \($denied | length) |",
      "",
      (if ($denied | length) > 0 then "Denied: " + ($denied | group_by(.) | map("`\(.[0])` (×\(length))") | join(", ")) else "No denied tool calls." end),
      "@@DENIED@@" + ($denied | unique | join("; "))
    ] | .[]
' "$execution")"

denied="$(grep '^@@DENIED@@' <<<"$stats" | sed 's/^@@DENIED@@//')"
{
  echo "### Agent run"
  echo
  grep -v '^@@DENIED@@' <<<"$stats"
  echo
  echo "### Report"
  echo
  if [[ -s "$report" ]] && jq -e 'type == "object"' "$report" >/dev/null 2>&1; then
    jq -r '
      def safe($re): if type == "string" and test($re) then . else "other" end;
      "| Changes | Needs a human | Skipped |", "|---|---|---|",
      "| \(.changes // [] | length) | \(.needs_human // [] | length) | \(.skipped // [] | length)\(
        if (.skipped // [] | length) > 0
        then " (" + (.skipped | map(.reason | safe("^[a-z_]{1,40}$")) | group_by(.) | map("\(.[0]) \(length)") | join(", ")) + ")"
        else "" end) |"' "$report"
  else
    echo "No usable report."
  fi
} >>"$summary"

if [[ -n "$denied" ]]; then
  echo "::warning title=Denied tool calls::$denied"
fi
