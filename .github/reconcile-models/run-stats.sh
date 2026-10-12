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
    (split(" ") | map(select(length > 0) | gsub("[\"\u0027]"; ""))) as $words
    | ($words[0] // "" | sub(".*/"; "") | safe("^[a-z0-9._-]{1,24}$")) as $first
    | if ($first | IN("git", "go", "task", "gh")) then $first + " " + ($words[1] // "" | safe("^[a-z0-9:._-]{1,24}$"))
      elif $first == "curl" then "curl " + (($words | map(select(test("^https?://"))) | first | capture("^https?://(?<h>[^/?#]+)").h) // "" | safe("^[a-z0-9.-]{1,60}$"))
      else $first end;
  # A tool call named by its tool, and for Bash by the programs it runs.
  def call_name($tool; $command):
    ($tool | safe("^[A-Za-z0-9_]{1,40}$")) as $t
    | if $t == "Bash" then
        "Bash: " + ([$command // "" | splits("\\s*(\\|\\||&&|\\||;|\\n)\\s*") | select(length > 0) | program][:6] | join(" + "))
      else $t end;
  def counts: group_by(.) | map({name: .[0], n: length}) | sort_by(-.n, .name) | map("`\(.name)` (×\(.n))") | join(", ");
  [inputs] | map(if type == "array" then .[] else . end)
  | (map(select(.type == "result")) | last) as $r
  | ($r.permission_denials // [] | map(call_name(.tool_name; .tool_input.command))) as $denied
  | ([.[] | select(.type == "assistant") | .message.content[]? | select(.type == "tool_use") | call_name(.name; .input.command)]) as $calls
  | [
      "| Result | Turns | Cost (USD) | Duration (s) | Denied tool calls |",
      "|---|---|---|---|---|",
      "| \($r.subtype | safe("^[a-z_]{1,30}$")) | \($r.num_turns | num) | \($r.total_cost_usd | num) | \(($r.duration_ms // null) | if type == "number" then (. / 1000 | round | tostring) else "?" end) | \($denied | length) |",
      "",
      (if ($calls | length) > 0 then "Tool calls (\($calls | length)): " + ($calls | counts) else "No tool calls recorded." end),
      "",
      (if ($denied | length) > 0 then "Denied: " + ($denied | counts) else "No denied tool calls." end),
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
