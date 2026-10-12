#!/usr/bin/env bash
# Runs the reconcile-models agent step locally, with the same prompt, tools
# and schema as the workflow. Run it in a scratch worktree, not your main clone.
# Usage: run-agent-local.sh PROVIDER OUT_DIR
set -euo pipefail

provider="${1:?usage: run-agent-local.sh PROVIDER OUT_DIR}"
out="${2:?usage: run-agent-local.sh PROVIDER OUT_DIR}"
mkdir -p "$out"
sources=.github/reconcile-models/named-sources.json
hosts="$(jq -r --arg p "$provider" '.[$p].hosts | join(", ")' "$sources")"
owned="$(jq -r --arg p "$provider" '.[$p].providers | join(", ")' "$sources")"
edits="$(jq -r --arg p "$provider" '.[$p].providers | map("Edit(./providers/\(.)/*.go)") | join(",")' "$sources")"
prompt="$(sed -e "s/{{PROVIDER}}/${provider}/g" -e "s/{{HOSTS}}/${hosts}/g" -e "s/{{PROVIDERS}}/${owned}/g" .github/reconcile-models/prompt.md)"
schema="$(jq -c . .github/reconcile-models/report.schema.json)"

# An empty config dir gives the run what a CI runner sees: project settings and
# skills only, no user settings, plugins, MCP servers or CLAUDE.md, and no
# stored login: set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN.
config_dir="$(mktemp -d)"
trap 'rm -rf "$config_dir"' EXIT

CLAUDE_CONFIG_DIR="$config_dir" CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1 claude -p "$prompt" \
  --model claude-opus-5-5 \
  --max-turns 120 \
  --settings '{"disableAllHooks": true}' \
  --allowedTools "Skill,Read,Grep,Glob,${edits},Edit(./catalog/facts_data.go),Bash(curl:*),Bash(grep:*),Bash(jq:*),Bash(task catalog:snapshot),Bash(task test:unit),Bash(task lint:new),Bash(git diff:*)" \
  --disallowedTools "WebFetch,WebSearch,Bash(git push:*),Bash(git commit:*),Bash(gh:*),AskUserQuestion,Task,Agent" \
  --json-schema "$schema" \
  --output-format json >"$out/run.json"

jq '.structured_output' "$out/run.json" >"$out/report.json"
git ls-files --others --exclude-standard -- ":(glob)providers/*/*.go" | xargs -r git add -N --
git diff -- ":(glob)providers/*/*.go" catalog/facts_data.go >"$out/patch.diff"
jq '{num_turns, total_cost_usd, duration_ms, is_error}' "$out/run.json"
