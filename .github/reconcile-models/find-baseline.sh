#!/usr/bin/env bash
# Finds the newest fingerprints artifact that a successful real run of the
# reconcile workflow left on this branch: the gate's baseline. Only a real
# run's record job uploads one, and only a green run, where the pr job pushed
# too, counts. Prints run_id=ID for
# $GITHUB_OUTPUT, or nothing; the workflow step continues on any error, and
# no baseline means the agent runs.
# Env: NAME (artifact name), BRANCH (github.ref_name), SHA (github.sha),
#      REPO (owner/name), REPO_ID, WORKFLOW_PATH (.github/workflows/x.yml).
# With GH_TOKEN set it calls `gh api`; without, anonymous curl (for local use).
set -uo pipefail
api() {
  if [[ -n "${GH_TOKEN:-}" ]]; then gh api "$1"; else curl -sfL "https://api.github.com/$1"; fi
}

# Newest first; same name, same repository (not a fork), same branch, not expired.
candidates="$(api "repos/$REPO/actions/artifacts?name=$NAME&per_page=100" | jq -r \
  --arg branch "$BRANCH" --argjson repo "$REPO_ID" '
  [.artifacts[]
   | select(.expired | not)
   | select(.workflow_run.repository_id == $repo and .workflow_run.head_repository_id == $repo)
   | select(.workflow_run.head_branch == $branch)]
  | sort_by(.created_at) | reverse | .[:10][]
  | [.id, .workflow_run.id, .workflow_run.head_sha, .created_at] | @tsv')" || exit 0

while IFS=$'\t' read -r artifact_id run_id head_sha created_at; do
  [[ -n "$artifact_id" ]] || continue
  # The run must be this workflow file, started by hand or by the schedule
  # (never a pull_request run of a PR's own copy of the file), and finished green.
  ok="$(api "repos/$REPO/actions/runs/$run_id" | jq -r --arg path "$WORKFLOW_PATH" '
    .path == $path and (.event == "workflow_dispatch" or .event == "schedule")
    and .conclusion == "success"')" || continue
  [[ "$ok" == "true" ]] || { echo "skip run $run_id: not a green dispatch or schedule run of $WORKFLOW_PATH" >&2; continue; }
  # Its commit must be in this run's history, so a tag or fork that reuses the
  # branch name can't stand in for it.
  status="$(api "repos/$REPO/compare/$head_sha...$SHA" | jq -r .status)" || continue
  [[ "$status" == "ahead" || "$status" == "identical" ]] || { echo "skip run $run_id: $head_sha is $status" >&2; continue; }
  printf 'run_id=%s\n' "$run_id"
  exit 0
done <<<"$candidates"
