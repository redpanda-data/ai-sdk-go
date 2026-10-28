#!/usr/bin/env bash
# Exits 0 when two lists of commits make the same changes in the same order,
# commit for commit, by git patch-id (messages, authors and line numbers
# aside), and 1 otherwise. The push step uses it so a run whose commits are
# the PR's own, change for change, doesn't push and dismiss approvals.
# Usage: same-patches.sh OLD NEW
#   OLD and NEW are JSON lists of commit SHAs, oldest first, all in the checkout.
set -euo pipefail

old="${1:?usage: same-patches.sh OLD NEW}"
new="${2:?usage: same-patches.sh OLD NEW}"

ids() {
  local sha id
  for sha in $(jq -r '.[]' <<<"$1"); do
    git cat-file -e "$sha^{commit}" || return 1
    id="$(git show "$sha" | git patch-id --stable)" || return 1
    echo "${id%% *}"
  done
}

old_ids="$(ids "$old")" || exit 1
new_ids="$(ids "$new")" || exit 1
[[ "$(jq length <<<"$old")" == "$(jq length <<<"$new")" && "$old_ids" == "$new_ids" ]]
