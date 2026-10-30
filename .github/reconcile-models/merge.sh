# shellcheck shell=bash
# Sourced by replay.sh and carry.sh. merge_onto PARENT BASE COMMIT prints the
# tree of COMMIT's change, taken from its own parent with a 3-way merge,
# applied to BASE: a merge stops at a conflict instead of landing on a nearby
# line. A conflict in catalog/snapshot.json alone doesn't count, since both
# scripts regenerate it. Any other conflict prints the conflicting paths to
# stderr and returns 1.
merge_onto() {
  local status=0 merged conflicts
  merged="$(git merge-tree --write-tree --name-only --no-messages --merge-base="$1" "$2" "$3")" || status=$?
  if ((status > 1)); then
    echo "git merge-tree failed" >&2
    return 2
  fi
  conflicts="$(tail -n +2 <<<"$merged" | grep -vx 'catalog/snapshot.json' || true)"
  if [[ -n "$conflicts" ]]; then
    paste -s -d ' ' - <<<"$conflicts" >&2
    return 1
  fi
  head -n 1 <<<"$merged"
}
