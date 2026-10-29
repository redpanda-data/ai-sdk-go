#!/usr/bin/env bash
# Checks out the commit one vendor's part of a reconcile run works on: the
# run's own commit, or the previous vendor's. That one arrives as a git
# bundle, because nothing is pushed until the end of the run. Fails unless
# HEAD is then exactly that commit.
# Usage: checkout-base.sh BASE_SHA [BUNDLE]
set -euo pipefail

base="${1:?usage: checkout-base.sh BASE_SHA [BUNDLE]}"
bundle="${2:-}"

if [[ "$(git rev-parse HEAD)" == "$base" ]]; then
  exit 0
fi

[[ -n "$bundle" && -f "$bundle" ]] || { echo "checkout-base: no bundle for $base" >&2; exit 1; }
git fetch -q "$bundle" HEAD
[[ "$(git rev-parse FETCH_HEAD)" == "$base" ]] || { echo "checkout-base: the bundle holds $(git rev-parse FETCH_HEAD), not $base" >&2; exit 1; }
git checkout -q --detach "$base"
