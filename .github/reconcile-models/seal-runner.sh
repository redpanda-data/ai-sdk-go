#!/usr/bin/env bash
# Runs last before Claude starts. The agent runs code it wrote (Go tests via
# task), and that code can write anything in the checkout, .git included. The
# steps after Claude run outside its sandbox, so they must not execute what
# the agent could have changed, and nothing from then on may become root:
# root on the runner reads every secret of the job out of the runner's memory.
#
# So: copy the scripts and the git directory those steps use into a root-owned
# directory, then remove sudo for the rest of the job.
# Usage: seal-runner.sh [TRUSTED_DIR]
set -euo pipefail
trusted="${1:-/opt/reconcile-trusted}"
sudo=/usr/bin/sudo

"$sudo" rm -rf "$trusted"
"$sudo" install -d -o root -g root -m 0755 "$trusted"
"$sudo" install -o root -g root -m 0755 .github/reconcile-models/run-stats.sh "$trusted/run-stats.sh"
"$sudo" cp -a .git "$trusted/git"
"$sudo" chown -R root:root "$trusted/git"
"$sudo" chmod -R go-w,a+rX "$trusted/git"

# Hosted runners grant sudo through /etc/sudoers.d; the check below catches
# any other grant.
"$sudo" sh -c 'rm -f /etc/sudoers.d/*'
if "$sudo" -n true 2>/dev/null; then
  echo "::error::sudo still works after removing /etc/sudoers.d, so code the agent runs could become root."
  exit 1
fi
if [[ "$(stat -c %U "$trusted" "$trusted/run-stats.sh" "$trusted/git" | sort -u)" != root ]]; then
  echo "::error::$trusted isn't owned by root."
  exit 1
fi
echo "Sealed: trusted copies in $trusted, sudo removed."
