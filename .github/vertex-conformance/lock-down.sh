#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd -P)"

"$here/../reconcile-models/stop-docker.sh"

# /etc/sudoers.d is readable only by root, so the glob must expand as root.
sudo sh -c 'rm -f /etc/sudoers.d/runner*'
if sudo -n true 2>/dev/null; then
  echo "::error::The runner user can still use sudo."
  exit 1
fi
echo "The runner user has no sudo."
