#!/usr/bin/env bash
# Stops Docker before Claude starts. Neither the agent nor the tests it runs
# need it, and its socket would give their code root on the runner, and so the
# model key: a Unix socket connection isn't network traffic, so Harden-Runner
# doesn't see it, and Claude Code's shell sandbox blocks Unix sockets only with
# an optional seccomp filter this job doesn't install.
set -euo pipefail
socket=/var/run/docker.sock

# A runner image without Docker has no units to stop; the check below decides.
sudo systemctl stop docker.socket docker.service || true

# 7 is a refused connection; anything else means something answers.
status=0
curl --silent --output /dev/null --max-time 5 --unix-socket "$socket" http://localhost/_ping || status=$?
if [[ "$status" -ne 7 ]]; then
  echo "::error::Something still answers on $socket (curl exit $status) after stopping Docker, so the agent's code could reach it."
  exit 1
fi
echo "Docker is stopped: nothing answers on $socket."
